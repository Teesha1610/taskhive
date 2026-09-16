"""Storage layer.

Four pragmas do most of the work of making SQLite usable for concurrent
writers:

- WAL lets readers run while a writer holds the log, so workers polling for
  work never block each other.
- busy_timeout makes a contended write wait instead of raising immediately,
  which is the difference between backpressure and a crash under load.
- synchronous=NORMAL is the right durability tradeoff under WAL: a process
  crash is safe, only an OS crash can lose the last commits.
- foreign_keys is off by default in SQLite, which surprises people.

Connections are per-thread because a sqlite3 connection is not safe to share
across threads, and the worker pool is threaded.
"""

from __future__ import annotations

import random
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from .errors import ConfigurationError

T = TypeVar("T")

SCHEMA_VERSION = 1

# Retries for a write or DDL statement that SQLite reports as busy.
BUSY_RETRIES = 12

# Backoff jitter, not a security primitive.
_rng = random.Random()  # noqa: S311


def is_busy(exc: sqlite3.OperationalError) -> bool:
    """True for the lock-contention errors worth retrying."""
    message = str(exc).lower()
    return "locked" in message or "busy" in message


def retry_on_busy(operation: Callable[[], T], retries: int = BUSY_RETRIES) -> T:
    """Run `operation`, retrying with jittered backoff while SQLite is busy.

    `busy_timeout` does not cover everything. Schema creation runs DDL, which
    takes an exclusive lock, and several processes opening the same database at
    once collide there before any queue operation happens. WAL adds its own
    case: a writer whose snapshot has moved on gets SQLITE_BUSY_SNAPSHOT
    immediately, with no wait, and the built-in busy handler does not retry it.
    """
    delay = 0.005
    for attempt in range(retries):
        try:
            return operation()
        except sqlite3.OperationalError as exc:
            if not is_busy(exc) or attempt == retries - 1:
                raise
            time.sleep(_rng.uniform(0, delay))
            delay = min(delay * 2, 0.25)
    raise AssertionError("unreachable")  # pragma: no cover

# RETURNING landed in SQLite 3.35. Below that the claim falls back to a
# select-then-update inside the same IMMEDIATE transaction, which is correct
# but costs an extra statement.
RETURNING_MIN_VERSION = (3, 35, 0)

MIGRATIONS: list[str] = [
    # 1: initial schema
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT    NOT NULL,
        queue         TEXT    NOT NULL DEFAULT 'default',
        payload       TEXT    NOT NULL DEFAULT '{}',
        state         TEXT    NOT NULL DEFAULT 'pending',
        priority      INTEGER NOT NULL DEFAULT 0,
        attempts      INTEGER NOT NULL DEFAULT 0,
        max_attempts  INTEGER NOT NULL DEFAULT 3,
        created_at    REAL    NOT NULL,
        available_at  REAL    NOT NULL,
        updated_at    REAL    NOT NULL,
        leased_until  REAL,
        lease_token   TEXT,
        worker_id     TEXT,
        started_at    REAL,
        finished_at   REAL,
        last_error    TEXT,
        result        TEXT,
        unique_key    TEXT,
        CHECK (state IN ('pending','running','succeeded','dead','cancelled')),
        CHECK (attempts >= 0),
        CHECK (max_attempts >= 1)
    );

    -- The claim query's access path: ready work in one queue, best first.
    CREATE INDEX IF NOT EXISTS idx_tasks_claim
        ON tasks (queue, state, priority DESC, available_at, id);

    -- Reclaiming expired leases scans only running rows.
    CREATE INDEX IF NOT EXISTS idx_tasks_leases
        ON tasks (state, leased_until);

    -- Idempotency: at most one live task per key. Terminal rows are excluded
    -- so the same key can be enqueued again once the first one finished.
    CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_unique_key
        ON tasks (unique_key)
        WHERE unique_key IS NOT NULL AND state IN ('pending','running');

    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER NOT NULL
    );
    """,
]


def supports_returning() -> bool:
    parts = tuple(int(p) for p in sqlite3.sqlite_version.split("."))
    return parts >= RETURNING_MIN_VERSION


def connect(path: str | Path, timeout: float = 10.0) -> sqlite3.Connection:
    """Open a tuned connection. `:memory:` is supported for tests."""
    database = str(path)
    if database != ":memory:":
        Path(database).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(database, timeout=timeout, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # PRAGMA does not accept bound parameters, so the int is formatted in.
    # Set busy_timeout first: it governs how the statements below wait.
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")

    if database != ":memory:":
        # Switching a database into WAL takes a brief exclusive lock, so several
        # connections opening the same file at once contend here, before a
        # single query has run. busy_timeout does not cover it, and the failure
        # surfaces as "database is locked" from what looks like a trivial
        # setup line. Reading the mode first means the lock is only taken when
        # the switch is actually needed, which is once in the file's lifetime.
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        if str(mode).lower() != "wal":
            retry_on_busy(lambda: conn.execute("PRAGMA journal_mode = WAL"))
        conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations. Returns the resulting schema version.

    Wrapped in the busy retry: opening a queue runs DDL, and several threads or
    processes constructing a TaskQueue at the same moment contend on an
    exclusive lock before any task has been enqueued. That is the first thing a
    concurrent workload does, so it has to be the first thing that is safe.
    """
    retry_on_busy(lambda: conn.executescript(MIGRATIONS[0]))
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    current = row["version"] if row else 0

    if current > SCHEMA_VERSION:
        raise ConfigurationError(
            f"database is at schema version {current}, this build understands {SCHEMA_VERSION}. "
            "Upgrade taskhive rather than downgrading the database."
        )

    for index in range(current, len(MIGRATIONS)):
        if index > 0:  # migration 0 already ran above
            retry_on_busy(lambda i=index: conn.executescript(MIGRATIONS[i]))  # type: ignore[misc]

    if row is None:
        retry_on_busy(
            lambda: conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        )
    elif current < SCHEMA_VERSION:
        retry_on_busy(lambda: conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,)))
    return SCHEMA_VERSION


class ConnectionPool:
    """One connection per thread, created lazily and reused.

    sqlite3 connections are not thread safe, and check_same_thread=False only
    silences the guard rather than making sharing correct. A thread-local is
    both simpler and actually safe.
    """

    def __init__(self, path: str | Path, timeout: float = 10.0) -> None:
        self.path = str(path)
        self.timeout = timeout
        self._local = threading.local()
        self._shared: sqlite3.Connection | None = None
        self._lock = threading.Lock()

        if self.path == ":memory:":
            # An in-memory database exists only inside its own connection, so
            # every thread must share one, guarded by a lock.
            self._shared = connect(self.path, timeout)
            migrate(self._shared)
        else:
            migrate(self.get())

    def get(self) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self.path, self.timeout)
            self._local.conn = conn
        return conn

    @property
    def serialized(self) -> bool:
        """True when callers must hold the lock, i.e. shared in-memory mode."""
        return self._shared is not None

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def close(self) -> None:
        if self._shared is not None:
            self._shared.close()
            self._shared = None
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
