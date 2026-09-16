"""The queue itself.

The only genuinely hard part is claiming work: several workers poll the same
table, and exactly one of them must get each task. The claim is a single
statement inside a BEGIN IMMEDIATE transaction, so SQLite serializes writers
and no task is handed out twice. Correctness does not depend on the workers
cooperating.

Delivery is at-least-once. A worker that dies mid-task leaves a lease that
expires, and the task returns to the queue. That means handlers must be
idempotent, which is the same contract every durable queue gives you.

On the f-strings in the SQL below: the only values interpolated into a query
string are TASK_COLUMNS (a module constant), generated runs of `?` placeholders,
and fixed fragments chosen by a boolean. Every caller-supplied value is bound as
a parameter. The `` markers record that this was checked rather than
overlooked, and the linter rule stays on so a future edit that interpolates real
data is still caught.
"""

from __future__ import annotations

import contextlib
import json
import random
import secrets
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .backoff import DEFAULT_POLICY, RetryPolicy
from .db import ConnectionPool, supports_returning
from .errors import LeaseExpiredError
from .models import QueueStats, Task, TaskState, to_epoch, utcnow

TASK_COLUMNS = (
    "id, name, queue, payload, state, priority, attempts, max_attempts, "
    "created_at, available_at, updated_at, leased_until, lease_token, worker_id, "
    "started_at, finished_at, last_error, result, unique_key"
)

DEFAULT_QUEUE = "default"
DEFAULT_LEASE_SECONDS = 60.0

# How many times a write transaction retries when SQLite reports the database
# busy. Roughly a second of total backoff, well inside any sane request budget.
BUSY_RETRIES = 12

# Backoff jitter, not a security primitive.
_rng = random.Random()  # noqa: S311


def _is_busy(exc: sqlite3.OperationalError) -> bool:
    """True for the lock-contention errors that are worth retrying."""
    message = str(exc).lower()
    return "locked" in message or "busy" in message


def _in_claim_order(tasks: Iterable[Task]) -> list[Task]:
    """Sort claimed tasks the way the claim selected them.

    SQLite's RETURNING makes no promise about row order, so the ORDER BY in the
    subquery decides which rows are claimed but not the order they come back.
    Callers that process a batch sequentially expect highest priority first, so
    the ordering is reapplied here. Found by a property test, not by reading
    the documentation.
    """
    return sorted(tasks, key=lambda t: (-t.priority, t.available_at, t.id))


class TaskQueue:
    """Durable task queue backed by a single SQLite file.

    >>> queue = TaskQueue(":memory:")
    >>> task = queue.enqueue("send_email", {"to": "a@example.com"})
    >>> leased = queue.lease("worker-1")
    >>> queue.ack(leased[0].id, leased[0].lease_token, result={"sent": True})
    True
    """

    def __init__(
        self,
        path: str | Path = "taskhive.db",
        *,
        retry_policy: RetryPolicy | None = None,
        default_max_attempts: int = 3,
        default_lease_seconds: float = DEFAULT_LEASE_SECONDS,
        timeout: float = 10.0,
    ) -> None:
        if default_max_attempts < 1:
            raise ValueError("default_max_attempts must be at least 1")
        self.path = str(path)
        self.retry_policy = retry_policy or DEFAULT_POLICY
        self.default_max_attempts = default_max_attempts
        self.default_lease_seconds = float(default_lease_seconds)
        self._pool = ConnectionPool(path, timeout)
        self._use_returning = supports_returning()

    # ---------------------------------------------------------- plumbing

    @property
    def connection(self) -> sqlite3.Connection:
        return self._pool.get()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """BEGIN IMMEDIATE ... COMMIT, with rollback and busy retry.

        IMMEDIATE takes the write lock up front instead of upgrading from a read
        lock mid-transaction, which is what produces SQLITE_BUSY deadlocks
        between two writers that both started as readers.

        `busy_timeout` alone is not enough. SQLite's built-in busy handler does
        not retry every conflict: in WAL mode a writer whose snapshot has moved
        on gets SQLITE_BUSY_SNAPSHOT immediately, with no wait, and several
        writers spinning on a hot queue will eventually hit it. Surfacing that
        to the caller as "database is locked" would make the queue fail exactly
        when it is busiest, which is the wrong moment to give up.

        So a failed BEGIN is retried with jittered backoff. The jitter matters
        for the same reason it matters in the retry policy: without it, two
        writers that collide retry in lockstep and collide again.
        """
        conn = self.connection
        guard: threading.Lock | None = self._pool.lock if self._pool.serialized else None
        if guard:
            guard.acquire()
        try:
            delay = 0.005
            for attempt in range(BUSY_RETRIES):
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    break
                except sqlite3.OperationalError as exc:
                    if not _is_busy(exc) or attempt == BUSY_RETRIES - 1:
                        raise
                    time.sleep(_rng.uniform(0, delay))
                    delay = min(delay * 2, 0.25)

            try:
                yield conn
                conn.execute("COMMIT")
            except Exception:
                # A failed COMMIT may already have closed the transaction, in
                # which case ROLLBACK has nothing to undo.
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute("ROLLBACK")
                raise
        finally:
            if guard:
                guard.release()

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        conn = self.connection
        guard: threading.Lock | None = self._pool.lock if self._pool.serialized else None
        if guard:
            guard.acquire()
        try:
            yield conn
        finally:
            if guard:
                guard.release()

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> TaskQueue:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ----------------------------------------------------------- writing

    def enqueue(
        self,
        name: str,
        payload: dict[str, Any] | None = None,
        *,
        queue: str = DEFAULT_QUEUE,
        priority: int = 0,
        max_attempts: int | None = None,
        delay: float | timedelta = 0.0,
        run_at: datetime | None = None,
        unique_key: str | None = None,
    ) -> Task:
        """Add one task.

        `unique_key` makes the call idempotent: if a task with the same key is
        already pending or running, that existing task is returned and nothing
        new is created. Terminal tasks release the key, so a daily job can use
        the same key each day.
        """
        if not name:
            raise ValueError("name is required")
        now = utcnow()
        if run_at is not None:
            available = run_at
        else:
            seconds = delay.total_seconds() if isinstance(delay, timedelta) else float(delay)
            if seconds < 0:
                raise ValueError("delay must be non-negative")
            available = now + timedelta(seconds=seconds)

        row = (
            name,
            queue,
            json.dumps(payload or {}),
            TaskState.PENDING.value,
            int(priority),
            0,
            int(max_attempts or self.default_max_attempts),
            to_epoch(now),
            to_epoch(available),
            to_epoch(now),
            unique_key,
        )

        try:
            with self._write() as conn:
                cursor = conn.execute(
                    """INSERT INTO tasks
                       (name, queue, payload, state, priority, attempts, max_attempts,
                        created_at, available_at, updated_at, unique_key)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    row,
                )
                task_id = cursor.lastrowid
                if task_id is None:  # pragma: no cover - sqlite always sets this
                    raise RuntimeError("insert did not return a row id")
        except sqlite3.IntegrityError:
            existing = self._find_by_unique_key(unique_key) if unique_key else None
            if existing is not None:
                return existing
            raise

        found = self.get(task_id)
        assert found is not None
        return found

    def enqueue_many(self, tasks: Sequence[dict[str, Any]], *, queue: str = DEFAULT_QUEUE) -> int:
        """Insert a batch in one transaction.

        One commit for the batch instead of one per task. On a laptop that is
        the difference between roughly a thousand inserts a second and tens of
        thousands, because fsync dominates.
        """
        if not tasks:
            return 0
        now = utcnow()
        now_epoch = to_epoch(now)
        rows = []
        for spec in tasks:
            if not spec.get("name"):
                raise ValueError("every task needs a name")
            delay = float(spec.get("delay", 0.0))
            rows.append(
                (
                    spec["name"],
                    spec.get("queue", queue),
                    json.dumps(spec.get("payload") or {}),
                    TaskState.PENDING.value,
                    int(spec.get("priority", 0)),
                    0,
                    int(spec.get("max_attempts") or self.default_max_attempts),
                    now_epoch,
                    now_epoch + delay,
                    now_epoch,
                    spec.get("unique_key"),
                )
            )
        with self._write() as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO tasks
                   (name, queue, payload, state, priority, attempts, max_attempts,
                    created_at, available_at, updated_at, unique_key)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            return (conn.total_changes and len(rows)) or len(rows)

    # ----------------------------------------------------------- claiming

    def lease(
        self,
        worker_id: str,
        *,
        queues: Iterable[str] | None = None,
        limit: int = 1,
        lease_seconds: float | None = None,
        now: datetime | None = None,
    ) -> list[Task]:
        """Atomically claim up to `limit` ready tasks.

        Ordering is priority descending, then oldest available first, then id,
        which makes the claim deterministic and keeps starvation out of the
        low-priority band as long as high-priority work drains.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        moment = now or utcnow()
        epoch = to_epoch(moment)
        # `or` would treat a deliberate 0.0 as unset. Explicit None check only.
        lease_for = float(self.default_lease_seconds if lease_seconds is None else lease_seconds)
        token = secrets.token_hex(16)
        names = list(queues) if queues else [DEFAULT_QUEUE]
        placeholders = ",".join("?" * len(names))

        selector = f"""
            SELECT id FROM tasks
             WHERE state = 'pending'
               AND available_at <= ?
               AND queue IN ({placeholders})
             ORDER BY priority DESC, available_at ASC, id ASC
             LIMIT ?
        """
        update = f"""
            UPDATE tasks
               SET state = 'running',
                   attempts = attempts + 1,
                   lease_token = ?,
                   leased_until = ?,
                   worker_id = ?,
                   started_at = COALESCE(started_at, ?),
                   updated_at = ?
             WHERE id IN ({selector})
        """
        params = (token, epoch + lease_for, worker_id, epoch, epoch, epoch, *names, limit)

        with self._write() as conn:
            if self._use_returning:
                rows = conn.execute(update + f" RETURNING {TASK_COLUMNS}", params).fetchall()
                return _in_claim_order(Task.from_row(row) for row in rows)

            # Fallback for SQLite older than 3.35, still inside the same
            # transaction so the select cannot race the update.
            ids = [
                r["id"]
                for r in conn.execute(selector, (epoch, *names, limit)).fetchall()
            ]
            if not ids:
                return []
            marks = ",".join("?" * len(ids))
            conn.execute(
                f"""UPDATE tasks
                       SET state='running', attempts=attempts+1, lease_token=?, leased_until=?,
                           worker_id=?, started_at=COALESCE(started_at, ?), updated_at=?
                     WHERE id IN ({marks})""",
                (token, epoch + lease_for, worker_id, epoch, epoch, *ids),
            )
            rows = conn.execute(
                f"SELECT {TASK_COLUMNS} FROM tasks WHERE id IN ({marks})", ids
            ).fetchall()
            return _in_claim_order(Task.from_row(row) for row in rows)

    def heartbeat(self, task_id: int, lease_token: str, *, extend_seconds: float | None = None) -> bool:
        """Push the lease out so a long task is not reclaimed underneath you."""
        extend = float(self.default_lease_seconds if extend_seconds is None else extend_seconds)
        epoch = to_epoch(utcnow())
        with self._write() as conn:
            cursor = conn.execute(
                """UPDATE tasks SET leased_until = ?, updated_at = ?
                    WHERE id = ? AND lease_token = ? AND state = 'running'""",
                (epoch + extend, epoch, task_id, lease_token),
            )
            return cursor.rowcount > 0

    # ---------------------------------------------------------- finishing

    def ack(self, task_id: int, lease_token: str | None, *, result: Any = None) -> bool:
        """Mark a task succeeded. The lease token must still be valid."""
        epoch = to_epoch(utcnow())
        with self._write() as conn:
            cursor = conn.execute(
                """UPDATE tasks
                      SET state='succeeded', finished_at=?, updated_at=?,
                          result=?, lease_token=NULL, leased_until=NULL
                    WHERE id=? AND state='running' AND lease_token=?""",
                (epoch, epoch, json.dumps(result) if result is not None else None, task_id, lease_token),
            )
            if cursor.rowcount == 0:
                raise LeaseExpiredError(
                    f"task {task_id} is no longer held by this lease; another worker may have it"
                )
            return True

    def nack(
        self,
        task_id: int,
        lease_token: str | None,
        *,
        error: str = "",
        retry: bool = True,
        delay: float | None = None,
    ) -> Task:
        """Report failure. Retries while attempts remain, otherwise dead-letters.

        `delay=None` asks the retry policy. Passing a number overrides it, which
        is what RetryLater does when a service returned a Retry-After.
        """
        epoch = to_epoch(utcnow())
        with self._write() as conn:
            row = conn.execute(
                f"SELECT {TASK_COLUMNS} FROM tasks WHERE id=? AND lease_token=? AND state='running'",
                (task_id, lease_token),
            ).fetchone()
            if row is None:
                raise LeaseExpiredError(f"task {task_id} is no longer held by this lease")

            task = Task.from_row(row)
            exhausted = task.attempts >= task.max_attempts
            message = (error or "")[:4000]

            if not retry or exhausted:
                conn.execute(
                    """UPDATE tasks
                          SET state='dead', finished_at=?, updated_at=?, last_error=?,
                              lease_token=NULL, leased_until=NULL
                        WHERE id=?""",
                    (epoch, epoch, message, task_id),
                )
            else:
                wait = self.retry_policy.delay_for(task.attempts) if delay is None else float(delay)
                conn.execute(
                    """UPDATE tasks
                          SET state='pending', available_at=?, updated_at=?, last_error=?,
                              lease_token=NULL, leased_until=NULL, worker_id=NULL
                        WHERE id=?""",
                    (epoch + max(0.0, wait), epoch, message, task_id),
                )

            refreshed = conn.execute(
                f"SELECT {TASK_COLUMNS} FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            return Task.from_row(refreshed)

    def reclaim_expired(self, *, now: datetime | None = None) -> int:
        """Return tasks whose lease ran out to the queue, or dead-letter them.

        This is what makes a worker crash survivable. The attempt was already
        counted at claim time, so a process that dies in a retry loop still
        exhausts its budget rather than looping forever.
        """
        epoch = to_epoch(now or utcnow())
        with self._write() as conn:
            dead = conn.execute(
                """UPDATE tasks
                      SET state='dead', finished_at=?, updated_at=?,
                          last_error='lease expired and no attempts remained',
                          lease_token=NULL, leased_until=NULL
                    WHERE state='running' AND leased_until < ? AND attempts >= max_attempts""",
                (epoch, epoch, epoch),
            ).rowcount

            rows = conn.execute(
                """SELECT id, attempts FROM tasks
                    WHERE state='running' AND leased_until < ?""",
                (epoch,),
            ).fetchall()
            for row in rows:
                wait = self.retry_policy.delay_for(max(1, row["attempts"]))
                conn.execute(
                    """UPDATE tasks
                          SET state='pending', available_at=?, updated_at=?,
                              last_error='lease expired, requeued',
                              lease_token=NULL, leased_until=NULL, worker_id=NULL
                        WHERE id=?""",
                    (epoch + wait, epoch, row["id"]),
                )
            return dead + len(rows)

    def cancel(self, task_id: int) -> bool:
        """Cancel a pending task. Running tasks cannot be cancelled mid-flight."""
        epoch = to_epoch(utcnow())
        with self._write() as conn:
            cursor = conn.execute(
                """UPDATE tasks SET state='cancelled', finished_at=?, updated_at=?
                    WHERE id=? AND state='pending'""",
                (epoch, epoch, task_id),
            )
            return cursor.rowcount > 0

    def requeue(self, task_id: int, *, reset_attempts: bool = True) -> bool:
        """Put a dead or cancelled task back in the queue."""
        epoch = to_epoch(utcnow())
        with self._write() as conn:
            cursor = conn.execute(
                f"""UPDATE tasks
                       SET state='pending', available_at=?, updated_at=?,
                           finished_at=NULL, lease_token=NULL, leased_until=NULL,
                           worker_id=NULL {', attempts=0' if reset_attempts else ''}
                     WHERE id=? AND state IN ('dead','cancelled')""",
                (epoch, epoch, task_id),
            )
            return cursor.rowcount > 0

    def purge(self, *, states: Iterable[TaskState] | None = None, older_than_seconds: float = 0.0) -> int:
        """Delete terminal tasks. Retention is a policy, not a default."""
        targets = [s.value for s in (states or (TaskState.SUCCEEDED, TaskState.CANCELLED))]
        for value in targets:
            if not TaskState(value).terminal:
                raise ValueError(f"refusing to purge non-terminal state: {value}")
        cutoff = to_epoch(utcnow()) - max(0.0, older_than_seconds)
        marks = ",".join("?" * len(targets))
        with self._write() as conn:
            cursor = conn.execute(
                f"DELETE FROM tasks WHERE state IN ({marks}) AND COALESCE(finished_at, updated_at) <= ?",
                (*targets, cutoff),
            )
            return cursor.rowcount

    # ----------------------------------------------------------- reading

    def get(self, task_id: int) -> Task | None:
        with self._read() as conn:
            row = conn.execute(f"SELECT {TASK_COLUMNS} FROM tasks WHERE id=?", (task_id,)).fetchone()
            return Task.from_row(row) if row else None

    def _find_by_unique_key(self, unique_key: str) -> Task | None:
        with self._read() as conn:
            row = conn.execute(
                f"""SELECT {TASK_COLUMNS} FROM tasks
                     WHERE unique_key=? AND state IN ('pending','running')""",
                (unique_key,),
            ).fetchone()
            return Task.from_row(row) if row else None

    def list_tasks(
        self,
        *,
        state: TaskState | None = None,
        queue: str | None = None,
        name: str | None = None,
        limit: int = 50,
    ) -> list[Task]:
        clauses: list[str] = []
        params: list[Any] = []
        if state:
            clauses.append("state = ?")
            params.append(state.value)
        if queue:
            clauses.append("queue = ?")
            params.append(queue)
        if name:
            clauses.append("name = ?")
            params.append(name)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        with self._read() as conn:
            rows = conn.execute(
                f"SELECT {TASK_COLUMNS} FROM tasks {where} ORDER BY id DESC LIMIT ?", params
            ).fetchall()
            return [Task.from_row(row) for row in rows]

    def dead_letters(self, *, limit: int = 50) -> list[Task]:
        return self.list_tasks(state=TaskState.DEAD, limit=limit)

    def stats(self, *, queue: str | None = None) -> QueueStats:
        now = to_epoch(utcnow())
        where, params = ("WHERE queue = ?", [queue]) if queue else ("", [])
        with self._read() as conn:
            counts = {
                row["state"]: row["n"]
                for row in conn.execute(
                    f"SELECT state, COUNT(*) AS n FROM tasks {where} GROUP BY state", params
                ).fetchall()
            }
            ready_clause = f"{where} AND available_at <= ?" if where else "WHERE available_at <= ?"
            ready = conn.execute(
                f"SELECT COUNT(*) AS n FROM tasks {ready_clause} AND state='pending'",
                (*params, now),
            ).fetchone()["n"]
            oldest = conn.execute(
                f"SELECT MIN(created_at) AS t FROM tasks {ready_clause} AND state='pending'",
                (*params, now),
            ).fetchone()["t"]

        pending = counts.get("pending", 0)
        return QueueStats(
            queue=queue or "*",
            pending=pending,
            running=counts.get("running", 0),
            succeeded=counts.get("succeeded", 0),
            dead=counts.get("dead", 0),
            cancelled=counts.get("cancelled", 0),
            ready_now=ready,
            scheduled_later=pending - ready,
            oldest_pending_age_seconds=round(now - oldest, 3) if oldest else None,
        )

    def wait_until_empty(self, *, timeout: float = 10.0, poll: float = 0.02) -> bool:
        """Block until nothing is pending or running. Returns False on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = self.stats()
            if current.pending == 0 and current.running == 0:
                return True
            time.sleep(poll)
        return False
