"""Storage layer and compatibility paths.

The fallback claim for SQLite older than 3.35 cannot run on a modern build, so
it is exercised by flipping the capability flag. Otherwise that branch would
only ever be tested by a user on an old distro, which is the worst place to
discover it is broken.
"""

from __future__ import annotations

import sqlite3

import pytest

from taskhive import ConfigurationError, TaskQueue, TaskState
from taskhive.db import ConnectionPool, connect, migrate, supports_returning

# ------------------------------------------------------- compatibility


def test_claim_fallback_matches_the_returning_path(tmp_path):
    """Force the pre-3.35 path and prove it behaves identically."""
    queue = TaskQueue(tmp_path / "legacy.db")
    queue._use_returning = False

    queue.enqueue("low", priority=0)
    queue.enqueue("high", priority=9)
    queue.enqueue("mid", priority=5)

    leased = queue.lease("worker-1", limit=3)
    assert [t.name for t in leased] == ["high", "mid", "low"]
    assert all(t.state is TaskState.RUNNING for t in leased)
    assert all(t.attempts == 1 for t in leased)
    assert len({t.lease_token for t in leased}) == 1, "one claim, one token"

    queue.ack(leased[0].id, leased[0].lease_token)
    assert queue.stats().succeeded == 1
    queue.close()


def test_fallback_on_an_empty_queue_returns_nothing(tmp_path):
    queue = TaskQueue(tmp_path / "legacy.db")
    queue._use_returning = False
    assert queue.lease("worker-1") == []
    queue.close()


def test_capability_probe_matches_the_runtime():
    expected = tuple(int(p) for p in sqlite3.sqlite_version.split(".")) >= (3, 35, 0)
    assert supports_returning() is expected


# ------------------------------------------------------------ pragmas


def test_opening_an_existing_wal_database_takes_no_lock(tmp_path):
    """The WAL switch happens once, not on every open.

    Flipping journal_mode takes a brief exclusive lock. Doing it on every
    connection means every concurrent open contends for that lock, which is how
    "database is locked" appeared on a line that only configures a pragma.
    """
    path = tmp_path / "wal.db"
    first = connect(path)
    assert first.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

    # A second connection should find WAL already set and leave it alone.
    second = connect(path)
    assert second.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    first.close()
    second.close()


def test_connection_enables_the_pragmas_that_matter(tmp_path):
    conn = connect(tmp_path / "tuned.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] > 0
    conn.close()


def test_parent_directories_are_created(tmp_path):
    nested = tmp_path / "a" / "b" / "c" / "tasks.db"
    queue = TaskQueue(nested)
    queue.enqueue("job")
    assert nested.exists()
    queue.close()


# --------------------------------------------------------- migrations


def test_migrate_is_idempotent(tmp_path):
    conn = connect(tmp_path / "m.db")
    assert migrate(conn) == migrate(conn) == 1
    count = conn.execute("SELECT COUNT(*) AS n FROM schema_version").fetchone()["n"]
    assert count == 1, "running migrations twice should not duplicate the version row"
    conn.close()


def test_a_newer_database_is_refused(tmp_path):
    conn = connect(tmp_path / "future.db")
    migrate(conn)
    conn.execute("UPDATE schema_version SET version = 99")
    with pytest.raises(ConfigurationError, match="Upgrade taskhive"):
        migrate(conn)
    conn.close()


def test_schema_rejects_an_invalid_state(tmp_path):
    conn = connect(tmp_path / "check.db")
    migrate(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO tasks (name, payload, state, created_at, available_at, updated_at)
               VALUES ('x', '{}', 'nonsense', 0, 0, 0)"""
        )
    conn.close()


def test_schema_rejects_zero_max_attempts(tmp_path):
    conn = connect(tmp_path / "check.db")
    migrate(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO tasks (name, payload, state, max_attempts,
                                  created_at, available_at, updated_at)
               VALUES ('x', '{}', 'pending', 0, 0, 0, 0)"""
        )
    conn.close()


# ---------------------------------------------------------------- pool


def test_memory_databases_share_one_connection():
    pool = ConnectionPool(":memory:")
    assert pool.serialized is True
    assert pool.get() is pool.get()
    pool.close()


def test_file_databases_are_per_thread(tmp_path):
    import threading

    pool = ConnectionPool(tmp_path / "threads.db")
    assert pool.serialized is False
    main_conn = pool.get()
    other: list[sqlite3.Connection] = []

    thread = threading.Thread(target=lambda: other.append(pool.get()))
    thread.start()
    thread.join()

    assert other[0] is not main_conn, "sharing a connection across threads is not safe"
    pool.close()


def test_in_memory_queue_round_trip():
    queue = TaskQueue(":memory:")
    task = queue.enqueue("job", {"n": 1})
    leased = queue.lease("w")[0]
    queue.ack(leased.id, leased.lease_token, result="done")
    assert queue.get(task.id).result == "done"
    queue.close()


# ------------------------------------------------------------ validation


def test_queue_rejects_nonsense_configuration():
    with pytest.raises(ValueError):
        TaskQueue(":memory:", default_max_attempts=0)


def test_enqueue_requires_a_name():
    queue = TaskQueue(":memory:")
    with pytest.raises(ValueError):
        queue.enqueue("")
    queue.close()


def test_lease_limit_must_be_positive():
    queue = TaskQueue(":memory:")
    with pytest.raises(ValueError):
        queue.lease("w", limit=0)
    queue.close()


def test_bulk_enqueue_requires_names():
    queue = TaskQueue(":memory:")
    assert queue.enqueue_many([]) == 0
    with pytest.raises(ValueError):
        queue.enqueue_many([{"payload": {}}])
    queue.close()


def test_wait_until_empty_times_out():
    queue = TaskQueue(":memory:")
    queue.enqueue("never_run")
    assert queue.wait_until_empty(timeout=0.1) is False
    queue.close()


def test_context_manager_closes():
    with TaskQueue(":memory:") as queue:
        queue.enqueue("job")
        assert queue.stats().pending == 1


def test_concurrent_opens_do_not_collide_on_schema_creation(tmp_path):
    """Sixteen threads opening the same new database at once must all succeed.

    Opening a queue runs DDL, which takes an exclusive lock. Several threads or
    processes constructing a TaskQueue at the same moment contend there before
    any task exists, and `busy_timeout` alone does not cover it. CI found this
    on loaded shared runners while developer machines passed every time: the
    failing test was doing its concurrent work through queues it had opened in
    parallel, so the error looked like a queue bug rather than a setup one.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from taskhive import TaskQueue

    path = tmp_path / "race.db"
    errors: list[Exception] = []
    guard = threading.Lock()
    ready = threading.Barrier(16)

    def open_and_use(index: int) -> None:
        queue = None
        try:
            ready.wait(timeout=30)
            queue = TaskQueue(path)
            queue.enqueue("job", {"i": index})
        except Exception as exc:
            with guard:
                errors.append(exc)
        finally:
            if queue is not None:
                queue.close()

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(open_and_use, range(16)))

    assert errors == [], f"concurrent open produced: {errors[:3]}"

    queue = TaskQueue(path)
    assert queue.stats().pending == 16
    queue.close()
