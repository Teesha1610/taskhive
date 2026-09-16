"""Concurrency.

The claim is the only operation where correctness is not obvious, so it gets
tested with real threads hammering a real database file rather than with a
mock. These are the tests that would catch a regression from someone
"simplifying" BEGIN IMMEDIATE away.
"""

from __future__ import annotations

import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest

from taskhive import FixedBackoff, TaskQueue, TaskState


@pytest.fixture
def path(tmp_path):
    return tmp_path / "concurrent.db"


def test_no_task_is_leased_twice(path):
    """Eight threads race for 200 tasks. Every task goes to exactly one of them."""
    producer = TaskQueue(path)
    producer.enqueue_many([{"name": "job", "payload": {"i": i}} for i in range(200)])
    producer.close()

    claimed: list[int] = []
    guard = threading.Lock()
    start = threading.Barrier(8)

    def drain(worker_index: int) -> None:
        queue = TaskQueue(path)
        start.wait(timeout=10)
        try:
            while True:
                batch = queue.lease(f"worker-{worker_index}", limit=5)
                if not batch:
                    break
                with guard:
                    claimed.extend(task.id for task in batch)
                for task in batch:
                    queue.ack(task.id, task.lease_token)
        finally:
            queue.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(drain, range(8)))

    duplicates = [task_id for task_id, n in Counter(claimed).items() if n > 1]
    assert duplicates == [], f"{len(duplicates)} task(s) were delivered more than once"
    assert len(claimed) == 200, "every task should have been claimed exactly once"

    checker = TaskQueue(path)
    assert checker.stats().succeeded == 200
    checker.close()


def test_concurrent_enqueue_and_lease_do_not_deadlock(path):
    """Writers and claimers interleave without SQLITE_BUSY escaping."""
    stop = threading.Event()
    errors: list[Exception] = []

    # Opening and closing the connection sit inside the try on purpose. An
    # earlier version constructed the queue outside it, so anything those steps
    # raised escaped the thread rather than landing in `errors`, and the
    # assertion below passed without proving anything. pytest surfaced that as
    # an unhandled thread exception, intermittently, on Windows only.
    def produce() -> None:
        queue = None
        try:
            queue = TaskQueue(path)
            for i in range(150):
                queue.enqueue("job", {"i": i})
        except Exception as exc:
            errors.append(exc)
        finally:
            stop.set()
            if queue is not None:
                queue.close()

    def consume() -> None:
        queue = None
        try:
            queue = TaskQueue(path)
            while not stop.is_set():
                for task in queue.lease("consumer", limit=3):
                    queue.ack(task.id, task.lease_token)
        except Exception as exc:
            errors.append(exc)
        finally:
            if queue is not None:
                queue.close()

    threads = [threading.Thread(target=produce), *(threading.Thread(target=consume) for _ in range(3))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive(), "a thread hung, which suggests a lock cycle"

    assert errors == [], f"unexpected errors: {errors}"

    queue = TaskQueue(path)
    remaining = queue.lease("cleanup", limit=500)
    for task in remaining:
        queue.ack(task.id, task.lease_token)
    assert queue.stats().succeeded == 150
    queue.close()


def test_unique_key_holds_under_concurrent_enqueue(path):
    """Twelve threads enqueue the same key. Exactly one task exists afterwards."""
    ready = threading.Barrier(12)
    ids: list[int] = []
    guard = threading.Lock()

    def submit(_index: int) -> None:
        queue = TaskQueue(path)
        ready.wait(timeout=10)
        task = queue.enqueue("nightly", unique_key="only-once")
        with guard:
            ids.append(task.id)
        queue.close()

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(submit, range(12)))

    assert len(set(ids)) == 1, "the unique index should collapse every duplicate"

    queue = TaskQueue(path)
    assert queue.stats().total == 1
    queue.close()


def test_only_one_worker_can_ack_a_reclaimed_task(path):
    """After a lease is reclaimed, the original holder's ack is refused."""
    queue = TaskQueue(path, retry_policy=FixedBackoff(delay=0.0))
    queue.enqueue("job", max_attempts=3)

    stalled = queue.lease("worker-slow", lease_seconds=0.0)[0]
    queue.reclaim_expired()
    second = queue.lease("worker-fast")[0]

    assert second.id == stalled.id
    assert second.lease_token != stalled.lease_token

    from taskhive import LeaseExpiredError

    with pytest.raises(LeaseExpiredError):
        queue.ack(stalled.id, stalled.lease_token)

    queue.ack(second.id, second.lease_token)
    assert queue.get(second.id).state is TaskState.SUCCEEDED
    queue.close()


def test_write_pressure_never_surfaces_database_is_locked(path):
    """Eight threads writing as fast as they can must not raise.

    SQLite's own busy handler does not cover every conflict: in WAL mode a
    writer whose snapshot has moved on gets SQLITE_BUSY_SNAPSHOT with no wait at
    all. Under enough pressure that reaches the caller as "database is locked",
    which is the queue failing at exactly the moment it is busiest.

    This is the test that was missing. CI on a loaded shared runner found the
    bug first, on one matrix cell out of ten, while a developer laptop passed
    every time.
    """
    errors: list[Exception] = []
    guard = threading.Lock()
    start = threading.Barrier(8)

    def hammer(index: int) -> None:
        queue = None
        try:
            queue = TaskQueue(path)
            start.wait(timeout=15)
            for i in range(40):
                queue.enqueue("job", {"worker": index, "i": i})
                for task in queue.lease(f"w{index}", limit=2):
                    if i % 3:
                        queue.ack(task.id, task.lease_token)
                    else:
                        queue.nack(task.id, task.lease_token, error="retry me")
        except Exception as exc:
            with guard:
                errors.append(exc)
        finally:
            if queue is not None:
                queue.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(hammer, range(8)))

    assert errors == [], f"write pressure produced {len(errors)} error(s): {errors[:3]}"

    queue = TaskQueue(path)
    stats = queue.stats()
    assert stats.total == 320, "every enqueue should have landed exactly once"
    queue.close()
