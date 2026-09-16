"""Worker runtime."""

from __future__ import annotations

import threading
import time

import pytest

from taskhive import (
    FixedBackoff,
    PermanentFailure,
    RetryLater,
    TaskQueue,
    TaskRegistry,
    TaskState,
    Worker,
)


@pytest.fixture
def queue(tmp_path) -> TaskQueue:
    q = TaskQueue(tmp_path / "worker.db", retry_policy=FixedBackoff(delay=0.0))
    yield q
    q.close()


# ------------------------------------------------------------- registry


def test_registry_resolves_by_name():
    registry = TaskRegistry()

    @registry.task("greet")
    def greet(name: str) -> str:
        return f"hello {name}"

    assert registry.resolve("greet")({"name": "world"}) == "hello world"
    assert registry.names == ["greet"]
    assert "greet" in registry


def test_registry_defaults_to_the_function_name():
    registry = TaskRegistry()

    @registry.task()
    def resize_image() -> None: ...

    assert registry.names == ["resize_image"]


def test_duplicate_registration_is_rejected():
    registry = TaskRegistry()
    registry.register("job", lambda: None)
    with pytest.raises(ValueError):

        @registry.task("job")
        def other() -> None: ...


# --------------------------------------------------------------- running


def test_worker_processes_a_task(queue: TaskQueue):
    registry = TaskRegistry()
    seen: list[dict] = []

    @registry.task("record")
    def record(**payload: object) -> dict:
        seen.append(payload)
        return {"ok": True}

    queue.enqueue("record", {"value": 42})
    stats = Worker(queue, registry, concurrency=1).run(until_empty=True)

    assert seen == [{"value": 42}]
    assert stats.succeeded == 1
    assert stats.processed == 1
    assert queue.stats().succeeded == 1


def test_handler_result_is_stored(queue: TaskQueue):
    registry = TaskRegistry()
    registry.register("compute", lambda x: {"doubled": x * 2})

    task = queue.enqueue("compute", {"x": 21})
    Worker(queue, registry, concurrency=1).run(until_empty=True)

    assert queue.get(task.id).result == {"doubled": 42}


def test_a_raising_handler_retries_then_dies(queue: TaskQueue):
    registry = TaskRegistry()
    calls = []

    @registry.task("explode")
    def explode() -> None:
        calls.append(1)
        raise RuntimeError("kaboom")

    task = queue.enqueue("explode", max_attempts=3)
    stats = Worker(queue, registry, concurrency=1).run(until_empty=True)

    assert len(calls) == 3, "it should use the whole retry budget"
    assert stats.retried == 2
    assert stats.dead == 1
    final = queue.get(task.id)
    assert final.state is TaskState.DEAD
    assert "kaboom" in final.last_error


def test_permanent_failure_skips_remaining_attempts(queue: TaskQueue):
    registry = TaskRegistry()
    calls = []

    @registry.task("malformed")
    def malformed() -> None:
        calls.append(1)
        raise PermanentFailure("payload will never be valid")

    task = queue.enqueue("malformed", max_attempts=5)
    Worker(queue, registry, concurrency=1).run(until_empty=True)

    assert len(calls) == 1, "retrying a permanent failure wastes the budget"
    assert queue.get(task.id).state is TaskState.DEAD


def test_retry_later_sets_an_explicit_delay(queue: TaskQueue):
    registry = TaskRegistry()

    @registry.task("rate_limited")
    def rate_limited() -> None:
        raise RetryLater(300, "slow down")

    task = queue.enqueue("rate_limited", max_attempts=3)
    Worker(queue, registry, concurrency=1).run(until_empty=True)

    updated = queue.get(task.id)
    assert updated.state is TaskState.PENDING
    from taskhive.models import utcnow

    assert (updated.available_at - utcnow()).total_seconds() > 250


def test_unknown_task_dies_without_retrying(queue: TaskQueue):
    task = queue.enqueue("nobody_handles_this", max_attempts=5)
    Worker(queue, TaskRegistry(), concurrency=1).run(until_empty=True)

    final = queue.get(task.id)
    assert final.state is TaskState.DEAD
    assert final.attempts == 1
    assert "no handler registered" in final.last_error


def test_worker_honors_max_tasks(queue: TaskQueue):
    registry = TaskRegistry()
    registry.register("job", lambda: None)
    for _ in range(10):
        queue.enqueue("job")

    stats = Worker(queue, registry, concurrency=1).run(max_tasks=3)
    assert stats.processed >= 3
    assert queue.stats().pending > 0


def test_worker_runs_tasks_concurrently(queue: TaskQueue):
    registry = TaskRegistry()
    active = {"now": 0, "peak": 0}
    guard = threading.Lock()

    @registry.task("slow")
    def slow() -> None:
        with guard:
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
        time.sleep(0.08)
        with guard:
            active["now"] -= 1

    for _ in range(8):
        queue.enqueue("slow")

    Worker(queue, registry, concurrency=4).run(until_empty=True)
    assert active["peak"] > 1, "concurrency=4 should overlap handlers"
    assert queue.stats().succeeded == 8


def test_worker_only_pulls_from_its_queues(queue: TaskQueue):
    registry = TaskRegistry()
    registry.register("job", lambda: None)

    queue.enqueue("job", queue="emails")
    queue.enqueue("job", queue="reports")

    Worker(queue, registry, queues=["emails"], concurrency=1).run(until_empty=True)
    stats = queue.stats()
    assert stats.succeeded == 1
    assert stats.pending == 1


# ------------------------------------------------------------- shutdown


def test_stop_drains_in_flight_work(queue: TaskQueue):
    registry = TaskRegistry()
    started = threading.Event()
    finished: list[int] = []

    @registry.task("slow")
    def slow() -> None:
        started.set()
        time.sleep(0.3)
        finished.append(1)

    queue.enqueue("slow")
    worker = Worker(queue, registry, concurrency=1, lease_seconds=30)
    thread = worker.run_in_thread()

    assert started.wait(timeout=5), "handler never started"
    worker.stop(drain=True)
    thread.join(timeout=10)

    assert finished == [1], "a draining shutdown must let running work finish"
    assert queue.stats().succeeded == 1


def test_lost_lease_discards_the_result(queue: TaskQueue):
    """If the lease is reclaimed mid-handler, the stale result is not written."""
    registry = TaskRegistry()
    running = threading.Event()
    release = threading.Event()

    @registry.task("slow")
    def slow() -> None:
        running.set()
        release.wait(timeout=5)

    task = queue.enqueue("slow", max_attempts=3)
    worker = Worker(queue, registry, concurrency=1, lease_seconds=0.05, heartbeat_interval=99)
    thread = worker.run_in_thread(max_tasks=1)

    assert running.wait(timeout=5)
    time.sleep(0.1)
    queue.reclaim_expired()  # simulate another worker reclaiming it
    release.set()
    thread.join(timeout=10)

    assert worker.stats.lost_leases == 1
    assert queue.get(task.id).state is not TaskState.SUCCEEDED


def test_worker_stats_shape(queue: TaskQueue):
    registry = TaskRegistry()
    registry.register("job", lambda: None)
    queue.enqueue("job")

    stats = Worker(queue, registry, concurrency=1).run(until_empty=True)
    data = stats.as_dict()
    assert set(data) == {
        "processed",
        "succeeded",
        "retried",
        "dead",
        "lost_leases",
        "uptime_seconds",
        "tasks_per_second",
    }


def test_heartbeat_uses_one_thread_for_all_tasks(queue: TaskQueue):
    """The sweeper must extend every in-flight lease, not just one.

    Asserts that `leased_until` actually moved rather than that nothing expired
    within a window. The window version passed or failed depending on how loaded
    the machine was, which is no kind of test at all: it failed on a CI runner
    and never on a laptop.
    """
    registry = TaskRegistry()
    release = threading.Event()
    running = threading.Semaphore(0)

    @registry.task("slow")
    def slow() -> None:
        running.release()
        release.wait(timeout=30)

    for _ in range(3):
        queue.enqueue("slow", max_attempts=2)

    before_threads = threading.active_count()
    # A long lease, so expiry cannot happen regardless of scheduling, and a
    # short heartbeat so several sweeps run inside the wait.
    worker = Worker(queue, registry, concurrency=3, lease_seconds=30, heartbeat_interval=0.05)
    thread = worker.run_in_thread(max_tasks=3)

    for _ in range(3):
        assert running.acquire(timeout=15), "handlers did not all start"

    leases_before = {t.id: t.leased_until for t in queue.list_tasks(state=TaskState.RUNNING)}
    assert len(leases_before) == 3

    time.sleep(0.5)  # several heartbeat intervals, however loaded the machine
    leases_after = {t.id: t.leased_until for t in queue.list_tasks(state=TaskState.RUNNING)}

    extended = [tid for tid, after in leases_after.items() if after > leases_before[tid]]
    assert len(extended) == 3, f"the sweeper extended {len(extended)} of 3 leases"

    extra_threads = threading.active_count() - before_threads
    assert extra_threads <= 6, "one sweeper plus the pool, not a thread per task"

    release.set()
    thread.join(timeout=30)
    assert queue.stats().succeeded == 3


def test_idle_backoff_resets_once_work_resumes(queue: TaskQueue):
    """A worker must not stay in slow-poll mode while a queue is draining.

    Regression guard: an edit once removed the line that reset `backoff` after a
    successful claim, so the interval kept growing toward max_poll_interval and
    every batch waited on it. Every existing test still passed, because they use
    a handful of tasks. This one uses enough batches for the difference to show:
    with the reset it finishes in well under a second, without it the same work
    takes roughly a hundred times longer.
    """
    registry = TaskRegistry()
    registry.register("noop", lambda i: None)
    queue.enqueue_many([{"name": "noop", "payload": {"i": i}} for i in range(200)])

    worker = Worker(
        queue, registry, concurrency=2, poll_interval=0.001, max_poll_interval=0.5
    )
    started = time.monotonic()
    worker.run(until_empty=True)
    elapsed = time.monotonic() - started

    assert worker.stats.succeeded == 200
    assert elapsed < 20, f"draining 200 tasks took {elapsed:.1f}s; the poll backoff is not resetting"
