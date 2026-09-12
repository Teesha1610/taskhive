"""Queue semantics."""

from __future__ import annotations

from datetime import timedelta

import pytest

from taskhive import (
    FixedBackoff,
    LeaseExpiredError,
    NoRetry,
    TaskQueue,
    TaskState,
)
from taskhive.models import utcnow


@pytest.fixture
def queue(tmp_path) -> TaskQueue:
    q = TaskQueue(tmp_path / "tasks.db", retry_policy=FixedBackoff(delay=0.0))
    yield q
    q.close()


# ------------------------------------------------------------- lifecycle


def test_enqueue_then_lease_then_ack(queue: TaskQueue):
    task = queue.enqueue("send_email", {"to": "a@example.com"})
    assert task.state is TaskState.PENDING
    assert task.attempts == 0

    leased = queue.lease("worker-1")
    assert len(leased) == 1
    assert leased[0].id == task.id
    assert leased[0].state is TaskState.RUNNING
    assert leased[0].attempts == 1, "the attempt is counted at claim time, not at failure"
    assert leased[0].lease_token

    queue.ack(leased[0].id, leased[0].lease_token, result={"delivered": True})
    done = queue.get(task.id)
    assert done.state is TaskState.SUCCEEDED
    assert done.result == {"delivered": True}
    assert done.lease_token is None


def test_leasing_an_empty_queue_returns_nothing(queue: TaskQueue):
    assert queue.lease("worker-1") == []


def test_a_leased_task_is_invisible_to_other_workers(queue: TaskQueue):
    queue.enqueue("job")
    first = queue.lease("worker-1")
    second = queue.lease("worker-2")
    assert len(first) == 1
    assert second == []


def test_lease_respects_limit_and_order(queue: TaskQueue):
    queue.enqueue("low", priority=0)
    high = queue.enqueue("high", priority=10)
    queue.enqueue("mid", priority=5)

    leased = queue.lease("worker-1", limit=2)
    assert [t.name for t in leased] == ["high", "mid"]
    assert leased[0].id == high.id


def test_queues_are_isolated(queue: TaskQueue):
    queue.enqueue("a", queue="emails")
    queue.enqueue("b", queue="reports")

    assert [t.name for t in queue.lease("w", queues=["emails"])] == ["a"]
    assert queue.lease("w", queues=["nothing-here"]) == []
    assert len(queue.lease("w", queues=["emails", "reports"], limit=5)) == 1


# --------------------------------------------------------------- retries


def test_failure_returns_the_task_to_pending(queue: TaskQueue):
    queue.enqueue("flaky", max_attempts=3)
    leased = queue.lease("worker-1")[0]

    updated = queue.nack(leased.id, leased.lease_token, error="boom")
    assert updated.state is TaskState.PENDING
    assert updated.attempts == 1
    assert updated.attempts_remaining == 2
    assert updated.last_error == "boom"


def test_exhausting_attempts_dead_letters(queue: TaskQueue):
    queue.enqueue("doomed", max_attempts=2)

    for _ in range(2):
        leased = queue.lease("worker-1")[0]
        task = queue.nack(leased.id, leased.lease_token, error="nope")

    assert task.state is TaskState.DEAD
    assert task.attempts == 2
    assert queue.lease("worker-1") == [], "dead tasks are not redelivered"
    assert [t.id for t in queue.dead_letters()] == [task.id]


def test_retry_false_skips_remaining_attempts(queue: TaskQueue):
    queue.enqueue("malformed", max_attempts=5)
    leased = queue.lease("worker-1")[0]

    task = queue.nack(leased.id, leased.lease_token, error="bad payload", retry=False)
    assert task.state is TaskState.DEAD
    assert task.attempts == 1, "a permanent failure does not burn the remaining budget"


def test_retry_delay_comes_from_the_policy(tmp_path):
    q = TaskQueue(tmp_path / "t.db", retry_policy=FixedBackoff(delay=120.0))
    q.enqueue("job")
    leased = q.lease("w")[0]

    task = q.nack(leased.id, leased.lease_token, error="x")
    wait = (task.available_at - utcnow()).total_seconds()
    assert 110 < wait <= 120
    assert q.lease("w") == [], "it is pending but not yet available"
    q.close()


def test_explicit_delay_overrides_the_policy(tmp_path):
    q = TaskQueue(tmp_path / "t.db", retry_policy=FixedBackoff(delay=999.0))
    q.enqueue("job")
    leased = q.lease("w")[0]

    task = q.nack(leased.id, leased.lease_token, error="rate limited", delay=0.0)
    assert (task.available_at - utcnow()).total_seconds() < 1
    assert len(q.lease("w")) == 1
    q.close()


def test_no_retry_policy_dead_letters_on_first_failure(tmp_path):
    q = TaskQueue(tmp_path / "t.db", retry_policy=NoRetry(), default_max_attempts=1)
    q.enqueue("job")
    leased = q.lease("w")[0]
    assert q.nack(leased.id, leased.lease_token, error="x").state is TaskState.DEAD
    q.close()


# ------------------------------------------------------------ scheduling


def test_delayed_tasks_are_not_available_yet(queue: TaskQueue):
    queue.enqueue("later", delay=60)
    assert queue.lease("worker-1") == []

    future = utcnow() + timedelta(seconds=61)
    assert len(queue.lease("worker-1", now=future)) == 1


def test_run_at_accepts_a_datetime(queue: TaskQueue):
    when = utcnow() + timedelta(hours=2)
    task = queue.enqueue("scheduled", run_at=when)
    assert abs((task.available_at - when).total_seconds()) < 1
    assert queue.lease("w") == []


def test_negative_delay_is_rejected(queue: TaskQueue):
    with pytest.raises(ValueError):
        queue.enqueue("job", delay=-5)


# ----------------------------------------------------------- idempotency


def test_unique_key_collapses_duplicates(queue: TaskQueue):
    first = queue.enqueue("nightly", unique_key="nightly-2026-09-11")
    second = queue.enqueue("nightly", unique_key="nightly-2026-09-11")

    assert first.id == second.id
    assert queue.stats().total == 1


def test_unique_key_is_released_once_terminal(queue: TaskQueue):
    first = queue.enqueue("nightly", unique_key="k")
    leased = queue.lease("w")[0]
    queue.ack(leased.id, leased.lease_token)

    second = queue.enqueue("nightly", unique_key="k")
    assert second.id != first.id, "a finished task should not block tomorrow's run"


def test_tasks_without_a_unique_key_are_never_collapsed(queue: TaskQueue):
    a = queue.enqueue("job", {"x": 1})
    b = queue.enqueue("job", {"x": 1})
    assert a.id != b.id


# --------------------------------------------------------------- leases


def test_ack_with_a_stale_token_is_refused(queue: TaskQueue):
    queue.enqueue("job")
    leased = queue.lease("worker-1")[0]

    with pytest.raises(LeaseExpiredError):
        queue.ack(leased.id, "not-the-real-token")

    assert queue.get(leased.id).state is TaskState.RUNNING


def test_expired_lease_returns_the_task(queue: TaskQueue):
    queue.enqueue("job", max_attempts=3)
    leased = queue.lease("worker-1", lease_seconds=0.0)[0]

    assert queue.reclaim_expired() == 1
    recovered = queue.get(leased.id)
    assert recovered.state is TaskState.PENDING
    assert recovered.attempts == 1, "the crashed attempt still counts"
    assert "lease expired" in recovered.last_error


def test_expired_lease_dead_letters_when_attempts_are_gone(queue: TaskQueue):
    queue.enqueue("job", max_attempts=1)
    leased = queue.lease("worker-1", lease_seconds=0.0)[0]

    assert queue.reclaim_expired() == 1
    assert queue.get(leased.id).state is TaskState.DEAD


def test_reclaim_leaves_live_leases_alone(queue: TaskQueue):
    queue.enqueue("job")
    queue.lease("worker-1", lease_seconds=300)
    assert queue.reclaim_expired() == 0


def test_heartbeat_extends_a_lease(queue: TaskQueue):
    queue.enqueue("slow")
    leased = queue.lease("worker-1", lease_seconds=0.5)[0]

    assert queue.heartbeat(leased.id, leased.lease_token, extend_seconds=300) is True
    assert queue.reclaim_expired() == 0
    assert queue.get(leased.id).state is TaskState.RUNNING


def test_heartbeat_with_a_bad_token_fails_quietly(queue: TaskQueue):
    queue.enqueue("job")
    leased = queue.lease("worker-1")[0]
    assert queue.heartbeat(leased.id, "wrong") is False


# ------------------------------------------------------------ management


def test_cancel_only_applies_to_pending_tasks(queue: TaskQueue):
    task = queue.enqueue("job")
    assert queue.cancel(task.id) is True
    assert queue.get(task.id).state is TaskState.CANCELLED

    other = queue.enqueue("job2")
    queue.lease("w")
    assert queue.cancel(other.id) is False, "a running task cannot be cancelled mid-flight"


def test_requeue_revives_a_dead_task(queue: TaskQueue):
    queue.enqueue("job", max_attempts=1)
    leased = queue.lease("w")[0]
    queue.nack(leased.id, leased.lease_token, error="x")

    assert queue.requeue(leased.id) is True
    revived = queue.get(leased.id)
    assert revived.state is TaskState.PENDING
    assert revived.attempts == 0
    assert len(queue.lease("w")) == 1


def test_purge_refuses_to_delete_live_tasks(queue: TaskQueue):
    with pytest.raises(ValueError):
        queue.purge(states=[TaskState.PENDING])


def test_purge_removes_finished_tasks(queue: TaskQueue):
    queue.enqueue("a")
    leased = queue.lease("w")[0]
    queue.ack(leased.id, leased.lease_token)
    queue.enqueue("b")

    assert queue.purge(states=[TaskState.SUCCEEDED]) == 1
    assert queue.stats().total == 1


def test_bulk_enqueue(queue: TaskQueue):
    count = queue.enqueue_many([{"name": "job", "payload": {"i": i}} for i in range(500)])
    assert count == 500
    assert queue.stats().pending == 500


# ------------------------------------------------------------------ stats


def test_stats_separate_ready_from_scheduled(queue: TaskQueue):
    queue.enqueue("now")
    queue.enqueue("later", delay=3600)
    queue.enqueue("running-one")
    queue.lease("w", limit=1)

    stats = queue.stats()
    assert stats.pending == 2
    assert stats.ready_now == 1
    assert stats.scheduled_later == 1
    assert stats.running == 1
    assert stats.oldest_pending_age_seconds is not None


def test_stats_can_filter_by_queue(queue: TaskQueue):
    queue.enqueue("a", queue="emails")
    queue.enqueue("b", queue="reports")
    assert queue.stats(queue="emails").pending == 1


def test_persistence_survives_reopening(tmp_path):
    path = tmp_path / "durable.db"
    first = TaskQueue(path)
    task = first.enqueue("survivor", {"n": 1})
    first.close()

    second = TaskQueue(path)
    found = second.get(task.id)
    assert found is not None
    assert found.payload == {"n": 1}
    assert found.state is TaskState.PENDING
    second.close()
