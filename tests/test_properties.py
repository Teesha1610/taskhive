"""Property-based tests.

Example tests check the cases you thought of. These check the cases you did
not: Hypothesis generates inputs, shrinks any failure to a minimal reproduction,
and remembers it. The invariants below are the ones that have to hold for the
queue to be safe, stated once instead of enumerated.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from taskhive import ExponentialBackoff, FixedBackoff, TaskQueue, TaskState

# Database work is slow relative to Hypothesis's default deadline.
db_settings = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


# ------------------------------------------------------- backoff policies


@given(
    base=st.floats(min_value=0.001, max_value=100, allow_nan=False),
    factor=st.floats(min_value=1.0, max_value=10, allow_nan=False),
    cap=st.floats(min_value=0.001, max_value=10_000, allow_nan=False),
    attempt=st.integers(min_value=1, max_value=64),
)
def test_backoff_never_exceeds_the_cap(base, factor, cap, attempt):
    assume(cap >= base)
    policy = ExponentialBackoff(base=base, factor=factor, max_delay=cap, rng=random.Random(0))
    delay = policy.delay_for(attempt)
    assert 0 <= delay <= cap


@given(attempt=st.integers(min_value=1, max_value=1000))
def test_backoff_survives_absurd_attempt_counts(attempt):
    """A large attempt number must not raise OverflowError."""
    policy = ExponentialBackoff(base=1.0, factor=2.0, max_delay=3600, jitter="none")
    assert policy.delay_for(attempt) == pytest.approx(min(2.0 ** (attempt - 1), 3600))


@given(attempt=st.integers(min_value=1, max_value=20))
def test_unjittered_backoff_is_monotonic(attempt):
    policy = ExponentialBackoff(jitter="none")
    assert policy.delay_for(attempt + 1) >= policy.delay_for(attempt)


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_full_jitter_spreads_retries(seed):
    """The point of jitter: two tasks failing together do not retry together."""
    policy = ExponentialBackoff(base=10, max_delay=100, rng=random.Random(seed))
    draws = [policy.delay_for(5) for _ in range(50)]
    assert len(set(draws)) > 1, "jitter that returns a constant is not jitter"
    assert all(0 <= d <= 100 for d in draws)


@given(
    delay=st.floats(min_value=0, max_value=1000, allow_nan=False),
    jitter=st.floats(min_value=0, max_value=1, allow_nan=False),
)
def test_fixed_backoff_stays_non_negative(delay, jitter):
    policy = FixedBackoff(delay=delay, jitter=jitter, rng=random.Random(1))
    assert policy.delay_for(1) >= 0


@given(bad=st.floats(max_value=0, allow_nan=False, allow_infinity=False))
def test_invalid_base_is_rejected(bad):
    with pytest.raises(ValueError):
        ExponentialBackoff(base=bad)


# --------------------------------------------------------- queue invariants


@db_settings
@given(
    names=st.lists(st.text(min_size=1, max_size=24), min_size=1, max_size=12),
    priorities=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=12),
)
def test_lease_order_always_follows_priority(tmp_path_factory, names, priorities):
    """Whatever goes in, the highest priority ready task comes out first."""
    path = tmp_path_factory.mktemp("prio") / "q.db"
    queue = TaskQueue(path)
    try:
        pairs = list(zip(names, priorities, strict=False))
        for name, priority in pairs:
            queue.enqueue(name, priority=priority)

        leased = queue.lease("w", limit=len(pairs))
        got = [t.priority for t in leased]
        assert got == sorted(got, reverse=True)
        assert got[0] == max(p for _, p in pairs)
    finally:
        queue.close()


@db_settings
@given(max_attempts=st.integers(min_value=1, max_value=6))
def test_a_task_never_runs_more_than_max_attempts(tmp_path_factory, max_attempts):
    """The core safety property: the retry budget is never exceeded."""
    path = tmp_path_factory.mktemp("attempts") / "q.db"
    queue = TaskQueue(path, retry_policy=FixedBackoff(delay=0.0))
    try:
        task = queue.enqueue("always_fails", max_attempts=max_attempts)
        runs = 0
        while True:
            batch = queue.lease("w")
            if not batch:
                break
            runs += 1
            assert runs <= max_attempts, "claimed more times than the budget allows"
            queue.nack(batch[0].id, batch[0].lease_token, error="fail")

        assert runs == max_attempts
        final = queue.get(task.id)
        assert final.state is TaskState.DEAD
        assert final.attempts == max_attempts
    finally:
        queue.close()


@db_settings
@given(
    count=st.integers(min_value=1, max_value=25),
    batch=st.integers(min_value=1, max_value=8),
)
def test_every_enqueued_task_reaches_a_terminal_state(tmp_path_factory, count, batch):
    """No task is lost, stuck, or duplicated while draining."""
    path = tmp_path_factory.mktemp("drain") / "q.db"
    queue = TaskQueue(path, retry_policy=FixedBackoff(delay=0.0))
    try:
        ids = {queue.enqueue("job", {"i": i}).id for i in range(count)}
        seen: list[int] = []

        while True:
            leased = queue.lease("w", limit=batch)
            if not leased:
                break
            for task in leased:
                seen.append(task.id)
                queue.ack(task.id, task.lease_token)

        assert sorted(seen) == sorted(ids)
        assert len(seen) == len(set(seen)), "a task was handed out twice"
        stats = queue.stats()
        assert stats.succeeded == count
        assert stats.pending == 0 and stats.running == 0
    finally:
        queue.close()


@db_settings
@given(keys=st.lists(st.text(min_size=1, max_size=12), min_size=1, max_size=15))
def test_unique_keys_never_produce_two_live_tasks(tmp_path_factory, keys):
    path = tmp_path_factory.mktemp("unique") / "q.db"
    queue = TaskQueue(path)
    try:
        for key in keys:
            queue.enqueue("job", unique_key=key)
        assert queue.stats().pending == len(set(keys))
    finally:
        queue.close()
