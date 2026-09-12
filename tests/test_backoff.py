"""Retry policy edges: validation, determinism, and the no-jitter contract."""

from __future__ import annotations

import random

import pytest

from taskhive import ExponentialBackoff, FixedBackoff, NoRetry
from taskhive.backoff import RetryPolicy


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base": 0},
        {"base": -1},
        {"factor": 0.5},
        {"max_delay": 0.1, "base": 1.0},
        {"jitter": "wild"},
    ],
)
def test_exponential_rejects_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        ExponentialBackoff(**kwargs)


@pytest.mark.parametrize("kwargs", [{"delay": -1}, {"jitter": 1.5}, {"jitter": -0.1}])
def test_fixed_rejects_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        FixedBackoff(**kwargs)


def test_attempt_is_one_based():
    with pytest.raises(ValueError):
        ExponentialBackoff().delay_for(0)


def test_no_jitter_is_exactly_the_curve():
    policy = ExponentialBackoff(base=2, factor=3, max_delay=1000, jitter="none")
    assert [policy.delay_for(n) for n in (1, 2, 3)] == [2, 6, 18]


def test_equal_jitter_keeps_half_the_delay_fixed():
    policy = ExponentialBackoff(base=100, max_delay=100, jitter="equal", rng=random.Random(3))
    draws = [policy.delay_for(1) for _ in range(30)]
    assert all(50 <= d <= 100 for d in draws), "equal jitter never returns less than half"


def test_a_seeded_policy_is_reproducible():
    first = ExponentialBackoff(rng=random.Random(99))
    second = ExponentialBackoff(rng=random.Random(99))
    assert [first.delay_for(n) for n in range(1, 6)] == [second.delay_for(n) for n in range(1, 6)]


def test_fixed_without_jitter_is_constant():
    policy = FixedBackoff(delay=42)
    assert {policy.delay_for(n) for n in range(1, 10)} == {42.0}


def test_no_retry_returns_zero():
    assert NoRetry().delay_for(5) == 0.0


@pytest.mark.parametrize("policy", [NoRetry(), FixedBackoff(), ExponentialBackoff()])
def test_policies_satisfy_the_protocol(policy):
    assert isinstance(policy, RetryPolicy)
    assert repr(policy)


def test_repr_shows_the_configuration():
    assert "max_delay=60.0" in repr(ExponentialBackoff(max_delay=60))
    assert "delay=5.0" in repr(FixedBackoff(delay=5))
