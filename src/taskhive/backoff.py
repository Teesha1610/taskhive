"""Retry policies.

Full jitter is the default rather than plain exponential backoff. Without
jitter, a batch of tasks that fail together retries together, and the
thundering herd that caused the first failure recurs on a schedule. Randomizing
across the whole interval spreads the load and is what AWS recommends after
measuring the alternatives.
"""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable


@runtime_checkable
class RetryPolicy(Protocol):
    """Maps an attempt number to a delay in seconds before the next try."""

    def delay_for(self, attempt: int) -> float:
        """Seconds to wait before attempt+1. `attempt` is 1-based."""
        ...


class NoRetry:
    """Fail permanently on the first error."""

    def delay_for(self, attempt: int) -> float:
        return 0.0

    def __repr__(self) -> str:
        return "NoRetry()"


class FixedBackoff:
    """Constant delay. Predictable, and fine when the work is idempotent."""

    __slots__ = ("_rng", "delay", "jitter")

    def __init__(self, delay: float = 30.0, jitter: float = 0.0, rng: random.Random | None = None) -> None:
        if delay < 0:
            raise ValueError("delay must be non-negative")
        if not 0 <= jitter <= 1:
            raise ValueError("jitter must be between 0 and 1")
        self.delay = float(delay)
        self.jitter = float(jitter)
        # Jitter spreads retry load. It is not a security primitive, so the
        # Mersenne Twister is the right tool and secrets would be the wrong one.
        self._rng = rng or random.Random()  # noqa: S311

    def delay_for(self, attempt: int) -> float:
        if not self.jitter:
            return self.delay
        spread = self.delay * self.jitter
        return max(0.0, self.delay + self._rng.uniform(-spread, spread))

    def __repr__(self) -> str:
        return f"FixedBackoff(delay={self.delay}, jitter={self.jitter})"


class ExponentialBackoff:
    """base * factor ** (attempt - 1), capped, then jittered.

    With the defaults the uncapped sequence is 1s, 2s, 4s, 8s, 16s. Full jitter
    turns each of those into a uniform draw from [0, value], so the cap bounds
    the worst case and the mean is half the nominal delay.
    """

    __slots__ = ("_rng", "base", "factor", "jitter", "max_delay")

    def __init__(
        self,
        base: float = 1.0,
        factor: float = 2.0,
        max_delay: float = 3600.0,
        jitter: str = "full",
        rng: random.Random | None = None,
    ) -> None:
        if base <= 0:
            raise ValueError("base must be positive")
        if factor < 1:
            raise ValueError("factor must be at least 1")
        if max_delay < base:
            raise ValueError("max_delay must be at least base")
        if jitter not in ("full", "equal", "none"):
            raise ValueError("jitter must be one of: full, equal, none")
        self.base = float(base)
        self.factor = float(factor)
        self.max_delay = float(max_delay)
        self.jitter = jitter
        # See FixedBackoff: jitter is load shaping, not cryptography.
        self._rng = rng or random.Random()  # noqa: S311

    def delay_for(self, attempt: int) -> float:
        if attempt < 1:
            raise ValueError("attempt is 1-based")
        # Compute in a way that cannot overflow on a large attempt count.
        try:
            nominal = self.base * (self.factor ** (attempt - 1))
        except OverflowError:
            nominal = self.max_delay
        capped = min(nominal, self.max_delay)

        if self.jitter == "none":
            return capped
        if self.jitter == "equal":
            half = capped / 2
            return half + self._rng.uniform(0, half)
        return self._rng.uniform(0, capped)

    def __repr__(self) -> str:
        return (
            f"ExponentialBackoff(base={self.base}, factor={self.factor}, "
            f"max_delay={self.max_delay}, jitter={self.jitter!r})"
        )


DEFAULT_POLICY: RetryPolicy = ExponentialBackoff()
