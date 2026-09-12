"""Errors.

The distinction that matters at runtime is PermanentFailure against everything
else. A handler raising PermanentFailure skips the remaining attempts and goes
straight to the dead letter queue, because retrying a malformed payload just
burns the retry budget on an outcome that cannot change.
"""

from __future__ import annotations


class TaskHiveError(Exception):
    """Base class for everything this package raises."""


class ConfigurationError(TaskHiveError):
    """The environment cannot support the requested behavior."""


class UnknownTaskError(TaskHiveError):
    """No handler is registered under that name."""


class LeaseExpiredError(TaskHiveError):
    """The lease was reclaimed before the worker reported back.

    Raised when acking or nacking with a token the database no longer accepts,
    which means another worker may already have picked the task up. The result
    is discarded rather than written over newer state.
    """


class PermanentFailure(TaskHiveError):
    """Raise inside a handler to skip retries and dead-letter immediately."""


class RetryLater(TaskHiveError):
    """Raise inside a handler to retry after an explicit delay.

    Useful for rate limits, where the service has told you exactly how long to
    wait and the backoff policy's guess would be worse.
    """

    def __init__(self, delay_seconds: float, message: str = "") -> None:
        super().__init__(message or f"retry in {delay_seconds}s")
        self.delay_seconds = float(delay_seconds)
