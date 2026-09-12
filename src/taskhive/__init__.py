"""taskhive: a durable task queue that needs nothing but a SQLite file.

    from taskhive import TaskQueue, TaskRegistry, Worker

    queue = TaskQueue("tasks.db")
    registry = TaskRegistry()

    @registry.task("send_email")
    def send_email(to: str, subject: str) -> dict:
        ...
        return {"delivered": True}

    queue.enqueue("send_email", {"to": "a@example.com", "subject": "hi"})
    Worker(queue, registry).run(until_empty=True)

Delivery is at-least-once: a worker that dies mid-task leaves a lease that
expires, and the task is retried. Handlers must be idempotent.
"""

from .backoff import ExponentialBackoff, FixedBackoff, NoRetry, RetryPolicy
from .errors import (
    ConfigurationError,
    LeaseExpiredError,
    PermanentFailure,
    RetryLater,
    TaskHiveError,
    UnknownTaskError,
)
from .models import QueueStats, Task, TaskState
from .queue import TaskQueue
from .worker import TaskRegistry, Worker, WorkerStats

__version__ = "1.0.0"

__all__ = [
    "ConfigurationError",
    "ExponentialBackoff",
    "FixedBackoff",
    "LeaseExpiredError",
    "NoRetry",
    "PermanentFailure",
    "QueueStats",
    "RetryLater",
    "RetryPolicy",
    "Task",
    "TaskHiveError",
    "TaskQueue",
    "TaskRegistry",
    "TaskState",
    "UnknownTaskError",
    "Worker",
    "WorkerStats",
    "__version__",
]
