"""Domain types.

Everything the caller touches is a frozen dataclass or an enum, so a task that
came out of the database cannot be mutated into an inconsistent state by
accident. State transitions go through TaskQueue, never through attribute
assignment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TaskState(str, Enum):
    """Lifecycle of a task.

    pending -> running -> succeeded
                       -> pending   (retry, attempts remain)
                       -> dead      (attempts exhausted or permanent failure)
    pending -> cancelled
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    DEAD = "dead"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (TaskState.SUCCEEDED, TaskState.DEAD, TaskState.CANCELLED)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_epoch(moment: datetime) -> float:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def from_epoch(value: float | None) -> datetime | None:
    return datetime.fromtimestamp(value, tz=timezone.utc) if value is not None else None


@dataclass(frozen=True, slots=True)
class Task:
    """One unit of work as stored."""

    id: int
    name: str
    queue: str
    payload: dict[str, Any]
    state: TaskState
    priority: int
    attempts: int
    max_attempts: int
    created_at: datetime
    available_at: datetime
    updated_at: datetime
    leased_until: datetime | None = None
    lease_token: str | None = None
    worker_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    last_error: str | None = None
    result: Any = None
    unique_key: str | None = None

    @property
    def attempts_remaining(self) -> int:
        return max(0, self.max_attempts - self.attempts)

    @classmethod
    def from_row(cls, row: Any) -> Task:
        return cls(
            id=row["id"],
            name=row["name"],
            queue=row["queue"],
            payload=json.loads(row["payload"]) if row["payload"] else {},
            state=TaskState(row["state"]),
            priority=row["priority"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            created_at=from_epoch(row["created_at"]),  # type: ignore[arg-type]
            available_at=from_epoch(row["available_at"]),  # type: ignore[arg-type]
            updated_at=from_epoch(row["updated_at"]),  # type: ignore[arg-type]
            leased_until=from_epoch(row["leased_until"]),
            lease_token=row["lease_token"],
            worker_id=row["worker_id"],
            started_at=from_epoch(row["started_at"]),
            finished_at=from_epoch(row["finished_at"]),
            last_error=row["last_error"],
            result=json.loads(row["result"]) if row["result"] else None,
            unique_key=row["unique_key"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "queue": self.queue,
            "payload": self.payload,
            "state": self.state.value,
            "priority": self.priority,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at.isoformat(),
            "available_at": self.available_at.isoformat(),
            "leased_until": self.leased_until.isoformat() if self.leased_until else None,
            "worker_id": self.worker_id,
            "last_error": self.last_error,
            "result": self.result,
            "unique_key": self.unique_key,
        }


@dataclass(frozen=True, slots=True)
class QueueStats:
    queue: str
    pending: int = 0
    running: int = 0
    succeeded: int = 0
    dead: int = 0
    cancelled: int = 0
    oldest_pending_age_seconds: float | None = None
    ready_now: int = 0
    scheduled_later: int = 0

    @property
    def total(self) -> int:
        return self.pending + self.running + self.succeeded + self.dead + self.cancelled

    def as_dict(self) -> dict[str, Any]:
        return {
            "queue": self.queue,
            "pending": self.pending,
            "ready_now": self.ready_now,
            "scheduled_later": self.scheduled_later,
            "running": self.running,
            "succeeded": self.succeeded,
            "dead": self.dead,
            "cancelled": self.cancelled,
            "total": self.total,
            "oldest_pending_age_seconds": self.oldest_pending_age_seconds,
        }
