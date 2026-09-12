"""A worked example: an outbound email pipeline.

Run it end to end:

    python examples/email_pipeline.py

Or split producer and consumer across two terminals:

    python examples/email_pipeline.py produce
    taskhive --db emails.db worker examples.email_pipeline:registry --until-empty
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from taskhive import (
    ExponentialBackoff,
    PermanentFailure,
    RetryLater,
    TaskQueue,
    TaskRegistry,
    Worker,
)

DB = Path(__file__).with_name("emails.db")
registry = TaskRegistry()
rng = random.Random(20260911)


@registry.task("send_email")
def send_email(to: str, subject: str, body: str = "") -> dict:
    """Pretend to call an email provider.

    The three outcomes below are the three every network handler has, and each
    one maps to different queue behavior.
    """
    if "@" not in to:
        # Retrying will not fix a malformed address. Skip the budget.
        raise PermanentFailure(f"not an address: {to!r}")

    roll = rng.random()
    if roll < 0.15:
        # The provider told us exactly how long to wait. Trust it over backoff.
        raise RetryLater(2.0, "provider rate limit")
    if roll < 0.35:
        # A transient failure. The policy decides when to try again.
        raise ConnectionError("provider returned 503")

    return {"delivered_to": to, "subject": subject}


@registry.task("send_digest")
def send_digest(user_id: int, period: str) -> dict:
    return {"user_id": user_id, "period": period, "items": rng.randint(1, 20)}


def produce(queue: TaskQueue) -> None:
    recipients = [
        "ana@example.com",
        "ben@example.com",
        "cai@example.com",
        "not-an-address",  # dead letters on the first attempt
        "dee@example.com",
    ]
    for index, address in enumerate(recipients):
        queue.enqueue(
            "send_email",
            {"to": address, "subject": f"Welcome, message {index}"},
            priority=10 if index == 0 else 0,
            max_attempts=4,
        )

    # Scheduled work: available in five seconds, not now.
    queue.enqueue("send_digest", {"user_id": 1, "period": "daily"}, delay=5)

    # Idempotent work: enqueue it as often as you like, it exists once.
    for _ in range(5):
        queue.enqueue(
            "send_digest",
            {"user_id": 2, "period": "weekly"},
            unique_key="digest-user-2-2026-W37",
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    queue = TaskQueue(DB, retry_policy=ExponentialBackoff(base=0.2, max_delay=3))

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("produce", "all"):
        produce(queue)
        print(f"queued: {queue.stats().as_dict()}")

    if mode in ("consume", "all"):
        worker = Worker(queue, registry, concurrency=3, lease_seconds=10)
        worker.install_signal_handlers()
        stats = worker.run(until_empty=True)
        print(f"\nworker: {stats.as_dict()}")
        print(f"queue:  {queue.stats().as_dict()}")

        dead = queue.dead_letters()
        if dead:
            print("\ndead letters:")
            for task in dead:
                print(f"  {task.id}  {task.name}  {task.last_error}")
            print("\nrequeue them with: taskhive --db examples/emails.db requeue --all")

    queue.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
