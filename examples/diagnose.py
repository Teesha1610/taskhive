"""Locate where worker time actually goes, on the machine that has the problem.

    python examples/diagnose.py

Prints a breakdown instead of one throughput number, so the slow component is
identified rather than guessed at.
"""

from __future__ import annotations

import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from taskhive import TaskQueue, TaskRegistry, Worker

N = 600


def fresh(tmp: Path, name: str) -> TaskQueue:
    queue = TaskQueue(tmp / name)
    queue.enqueue_many([{"name": "noop", "payload": {"i": i}} for i in range(N)])
    return queue


def summarize(label: str, samples: list[float]) -> None:
    if not samples:
        print(f"  {label:<38} (none)")
        return
    ms = [s * 1000 for s in samples]
    print(
        f"  {label:<38} n={len(ms):<5} "
        f"mean={statistics.mean(ms):7.2f}ms  median={statistics.median(ms):7.2f}ms  "
        f"max={max(ms):8.2f}ms  total={sum(ms)/1000:6.2f}s"
    )


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="taskhive-diag-"))
    print("\ntimer granularity check (what a 1ms sleep really costs)")
    sleeps = []
    for _ in range(20):
        t0 = time.perf_counter()
        time.sleep(0.001)
        sleeps.append(time.perf_counter() - t0)
    summarize("time.sleep(0.001)", sleeps)

    event = threading.Event()
    waits = []
    for _ in range(20):
        t0 = time.perf_counter()
        event.wait(0.001)
        waits.append(time.perf_counter() - t0)
    summarize("Event.wait(0.001)", waits)

    print(f"\nper-call cost inside a single thread ({N} tasks)")
    queue = fresh(tmp, "single.db")
    lease_times, ack_times = [], []
    while True:
        t0 = time.perf_counter()
        batch = queue.lease("diag", limit=4)
        lease_times.append(time.perf_counter() - t0)
        if not batch:
            break
        for task in batch:
            t0 = time.perf_counter()
            queue.ack(task.id, task.lease_token)
            ack_times.append(time.perf_counter() - t0)
    summarize("queue.lease(limit=4)", lease_times)
    summarize("queue.ack()", ack_times)
    queue.close()

    registry = TaskRegistry()
    registry.register("noop", lambda i: None)

    for concurrency in (1, 2, 4):
        queue = fresh(tmp, f"w{concurrency}.db")
        worker = Worker(queue, registry, concurrency=concurrency, poll_interval=0.001)
        t0 = time.perf_counter()
        worker.run(until_empty=True)
        elapsed = time.perf_counter() - t0
        print(
            f"\n  worker concurrency={concurrency:<2} {elapsed:6.2f}s  "
            f"{N/elapsed:8,.0f} tasks/sec  {elapsed/N*1000:6.2f}ms per task"
        )
        queue.close()

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
