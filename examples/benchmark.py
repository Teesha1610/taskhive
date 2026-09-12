"""Throughput benchmark.

Measures the three operations that matter separately, because they have very
different costs: inserts are fsync bound, claims are lock bound, and the round
trip is both plus handler time.

    python examples/benchmark.py
    python examples/benchmark.py --tasks 20000 --workers 8
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from taskhive import TaskQueue, TaskRegistry, Worker


def timed(label: str, count: int, function) -> float:
    start = time.perf_counter()
    function()
    elapsed = time.perf_counter() - start
    rate = count / elapsed if elapsed else float("inf")
    print(f"  {label:<34} {elapsed:7.3f}s  {rate:10,.0f} tasks/sec")
    return rate


def main() -> int:
    parser = argparse.ArgumentParser(description="taskhive throughput")
    parser.add_argument("--tasks", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=50)
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="taskhive-bench-"))
    db = workdir / "bench.db"
    print(f"\ntaskhive benchmark: {args.tasks:,} tasks, {args.workers} workers, batch {args.batch}")
    print(f"database: {db}\n")

    try:
        # 1. Bulk insert, one transaction.
        queue = TaskQueue(db)
        timed(
            "enqueue_many (one transaction)",
            args.tasks,
            lambda: queue.enqueue_many([{"name": "noop", "payload": {"i": i}} for i in range(args.tasks)]),
        )

        # 2. Claim and ack, single threaded.
        def drain_single() -> None:
            while True:
                batch = queue.lease("bench", limit=args.batch)
                if not batch:
                    break
                for task in batch:
                    queue.ack(task.id, task.lease_token)

        timed("lease + ack (1 thread)", args.tasks, drain_single)
        queue.purge()

        # 3. Claim and ack, contended across threads.
        queue.enqueue_many([{"name": "noop", "payload": {"i": i}} for i in range(args.tasks)])
        counted = {"n": 0}
        guard = threading.Lock()

        def drain_parallel(index: int) -> None:
            local = TaskQueue(db)
            try:
                while True:
                    batch = local.lease(f"bench-{index}", limit=args.batch)
                    if not batch:
                        break
                    for task in batch:
                        local.ack(task.id, task.lease_token)
                    with guard:
                        counted["n"] += len(batch)
            finally:
                local.close()

        def run_parallel() -> None:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(drain_parallel, range(args.workers)))

        timed(f"lease + ack ({args.workers} threads)", args.tasks, run_parallel)
        assert counted["n"] == args.tasks, f"lost tasks: {args.tasks - counted['n']}"
        queue.purge()

        # 4. Full round trip through the worker runtime.
        registry = TaskRegistry()
        registry.register("noop", lambda i: None)
        queue.enqueue_many([{"name": "noop", "payload": {"i": i}} for i in range(args.tasks)])
        worker = Worker(queue, registry, concurrency=args.workers, poll_interval=0.001)
        timed(
            f"worker round trip ({args.workers} threads)",
            args.tasks,
            lambda: worker.run(until_empty=True),
        )

        size_mb = db.stat().st_size / 1_048_576
        print(f"\n  database grew to {size_mb:.1f} MB for {args.tasks:,} completed tasks")
        print(f"  worker stats: {worker.stats.as_dict()}\n")
        queue.close()
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
