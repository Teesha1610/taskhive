"""Command line interface.

argparse rather than click or typer: this is a library, and a library that
drags a CLI framework into every install is a library people vendor around.
The standard library is enough for eight subcommands.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
from .models import TaskState
from .queue import DEFAULT_QUEUE, TaskQueue
from .worker import TaskRegistry, Worker


def load_registry(spec: str) -> TaskRegistry:
    """Import `module:attribute` and return the registry it names."""
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise SystemExit(f"expected module:attribute, got {spec!r}")
    sys.path.insert(0, "")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"cannot import {module_name!r}: {exc}") from exc
    registry = getattr(module, attribute, None)
    if not isinstance(registry, TaskRegistry):
        raise SystemExit(f"{spec} is not a TaskRegistry")
    return registry


def _table(rows: list[dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "  (none)"
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    header = "  " + "  ".join(c.ljust(widths[c]) for c in columns)
    divider = "  " + "  ".join("-" * widths[c] for c in columns)
    body = [
        "  " + "  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns) for row in rows
    ]
    return "\n".join([header, divider, *body])


def cmd_enqueue(args: argparse.Namespace) -> int:
    queue = TaskQueue(args.db)
    try:
        payload = json.loads(args.payload) if args.payload else {}
    except json.JSONDecodeError as exc:
        print(f"payload is not valid JSON: {exc}", file=sys.stderr)
        return 1
    task = queue.enqueue(
        args.name,
        payload,
        queue=args.queue,
        priority=args.priority,
        max_attempts=args.max_attempts,
        delay=args.delay,
        unique_key=args.unique_key,
    )
    print(json.dumps(task.as_dict(), indent=2) if args.json else f"enqueued task {task.id} ({task.name})")
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )
    registry = load_registry(args.registry)
    queue = TaskQueue(args.db)
    worker = Worker(
        queue,
        registry,
        queues=args.queues,
        concurrency=args.concurrency,
        lease_seconds=args.lease_seconds,
    )
    worker.install_signal_handlers()
    print(
        f"worker {worker.worker_id} on {', '.join(args.queues)} "
        f"with {args.concurrency} thread(s); handlers: {', '.join(registry.names) or 'none'}"
    )
    stats = worker.run(until_empty=args.until_empty)
    print(json.dumps(stats.as_dict(), indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    stats = TaskQueue(args.db).stats(queue=args.queue)
    if args.json:
        print(json.dumps(stats.as_dict(), indent=2))
        return 0
    data = stats.as_dict()
    width = max(len(k) for k in data)
    for key, value in data.items():
        print(f"  {key.ljust(width)}  {value if value is not None else '-'}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    tasks = TaskQueue(args.db).list_tasks(
        state=TaskState(args.state) if args.state else None,
        queue=args.queue,
        name=args.name,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps([t.as_dict() for t in tasks], indent=2))
        return 0
    rows = [
        {
            "id": t.id,
            "name": t.name,
            "queue": t.queue,
            "state": t.state.value,
            "attempts": f"{t.attempts}/{t.max_attempts}",
            "error": (t.last_error or "")[:48],
        }
        for t in tasks
    ]
    print(_table(rows, ["id", "name", "queue", "state", "attempts", "error"]))
    return 0


def cmd_requeue(args: argparse.Namespace) -> int:
    queue = TaskQueue(args.db)
    if args.all:
        moved = sum(1 for task in queue.dead_letters(limit=10_000) if queue.requeue(task.id))
        print(f"requeued {moved} dead task(s)")
        return 0
    if queue.requeue(args.task_id):
        print(f"requeued task {args.task_id}")
        return 0
    print(f"task {args.task_id} is not dead or cancelled", file=sys.stderr)
    return 1


def cmd_cancel(args: argparse.Namespace) -> int:
    if TaskQueue(args.db).cancel(args.task_id):
        print(f"cancelled task {args.task_id}")
        return 0
    print(f"task {args.task_id} is not pending, so it cannot be cancelled", file=sys.stderr)
    return 1


def cmd_reclaim(args: argparse.Namespace) -> int:
    print(f"reclaimed {TaskQueue(args.db).reclaim_expired()} expired lease(s)")
    return 0


def cmd_purge(args: argparse.Namespace) -> int:
    states = [TaskState(s) for s in args.states]
    removed = TaskQueue(args.db).purge(states=states, older_than_seconds=args.older_than)
    print(f"deleted {removed} task(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="taskhive", description="Durable task queue on SQLite")
    parser.add_argument("--version", action="version", version=f"taskhive {__version__}")
    parser.add_argument("--db", default="taskhive.db", help="database file (default: taskhive.db)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("enqueue", help="add a task")
    p.add_argument("name")
    p.add_argument("payload", nargs="?", help="JSON object")
    p.add_argument("--queue", default=DEFAULT_QUEUE)
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--delay", type=float, default=0.0, help="seconds before it becomes available")
    p.add_argument("--unique-key", default=None, help="collapse duplicates while one is live")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_enqueue)

    p = sub.add_parser("worker", help="run a worker")
    p.add_argument("registry", help="module:attribute holding a TaskRegistry")
    p.add_argument("--queues", nargs="+", default=[DEFAULT_QUEUE])
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--lease-seconds", type=float, default=60.0)
    p.add_argument("--until-empty", action="store_true", help="exit once the queue drains")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_worker)

    p = sub.add_parser("stats", help="counts by state")
    p.add_argument("--queue", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("ls", help="list tasks")
    p.add_argument("--state", choices=[s.value for s in TaskState], default=None)
    p.add_argument("--queue", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("requeue", help="send dead tasks back to the queue")
    p.add_argument("task_id", type=int, nargs="?", default=0)
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=cmd_requeue)

    p = sub.add_parser("cancel", help="cancel a pending task")
    p.add_argument("task_id", type=int)
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser("reclaim", help="return expired leases to the queue")
    p.set_defaults(func=cmd_reclaim)

    p = sub.add_parser("purge", help="delete terminal tasks")
    p.add_argument("--states", nargs="+", default=["succeeded", "cancelled"])
    p.add_argument("--older-than", type=float, default=0.0, help="seconds")
    p.set_defaults(func=cmd_purge)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
