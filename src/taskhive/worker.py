"""Worker runtime.

Handlers run in a thread pool. Threads rather than processes because the work
this queue is built for is IO bound (HTTP calls, database writes, file
shuffling), and threads avoid paying pickling costs on every payload. A CPU
bound handler should hand off to a process pool inside the handler itself.

Shutdown is the part worth reading. On SIGINT or SIGTERM the worker stops
claiming new tasks and waits for in-flight ones to finish, so a deploy does not
turn running work into dead letters. A second signal stops waiting.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .errors import LeaseExpiredError, PermanentFailure, RetryLater, UnknownTaskError
from .models import Task
from .queue import DEFAULT_QUEUE, TaskQueue

log = logging.getLogger("taskhive")

Handler = Callable[..., Any]


class TaskRegistry:
    """Name to handler mapping.

    >>> registry = TaskRegistry()
    >>> @registry.task("greet")
    ... def greet(name: str) -> str:
    ...     return f"hello {name}"
    >>> registry.resolve("greet")({"name": "world"})
    'hello world'
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def task(self, name: str | None = None) -> Callable[[Handler], Handler]:
        def decorate(func: Handler) -> Handler:
            key = name or func.__name__
            if key in self._handlers:
                raise ValueError(f"handler already registered for {key!r}")
            self._handlers[key] = func
            return func

        return decorate

    def register(self, name: str, handler: Handler) -> None:
        self._handlers[name] = handler

    def resolve(self, name: str) -> Callable[[dict[str, Any]], Any]:
        handler = self._handlers.get(name)
        if handler is None:
            raise UnknownTaskError(f"no handler registered for {name!r}")
        return lambda payload: handler(**payload)

    @property
    def names(self) -> list[str]:
        return sorted(self._handlers)

    def __contains__(self, name: object) -> bool:
        return name in self._handlers


@dataclass
class WorkerStats:
    processed: int = 0
    succeeded: int = 0
    retried: int = 0
    dead: int = 0
    lost_leases: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def throughput(self) -> float:
        elapsed = self.uptime_seconds
        return round(self.processed / elapsed, 2) if elapsed > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "succeeded": self.succeeded,
            "retried": self.retried,
            "dead": self.dead,
            "lost_leases": self.lost_leases,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "tasks_per_second": self.throughput,
        }


class Worker:
    """Claims tasks, runs handlers, reports outcomes."""

    def __init__(
        self,
        queue: TaskQueue,
        registry: TaskRegistry,
        *,
        queues: Iterable[str] = (DEFAULT_QUEUE,),
        concurrency: int = 4,
        poll_interval: float = 0.1,
        max_poll_interval: float = 2.0,
        lease_seconds: float = 60.0,
        heartbeat_interval: float = 15.0,
        worker_id: str | None = None,
        reclaim_interval: float = 30.0,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.queue = queue
        self.registry = registry
        self.queues = list(queues)
        self.concurrency = concurrency
        self.poll_interval = poll_interval
        self.max_poll_interval = max_poll_interval
        self.lease_seconds = lease_seconds
        self.heartbeat_interval = heartbeat_interval
        self.reclaim_interval = reclaim_interval
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"

        self.stats = WorkerStats()
        # Counts free handler slots. Acquired on submit, released by a done
        # callback, so the loop can block on it without a timer.
        self._slots = threading.Semaphore(concurrency)
        self._stop = threading.Event()
        self._draining = threading.Event()
        self._in_flight: dict[int, str] = {}
        self._in_flight_lock = threading.Lock()
        self._last_reclaim = 0.0

    # ------------------------------------------------------------ control

    def stop(self, *, drain: bool = True) -> None:
        """Ask the worker to finish. With drain, in-flight tasks complete."""
        if drain:
            self._draining.set()
        self._stop.set()

    def install_signal_handlers(self) -> None:
        def handle(signum: int, _frame: Any) -> None:
            if self._stop.is_set():
                log.warning("second signal received, abandoning in-flight tasks")
                os._exit(1)
            log.info("signal %s received, draining %d in-flight task(s)", signum, len(self._in_flight))
            self.stop(drain=True)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handle)
            except ValueError:
                # Not on the main thread. Callers can still use stop().
                log.debug("cannot install handler for %s off the main thread", sig)

    # -------------------------------------------------------------- loop

    def run(self, *, max_tasks: int | None = None, until_empty: bool = False) -> WorkerStats:
        """Process until stopped, or until a bound is reached.

        `max_tasks` and `until_empty` exist so tests and one-shot drains do not
        need to race a background thread.
        """
        backoff = self.poll_interval
        submitted = 0

        # One sweeper extends every in-flight lease. See _heartbeat_loop for
        # why this is not a thread per task.
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(heartbeat_stop,),
            daemon=True,
            name="taskhive-heartbeat",
        )
        heartbeat.start()

        try:
            with ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="taskhive") as pool:
                pending: set[Future[None]] = set()

                while not self._stop.is_set():
                    self._maybe_reclaim()

                    # Reap finished futures before measuring capacity, so a batch
                    # that just completed frees its slots in this pass rather than
                    # the next one.
                    pending = {f for f in pending if not f.done()}

                    capacity = self.concurrency - len(pending)
                    if max_tasks is not None:
                        # Bound claims, not completions. A worker told to take one
                        # task must not claim a second because the first lost its
                        # lease, or it would run more work than it was asked to.
                        capacity = min(capacity, max_tasks - submitted)

                    claimed: list[Task] = []
                    if capacity > 0:
                        claimed = self.queue.lease(
                            self.worker_id,
                            queues=self.queues,
                            limit=capacity,
                            lease_seconds=self.lease_seconds,
                        )

                    for task in claimed:
                        with self._in_flight_lock:
                            self._in_flight[task.id] = task.lease_token or ""
                        self._slots.acquire()
                        future = pool.submit(self._run_one, task)
                        future.add_done_callback(lambda _f: self._slots.release())
                        pending.add(future)
                        submitted += 1

                    if max_tasks is not None and submitted >= max_tasks and not pending:
                        break

                    if claimed:
                        # Work is flowing: drop back to the fast poll interval
                        # so the next batch is claimed immediately.
                        backoff = self.poll_interval
                        continue

                    # Nothing claimed. Only safe to exit once nothing is in flight
                    # either: a running handler can requeue work by failing, and
                    # exiting before it finishes would leave that work behind.
                    if until_empty and not pending:
                        break
                    if max_tasks is not None and submitted >= max_tasks and not pending:
                        break

                    if pending:
                        # Every slot is busy. Block on the slot semaphore, which
                        # a completion callback releases, so this wakes the
                        # instant a handler finishes.
                        #
                        # Any *timed* wait here is a trap on Windows, where
                        # threading.Event.wait has about 15.6ms of granularity
                        # (time.sleep does not: it uses a high resolution timer).
                        # Both a 1ms poll and concurrent.futures.wait cost 15.6ms
                        # per call there, and since either runs once per task,
                        # end to end throughput collapsed from thousands per
                        # second to 64. Linux hid it entirely at about 0.1ms a
                        # call. An untimed acquire has no such floor: the
                        # release wakes it directly.
                        self._slots.acquire()
                        self._slots.release()
                        backoff = self.poll_interval
                        continue

                    # The queue is genuinely empty. Back off so a quiet worker
                    # does not spin the CPU.
                    self._stop.wait(backoff)
                    backoff = min(backoff * 1.5, self.max_poll_interval)

                if self._draining.is_set() or not self._stop.is_set():
                    for future in list(pending):
                        future.result()

        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=1.0)

        return self.stats

    def run_in_thread(self, **kwargs: Any) -> threading.Thread:
        thread = threading.Thread(target=self.run, kwargs=kwargs, daemon=True, name="taskhive-worker")
        thread.start()
        return thread

    # ------------------------------------------------------------ one task

    def _run_one(self, task: Task) -> None:
        token = task.lease_token or ""
        try:
            handler = self.registry.resolve(task.name)
            result = handler(task.payload)
            self._finish(task, token, result=result)
        except UnknownTaskError as exc:
            # A handler that does not exist will not start existing on retry.
            self._fail(task, token, str(exc), retry=False)
        except PermanentFailure as exc:
            self._fail(task, token, f"{type(exc).__name__}: {exc}", retry=False)
        except RetryLater as exc:
            self._fail(task, token, str(exc), retry=True, delay=exc.delay_seconds)
        except Exception as exc:
            log.exception("task %s (%s) raised", task.id, task.name)
            self._fail(task, token, f"{type(exc).__name__}: {exc}", retry=True)
        finally:
            with self._in_flight_lock:
                self._in_flight.pop(task.id, None)
            self.stats.processed += 1

    def _finish(self, task: Task, token: str, *, result: Any) -> None:
        try:
            self.queue.ack(task.id, token, result=result)
            self.stats.succeeded += 1
        except LeaseExpiredError:
            # The lease was reclaimed while the handler ran. Another worker may
            # already be redoing this task, so the result is dropped rather
            # than written over newer state.
            self.stats.lost_leases += 1
            log.warning("lease lost for task %s before ack; result discarded", task.id)

    def _fail(self, task: Task, token: str, error: str, *, retry: bool, delay: float | None = None) -> None:
        try:
            updated = self.queue.nack(task.id, token, error=error, retry=retry, delay=delay)
            if updated.state.value == "dead":
                self.stats.dead += 1
            else:
                self.stats.retried += 1
        except LeaseExpiredError:
            self.stats.lost_leases += 1
            log.warning("lease lost for task %s before nack", task.id)

    def _heartbeat_loop(self, stop: threading.Event) -> None:
        """Extend every in-flight lease from a single thread.

        An earlier version started one thread per task. That is fine on Linux,
        where creating a thread costs microseconds, and ruinous on Windows,
        where five thousand of them dominated a benchmark run: end to end
        throughput fell from roughly 2,400 tasks a second to 67. One sweeper
        does the same work at fixed cost, and holds the lock only long enough
        to copy the in-flight map.
        """
        while not stop.wait(self.heartbeat_interval):
            with self._in_flight_lock:
                live = list(self._in_flight.items())
            for task_id, token in live:
                try:
                    self.queue.heartbeat(task_id, token, extend_seconds=self.lease_seconds)
                except Exception:
                    log.debug("heartbeat failed for task %s", task_id, exc_info=True)

    def _maybe_reclaim(self) -> None:
        now = time.monotonic()
        if now - self._last_reclaim < self.reclaim_interval:
            return
        self._last_reclaim = now
        try:
            recovered = self.queue.reclaim_expired()
            if recovered:
                log.info("reclaimed %d expired lease(s)", recovered)
        except Exception:
            log.debug("reclaim pass failed", exc_info=True)
