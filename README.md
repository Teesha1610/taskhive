# taskhive

**A durable task queue on a single SQLite file.** No broker, no daemon, no
runtime dependencies. Enqueue in one process, run in another, survive a crash in
either.

![CI](https://github.com/Teesha1610/taskhive/actions/workflows/ci.yml/badge.svg)
`107 tests` · `92% coverage` · `strict mypy` · `Linux, macOS, Windows` · `Python 3.10+`

**What is interesting here**

- **Atomic claims under concurrency.** One statement inside `BEGIN IMMEDIATE`
  hands each task to exactly one worker. Proven with eight real threads racing
  for 200 tasks and asserting zero duplicates.
- **Crash recovery by lease expiry.** A worker that dies mid-task leaves a lease
  that runs out; the task returns to the queue. At-least-once delivery, stated
  explicitly rather than pretended away.
- **Retries with full jitter**, plus escape hatches for failures that will never
  succeed and for rate limits that tell you exactly how long to wait.
- **Four kinds of test**, including Hypothesis properties and a compatibility
  path for SQLite versions this machine cannot run.
- **A 9x performance bug found by measurement.** End to end throughput on
  Windows was 64 tasks a second against 5,350 on Linux. Three hypotheses were
  wrong before instrumentation found the cause: `threading.Event.wait(0.001)`
  takes 15.6ms on Windows while `time.sleep(0.001)` takes 1.6ms, and the claim
  loop was paying it once per task. [The whole story, dead ends
  included.](#five-bugs-the-tests-and-benchmarks-caught)

```python
from taskhive import TaskQueue, TaskRegistry, Worker

queue = TaskQueue("tasks.db")
registry = TaskRegistry()

@registry.task("send_email")
def send_email(to: str, subject: str) -> dict:
    provider.send(to, subject)
    return {"delivered": True}

queue.enqueue("send_email", {"to": "ana@example.com", "subject": "Welcome"})
Worker(queue, registry, concurrency=4).run()
```

```bash
pip install -e ".[dev]"    # install with test and lint extras
pytest                     # 107 tests, no services required
python examples/benchmark.py
```

**Where to look:** [`src/taskhive/queue.py`](src/taskhive/queue.py) for the
claim, [`src/taskhive/worker.py`](src/taskhive/worker.py) for the runtime,
[`tests/test_concurrency.py`](tests/test_concurrency.py) for the threading
proofs, [`tests/test_properties.py`](tests/test_properties.py) for the
Hypothesis invariants.

---

## Why this exists

Celery and RQ are excellent and both want a broker running somewhere. Plenty of
applications have background work but not the operational appetite for another
service: a CLI tool doing deferred uploads, a small web app sending email, a
data pipeline retrying flaky API calls. Those need durability and retries, not a
cluster.

SQLite is already on the machine, already crash safe, and in WAL mode handles
concurrent readers with a single writer perfectly well. The interesting question
is whether the claim can be made safe, and it can, in one statement.

## The design decisions worth knowing

### Claiming is one atomic statement

Several workers poll the same table. Exactly one must get each task. The claim
is a single `UPDATE ... WHERE id IN (SELECT ... ORDER BY ... LIMIT n)` inside a
`BEGIN IMMEDIATE` transaction:

```sql
UPDATE tasks
   SET state = 'running', attempts = attempts + 1,
       lease_token = ?, leased_until = ?, worker_id = ?
 WHERE id IN (
       SELECT id FROM tasks
        WHERE state = 'pending' AND available_at <= ? AND queue IN (...)
        ORDER BY priority DESC, available_at ASC, id ASC
        LIMIT ?)
RETURNING ...;
```

`IMMEDIATE` takes the write lock up front rather than upgrading from a read lock
midway, which is what produces `SQLITE_BUSY` deadlocks between two writers that
both began as readers. Correctness does not depend on workers cooperating, and
`tests/test_concurrency.py` proves it with eight real threads racing for 200
tasks and asserting zero duplicates.

### Delivery is at-least-once, by choice

A worker that dies mid-task leaves a lease that expires. `reclaim_expired()`
returns that task to the queue, so work is never silently lost. The tradeoff is
that a handler can run twice, so **handlers must be idempotent**. Every durable
queue makes this tradeoff; exactly-once delivery is not available over a network
and pretending otherwise just hides the problem.

The attempt is counted at claim time, not at failure. A process that crashes in
a loop still exhausts its retry budget instead of looping forever.

### Retries use full jitter

Plain exponential backoff makes a batch that failed together retry together,
recreating the load spike that caused the failure. Full jitter draws uniformly
from `[0, min(cap, base * factor ** (attempt - 1))]`, spreading retries across
the interval.

Three failure kinds get three behaviors:

| Handler raises | Queue does |
| --- | --- |
| any exception | retry with backoff until `max_attempts` |
| `PermanentFailure` | dead-letter immediately, budget untouched |
| `RetryLater(seconds)` | retry after exactly that delay, ignoring the policy |

A malformed payload will not become valid on the third attempt. Burning the
budget on it just delays the dead letter.

### Idempotent enqueue via a partial unique index

```sql
CREATE UNIQUE INDEX idx_tasks_unique_key ON tasks (unique_key)
  WHERE unique_key IS NOT NULL AND state IN ('pending','running');
```

Enqueue with `unique_key="digest-2026-09-11"` as many times as you like and one
task exists. Because the index excludes terminal rows, the key is released once
the task finishes, so tomorrow's run can reuse the same shape of key. The
database enforces this, not application logic, so it holds under concurrent
writers from separate processes.

### Shutdown drains rather than drops

On `SIGINT` or `SIGTERM` the worker stops claiming and waits for in-flight
handlers to finish, so a deploy does not convert running work into dead letters.
A second signal stops waiting.

## Performance

`examples/benchmark.py`, 5,000 tasks, 4 workers, on two machines:

| Operation | Linux (py3.12) | Windows 11 (py3.13) |
| --- | --- | --- |
| `enqueue_many`, one transaction | 121,000/sec | 57,000/sec |
| lease and ack, single thread | 11,500/sec | 3,540/sec |
| lease and ack, 4 threads | 10,900/sec | 1,240 to 2,700/sec |
| full worker round trip, 4 threads | 5,350/sec | 590/sec |

Windows is slower across the board, mostly because `FlushFileBuffers` costs more
than Linux's fsync and every commit pays it.

The contended row is the unstable one. Across runs on the same Windows machine
four threads measured anywhere from 1,240 to 2,700 tasks a second, sometimes
faster than a single thread and sometimes half its speed, while on Linux four
threads are consistently a little *slower* than one. SQLite serializes writers,
so the claim gains nothing from parallelism; what varies is how much of another
thread's flush a blocked writer overlaps with. Treat any single measurement of
that row as noise.

What follows: concurrency buys overlap on handler IO, not on queue operations,
and bulk insert beats individual enqueues by an order of magnitude on both
platforms because one commit means one fsync.

Run it on your own hardware before quoting any of these:

```bash
python examples/benchmark.py --tasks 20000 --workers 8
```

## API

```python
queue = TaskQueue("tasks.db", retry_policy=ExponentialBackoff(base=2, max_delay=600))

queue.enqueue("name", {"payload": 1}, priority=5, max_attempts=4, delay=30)
queue.enqueue("nightly", unique_key="2026-09-11")     # idempotent
queue.enqueue_many([{"name": "job", "payload": {}} for _ in range(1000)])

tasks = queue.lease("worker-1", limit=10, lease_seconds=60)
queue.heartbeat(task.id, task.lease_token)            # extend a long lease
queue.ack(task.id, task.lease_token, result={"ok": True})
queue.nack(task.id, task.lease_token, error="boom")   # retry or dead-letter

queue.reclaim_expired()        # recover from a crashed worker
queue.cancel(task_id)          # pending tasks only
queue.requeue(task_id)         # revive a dead letter
queue.purge(older_than_seconds=86400)
queue.stats()                  # counts, ready vs scheduled, oldest pending age
```

## CLI

```bash
taskhive enqueue send_email '{"to": "a@example.com"}' --priority 5
taskhive worker myapp.tasks:registry --concurrency 8 --queues emails reports
taskhive stats --json
taskhive ls --state dead
taskhive requeue --all
taskhive reclaim
taskhive purge --states succeeded --older-than 86400
```

## Tests

```bash
pytest                              # 107 tests
pytest --cov --cov-report=term      # 92% coverage
mypy                                # strict, clean
ruff check .                        # clean
```

Four kinds of test, because they catch different things:

- **Example tests** for the behavior the API promises.
- **Concurrency tests** with real threads on a real file, since a mocked
  database cannot demonstrate that a lock works.
- **Property tests** with Hypothesis, asserting invariants over generated
  inputs: a task never runs more than `max_attempts` times, every enqueued task
  reaches a terminal state exactly once, backoff never exceeds its cap.
- **Compatibility tests** that force the pre-3.35 SQLite claim path, which
  cannot run on a modern build and would otherwise rot unnoticed.

### Five bugs the tests and benchmarks caught

Worth recording, because they are the kind that survive code review:

1. `lease_seconds=0.0` silently fell back to the 60 second default, because the
   code used `lease_seconds or default` and `0.0` is falsy. An explicit `None`
   check fixed it. Caught by a test that asked for an already-expired lease.
2. SQLite's `RETURNING` makes no ordering guarantee. The claim selected the
   right tasks but handed them back in arbitrary order, so a batch was not
   processed highest priority first. Caught by a Hypothesis property, not by
   reading the documentation.
3. The worker could exit the instant an in-flight batch finished, without
   re-polling for work that batch had made available. Caught by a concurrency
   test that failed a handler and expected the retry to run.
4. A later edit removed the line that reset the poll interval after a successful
   claim, so the idle backoff kept growing while the queue was draining. Every
   test still passed, because they use a handful of tasks each. Caught by the
   benchmark, and now guarded by a test that drains 200 tasks against a clock.
5. **Any timed wait in the claim loop is a trap on Windows.** End to end
   throughput there was 64 tasks a second while raw lease-and-ack managed 2,900,
   so the runtime was burning about 15.9ms per task doing nothing. Measurement,
   not intuition, found it: `threading.Event.wait(0.001)` takes **15.6ms** on
   Windows while `time.sleep(0.001)` takes 1.6ms, because `Event.wait` inherits
   the system timer granularity and `sleep` uses a high resolution timer.
   `concurrent.futures.wait` is built on `Event.wait`, so replacing the sleep
   with it changed nothing. The loop now blocks on a semaphore that a completion
   callback releases: an untimed acquire has no granularity floor, since the
   release wakes it directly.

Getting there took three wrong guesses, which is the part worth keeping. First
suspicion was thread churn from a per-task heartbeat thread; replacing it with a
single sweeper was correct on its own merits and moved the number not at all.
Second was the poll sleep; swapping in `concurrent.futures.wait` moved it not at
all either, for the reason above. Only instrumenting every blocking call in the
loop, on the machine that actually had the problem, identified it: one 15.6ms
wait per task, at every level of concurrency. `examples/diagnose.py` is that
instrumentation, kept in the repository.

The broader lesson: a test suite that only exercises small inputs on one
operating system will not tell you the throughput collapsed. Benchmarks and a
cross-platform CI matrix are part of the tests, not decoration. And a
performance fix that is not measured on the platform that was slow is a guess.

## When not to use this

- **Many machines.** SQLite is one file. Network filesystems and SQLite are a
  known bad combination. Use Postgres or a real broker.
- **Very high throughput.** Thousands of tasks a second is fine; hundreds of
  thousands is not what this is for.
- **Fan-out or pub/sub.** This is a work queue, not a message bus.

Within one machine, with durability and retries mattering more than raw scale,
it is a good trade.

## License

MIT
