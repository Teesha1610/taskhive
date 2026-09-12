"""CLI. Exercised through main() so argument parsing is covered too."""

from __future__ import annotations

import json

import pytest

from taskhive import TaskQueue, TaskState
from taskhive.cli import main


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "cli.db")


def run(args: list[str]) -> int:
    return main(args)


def test_enqueue_and_stats(db, capsys):
    assert run(["--db", db, "enqueue", "send_email", '{"to": "a@example.com"}']) == 0
    assert "enqueued task 1" in capsys.readouterr().out

    assert run(["--db", db, "stats", "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["pending"] == 1


def test_enqueue_rejects_bad_json(db, capsys):
    assert run(["--db", db, "enqueue", "job", "{not json}"]) == 1
    assert "not valid JSON" in capsys.readouterr().err


def test_enqueue_flags_are_applied(db, capsys):
    run(["--db", db, "enqueue", "job", "--priority", "7", "--max-attempts", "9",
         "--queue", "emails", "--unique-key", "k", "--json"])
    task = json.loads(capsys.readouterr().out)
    assert task["priority"] == 7
    assert task["max_attempts"] == 9
    assert task["queue"] == "emails"
    assert task["unique_key"] == "k"


def test_list_and_cancel(db, capsys):
    run(["--db", db, "enqueue", "job"])
    capsys.readouterr()

    assert run(["--db", db, "ls"]) == 0
    assert "pending" in capsys.readouterr().out

    assert run(["--db", db, "cancel", "1"]) == 0
    assert "cancelled task 1" in capsys.readouterr().out

    assert run(["--db", db, "cancel", "1"]) == 1, "cancelling twice should fail"
    capsys.readouterr()


def test_requeue_dead_tasks(db, capsys):
    queue = TaskQueue(db)
    queue.enqueue("job", max_attempts=1)
    leased = queue.lease("w")[0]
    queue.nack(leased.id, leased.lease_token, error="x")
    queue.close()

    assert run(["--db", db, "ls", "--state", "dead"]) == 0
    assert "dead" in capsys.readouterr().out

    assert run(["--db", db, "requeue", "--all"]) == 0
    assert "requeued 1" in capsys.readouterr().out

    queue = TaskQueue(db)
    assert queue.get(1).state is TaskState.PENDING
    queue.close()


def test_reclaim_and_purge(db, capsys):
    queue = TaskQueue(db)
    queue.enqueue("job")
    leased = queue.lease("w", lease_seconds=0.0)[0]
    queue.close()

    assert run(["--db", db, "reclaim"]) == 0
    assert "reclaimed 1" in capsys.readouterr().out

    # The reclaimed task carries the default backoff, so it is pending but not
    # yet available. Lease from a point in the future rather than sleeping.
    from datetime import timedelta

    from taskhive.models import utcnow

    queue = TaskQueue(db)
    leased = queue.lease("w", now=utcnow() + timedelta(seconds=120))[0]
    queue.ack(leased.id, leased.lease_token)
    queue.close()

    assert run(["--db", db, "purge", "--states", "succeeded"]) == 0
    assert "deleted 1" in capsys.readouterr().out


def test_worker_command_runs_handlers(db, tmp_path, capsys, monkeypatch):
    module = tmp_path / "handlers_under_test.py"
    module.write_text(
        "from taskhive import TaskRegistry\n"
        "registry = TaskRegistry()\n"
        "\n"
        "@registry.task('greet')\n"
        "def greet(name):\n"
        "    return {'greeted': name}\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    run(["--db", db, "enqueue", "greet", '{"name": "world"}'])
    capsys.readouterr()

    assert run(["--db", db, "worker", "handlers_under_test:registry", "--until-empty"]) == 0
    output = capsys.readouterr().out
    assert "greet" in output

    queue = TaskQueue(db)
    assert queue.get(1).result == {"greeted": "world"}
    queue.close()


def test_worker_rejects_a_bad_registry(db, capsys):
    with pytest.raises(SystemExit):
        run(["--db", db, "worker", "not_a_module:registry", "--until-empty"])


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        run(["--version"])
    assert exc.value.code == 0
    assert "taskhive" in capsys.readouterr().out
