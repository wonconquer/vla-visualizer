"""Bounded logical archives preserve source artifacts and evaluation references."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import zipfile

import pytest

from vla_visualizer.execution.jobs import (
    JobConfig,
    JobManager,
    acquire_lease,
    atomic_json,
    read_json,
)


def old_job(manager, number=1, state="completed", kind="training"):
    job = f"{number:032x}"
    d = manager.directory(job)
    d.mkdir()
    atomic_json(
        d / "status.json",
        {
            "run_id": job,
            "state": state,
            "kind": kind,
            "owner_hash": sha256(b"owner").hexdigest(),
            "created_unix": number,
        },
    )
    atomic_json(d / "config.json", {"kind": kind})
    atomic_json(d / "result.json", {"saved_weight_sha256": "fixture-sha"})
    return job


def test_recent_window_archives_without_moving_checkpoint_and_allows_evaluation(
    tmp_path, monkeypatch
):
    m = JobManager(tmp_path)
    parent = old_job(m)
    d = m.directory(parent)
    (d / "checkpoint").mkdir()
    atomic_json(d / "checkpoint/checkpoint_manifest.json", {"kind": "delta", "unit_fixture": True})
    payload = d / "checkpoint/delta.safetensors"
    payload.write_bytes(b"unit-fixture-not-real-weights")
    before = {p: p.read_bytes() for p in d.rglob("*") if p.is_file()}
    monkeypatch.setenv("VLA_MAX_JOBS", "1")
    launches = []
    monkeypatch.setattr(
        "vla_visualizer.execution.jobs.subprocess.Popen", lambda *a, **kw: launches.append(a)
    )
    primary = m.submit(JobConfig(), "owner")
    assert m.status(parent, "owner")["archived"]
    assert not m.status(primary, "owner")["archived"]
    assert m.inventory()["recent_primary_records"] == 1
    assert m.inventory()["retained_records"] == 2
    assert m.inventory()["archived"] == 1
    child = m.submit(JobConfig(kind="evaluate", checkpoint_job=parent), "owner")
    config = read_json(m.directory(child) / "config.json")
    assert config["parent_training_run_id"] == parent
    assert config["checkpoint_directory"] == str(d / "checkpoint")
    assert any(r["run_id"] == parent for r in m.completed("owner", "training"))
    assert all(p.read_bytes() == value for p, value in before.items())
    with zipfile.ZipFile(BytesIO(m.export(parent, "owner"))) as archive:
        assert archive.read("checkpoint/delta.safetensors") == payload.read_bytes()
    with pytest.raises(PermissionError):
        m.status(parent, "different-owner")
    assert len(launches) == 2


def test_archived_records_still_hit_the_total_record_bound(tmp_path, monkeypatch):
    m = JobManager(tmp_path)
    one = old_job(m, 1)
    old_job(m, 2)
    atomic_json(tmp_path / ".record_retention.json", {"schema_version": 1, "archived": {one: {}}})
    monkeypatch.setenv("VLA_MAX_RECORDS", "2")
    with pytest.raises(RuntimeError, match="Retained record limit reached: 2 / 2"):
        m.submit(JobConfig(), "owner")
    assert m.inventory()["retained_records"] == 2


def test_archive_cannot_bypass_bytes_and_does_not_commit_on_preflight_failure(
    tmp_path, monkeypatch
):
    from vla_visualizer.execution.storage import StorageBudgetError

    m = JobManager(tmp_path)
    parent = old_job(m)
    with (m.directory(parent) / "old.safetensors").open("wb") as f:
        f.truncate(64 * 1024**2)
    monkeypatch.setenv("VLA_MAX_JOBS", "1")
    monkeypatch.setenv("VLA_TOTAL_MB", "64")
    with pytest.raises(StorageBudgetError) as exc:
        m.submit(JobConfig(), "owner")
    assert exc.value.report["used_bytes"] >= 64 * 1024**2
    assert not m.status(parent, "owner")["archived"]
    assert len(list(tmp_path.glob("*/status.json"))) == 1
    assert list((tmp_path / "submission_failures").glob("*.json"))


def test_failed_launch_rolls_back_archive_and_releases_lease(tmp_path, monkeypatch):
    m = JobManager(tmp_path)
    parent = old_job(m)
    monkeypatch.setenv("VLA_MAX_JOBS", "1")

    def fail(*args, **kwargs):
        raise OSError("fixture process launch failure")

    monkeypatch.setattr("vla_visualizer.execution.jobs.subprocess.Popen", fail)
    with pytest.raises(OSError, match="fixture process launch failure"):
        m.submit(JobConfig(), "owner")
    assert not m.status(parent, "owner")["archived"]
    assert m.inventory()["by_status"] == {"completed": 1, "failed": 1}
    acquire_lease(tmp_path).close()


def test_no_archive_when_another_model_owns_lease(tmp_path, monkeypatch):
    m = JobManager(tmp_path)
    parent = old_job(m)
    monkeypatch.setenv("VLA_MAX_JOBS", "1")
    lease = acquire_lease(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="Another model job"):
            m.submit(JobConfig(), "owner")
        assert not m.status(parent, "owner")["archived"]
    finally:
        lease.close()


def test_concurrent_admission_keeps_total_bound_and_never_archives_active_job(
    tmp_path, monkeypatch
):
    m = JobManager(tmp_path)
    parent = old_job(m)
    monkeypatch.setenv("VLA_MAX_JOBS", "1")
    monkeypatch.setenv("VLA_MAX_RECORDS", "2")
    launches = []
    monkeypatch.setattr(
        "vla_visualizer.execution.jobs.subprocess.Popen", lambda *a, **kw: launches.append(a)
    )

    def submit(_):
        try:
            return m.submit(JobConfig(), "owner")
        except RuntimeError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, range(2)))
    assert sum("Retained record limit" in r for r in results) == 1
    assert len(launches) == 1 and m.inventory()["retained_records"] == 2
    assert m.status(parent, "owner")["archived"]
    assert m.inventory()["active"] == 1


def test_unfinished_records_are_not_archived(tmp_path, monkeypatch):
    m = JobManager(tmp_path)
    parent = old_job(m, state="starting")
    monkeypatch.setenv("VLA_MAX_JOBS", "1")
    with pytest.raises(RuntimeError, match="unfinished jobs"):
        m.submit(JobConfig(), "owner")
    assert not m.status(parent, "owner")["archived"]


def test_repeated_runs_roll_the_window_without_losing_cancelled_or_failed_results(
    tmp_path, monkeypatch
):
    m = JobManager(tmp_path)
    original = [old_job(m, 1), old_job(m, 2)]
    monkeypatch.setenv("VLA_MAX_JOBS", "2")
    monkeypatch.setenv("VLA_MAX_RECORDS", "8")
    monkeypatch.setattr("vla_visualizer.execution.jobs.subprocess.Popen", lambda *a, **kw: None)
    for step in range(6):
        job = m.submit(JobConfig(), "owner")
        d = m.directory(job)
        state = ["completed", "failed", "cancelled"][step % 3]
        if state == "cancelled":
            m.cancel(job, "owner")
            assert (d / "cancel").exists()
        atomic_json(d / "status.json", {**read_json(d / "status.json"), "state": state})
        inv = m.inventory()
        assert inv["recent_primary_records"] == 2
        assert inv["retained_records"] == step + 3
        assert inv["archived"] == step + 1
    assert all(m.directory(job).is_dir() and m.status(job, "owner")["archived"] for job in original)
    with pytest.raises(RuntimeError, match="Retained record limit reached: 8 / 8"):
        m.submit(JobConfig(), "owner")
