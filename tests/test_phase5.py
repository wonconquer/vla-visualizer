"""Phase 5 boundaries. Unit fixtures do not stand in for simulator acceptance."""

import subprocess
import sys
import time

import pytest
from streamlit.testing.v1 import AppTest

from vla_visualizer.config import ROOT, cache_dir
from vla_visualizer.execution.jobs import (
    JobConfig, JobManager, acquire_lease, atomic_json, TERMINAL,
)


@pytest.mark.parametrize("options", [
    {"kind": "unknown"}, {"training_steps": 4}, {"batch_size": 2},
    {"max_steps": 281}, {"max_seconds": 10000}, {"action_steps": 2},
    {"kind": "evaluate"}, {"seed": -1}, {"device": "all_gpus"},
])
def test_reject_unbounded_jobs(options):
    with pytest.raises(ValueError):
        JobConfig(**options)


def test_cross_process_lease(tmp_path):
    lease = acquire_lease(tmp_path)
    try:
        code = "from pathlib import Path; from vla_visualizer.execution.jobs import acquire_lease; acquire_lease(Path(__import__('sys').argv[1]))"
        result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True)
        assert result.returncode != 0 and "Another model job" in result.stderr
    finally:
        lease.close()
    acquire_lease(tmp_path).close()


def test_session_ownership_export_and_paths(tmp_path):
    from hashlib import sha256
    manager = JobManager(tmp_path)
    job = "1" * 32
    directory = manager.directory(job)
    directory.mkdir()
    atomic_json(directory / "status.json", {
        "state": "completed", "owner_hash": sha256(b"session-a").hexdigest(),
        "kind": "training", "run_id": job,
    })
    atomic_json(directory / "result.json", {"real_fixture": False})
    (directory / "checkpoint").mkdir()
    (directory / "checkpoint/model.safetensors").write_bytes(b"fixture")
    with pytest.raises(PermissionError):
        manager.status(job, "session-b")
    with pytest.raises(ValueError):
        manager.directory("../other")
    import zipfile
    from io import BytesIO
    with zipfile.ZipFile(BytesIO(manager.export(job, "session-a"))) as archive:
        assert archive.namelist() == ["result.json"]


def test_supervisor_failure_releases_lease(tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_WORKER_PYTHON", str(tmp_path / "missing-interpreter"))
    manager = JobManager(tmp_path)
    job = manager.submit(JobConfig(), "owner")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = manager.status(job, "owner")
        if status["state"] in TERMINAL:
            break
        time.sleep(.05)
    assert status["state"] == "failed" and "No such file" in status["error"]
    acquire_lease(tmp_path).close()


def test_portable_cache(monkeypatch):
    monkeypatch.delenv("VLA_MODEL_CACHE", raising=False)
    assert cache_dir() == ROOT / ".cache/huggingface/hub"
    monkeypatch.setenv("VLA_MODEL_CACHE", "outputs/test-cache")
    assert cache_dir() == ROOT / "outputs/test-cache"


def test_primary_ui_selectors_and_recorded_replay_never_infer(monkeypatch, tmp_path):
    from vla_visualizer.inference.runner import InferenceRunner
    from vla_visualizer.execution.jobs import JobManager

    def forbidden(*args, **kwargs):
        raise AssertionError("visualization-only control attempted computation")
    monkeypatch.setattr(InferenceRunner, "load", forbidden)
    monkeypatch.setattr(InferenceRunner, "run_inference", forbidden)
    monkeypatch.setattr(JobManager, "submit", forbidden)
    monkeypatch.setenv("VLA_ARTIFACT_DIR", str(tmp_path))
    app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
    assert not app.exception
    assert app.selectbox(key="education_model").value == "smolvla"
    assert app.selectbox(key="instruction_example").value == "Pick up the cube and place it in the box."
    assert not app.get("graphviz_chart")
    app.selectbox(key="workflow_mode").select("Simulation").run()
    assert not app.exception and app.button(key="start_job").disabled
    app.selectbox(key="workflow_mode").select("Training").run()
    assert not app.exception
    app.selectbox(key="workflow_mode").select("Explore / Inference").run()
    app.button(key="saved_evidence").click().run()
    assert not app.exception
    identity = app.session_state["education_run"].run_id
    app.selectbox(key="_edu_stage").select("state_normalize").run()
    assert not app.exception
    assert app.session_state["education_run"].run_id == identity


def test_training_targets_are_actual_recorded_actions():
    from hashlib import sha256
    from vla_visualizer.execution.training import PARQUET_SHA256
    import pyarrow.parquet as pq
    path = ROOT / "assets/training/so100_episode0.parquet"
    assert sha256(path.read_bytes()).hexdigest() == PARQUET_SHA256
    rows = pq.read_table(path).slice(0, 50).to_pylist()
    assert len(rows) == 50 and rows[0]["action"][0] == .615234375
    assert [r["frame_index"] for r in rows] == list(range(50))


def test_actual_native_libero_processors_with_unit_observation():
    """Use a synthetic observation to test native conversion, not simulator acceptance."""
    import numpy as np
    import torch
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from vla_visualizer.simulation.policy import batch_observation
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    raw = {"pixels": {"image": image, "image2": image.copy()}, "robot_state": {
        "eef": {"pos": np.array([.1, .2, .3]), "quat": np.array([0., 0., 0., 1.])},
        "gripper": {"qpos": np.array([.02, -.02])},
    }}
    pre, post = LiberoEnv().get_env_processors()
    converted = pre(preprocess_observation(batch_observation(raw)))
    assert converted["observation.state"].shape == (1, 8)
    torch.testing.assert_close(converted["observation.state"][0], torch.tensor([.1,.2,.3,0,0,0,.02,-.02]))
    expected = torch.from_numpy(image[::-1, ::-1].copy()).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(converted["observation.images.image"][0], expected)
    action = torch.linspace(-1, 1, 7).unsqueeze(0)
    torch.testing.assert_close(post({"action": action})["action"], action)
