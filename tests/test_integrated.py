"""Test artifact mapping and model-free UI regression separately from browser acceptance."""

from copy import deepcopy
import json
import os

import pytest
from streamlit.testing.v1 import AppTest

from vla_visualizer.config import ROOT
from vla_visualizer.execution.jobs import JobConfig, JobManager
from vla_visualizer.execution.storage import StorageBudgetError, budgets, preflight, usage
from vla_visualizer.inference.replay import load_run
from vla_visualizer.ui.education_views import STAGES, pipeline_visuals
from vla_visualizer.ui.execution_flow import call_events, iterations, step_position


def full_data():
    return load_run((ROOT / "assets/education/full_loop_replay.json").read_bytes()).read()


def test_full_loop_transition_and_sparse_compatibility():
    data = full_data()
    assert iterations(data) == list(range(10))
    assert step_position("loop_next", 0, list(STAGES), iterations(data), 1) == ("action_state", 1)
    assert step_position("loop_next", 9, list(STAGES), iterations(data), 1) == ("actions", 9)
    assert step_position("action_state", 4, list(STAGES), iterations(data), -1) == ("loop_next", 3)
    assert step_position("loop_next", 0, list(STAGES), [0, 4, 9], 1) == ("action_state", 4)
    old = deepcopy(data)
    del old["capture"]["intermediates"]["extended_capture"]
    assert iterations(old) == [0, 4, 9]


def test_actual_token_sources_and_frozen_hover_data():
    data = full_data()
    before = json.dumps(data, sort_keys=True)
    visuals = pipeline_visuals(data, "pending instruction", 4)
    for key, size in [
        ("vision_top", 1024),
        ("connector_top", 64),
        ("vision_wrist", 1024),
        ("connector_wrist", 64),
    ]:
        item = visuals[key]
        assert len(item["norms"]) == size
        assert len(item["features"]["values"]) == size
        assert item["mapping"]["camera"] in data["view_spec"]["camera_to_native"]
        assert item["mapping"]["record_id"] == item["features"]["record_id"]
        assert len(item["features"]["channel_indices"]) == 16
        assert item["input_image"].startswith("data:image/png;base64,")
    assert len(visuals["action_state"]["full_vectors"]) == 50
    assert len(visuals["action_state"]["full_vectors"][0]) == 32
    assert len(visuals["actions"]["full_vectors"][0]) == 6
    assert visuals["expert"]["features"]["stage"] == "expert_hidden"
    assert visuals["expert"]["mapping"]["flow_iteration"] == 4
    assert json.dumps(data, sort_keys=True) == before
    assert [e["stage"] for e in call_events(data)][:4] == [
        "top_vision",
        "top_connector",
        "wrist_vision",
        "wrist_connector",
    ]
    assert (
        "norms" not in pipeline_visuals(data, "", 3)["expert"]
        or not pipeline_visuals(data, "", 3)["expert"]["norms"]
    )


def test_storage_categories_dedup_and_failure_diagnostic(tmp_path, monkeypatch):
    root = tmp_path / "jobs"
    root.mkdir()
    a = root / "checkpoint"
    a.mkdir()
    (a / "delta.safetensors").write_bytes(b"1234")
    os.link(a / "delta.safetensors", a / "duplicate.safetensors")
    (root / "symlink").symlink_to(a / "delta.safetensors")
    assert usage(root)["checkpoint"] == 4
    assert budgets("training")["checkpoint"] == 256 * 1024
    assert preflight(root, JobConfig(kind="training"))["base_cache_included"] is False
    monkeypatch.setenv("VLA_TOTAL_MB", "64")
    with (root / "large.mp4").open("wb") as handle:
        handle.truncate(63 * 1024**2)
    with pytest.raises(StorageBudgetError) as exc:
        JobManager(root).submit(JobConfig(kind="training"), "owner")
    assert exc.value.report["failed_stage"] == "total_job_quota"
    assert exc.value.report["deficit_bytes"] > 0
    assert list((root / "submission_failures").glob("*.json"))
    assert (root / "large.mp4").exists()


def test_connector_geometry_matches_installed_pixel_shuffle():
    import torch
    from transformers.models.idefics3.modeling_idefics3 import Idefics3Connector

    positions = torch.arange(32 * 32).reshape(1, 32 * 32, 1)
    shuffled = Idefics3Connector.pixel_shuffle(None, positions, 4)[0]
    for token in range(64):
        y, x = divmod(token, 8)
        expected = sorted((y * 4 + dy) * 32 + x * 4 + dx for dy in range(4) for dx in range(4))
        assert sorted(shuffled[token].tolist()) == expected


def test_separate_worker_readiness_fails_explicitly(tmp_path, monkeypatch):
    from vla_visualizer.simulation.compatibility import worker_readiness

    monkeypatch.setenv("VLA_SIM_PYTHON", str(tmp_path / "missing-worker"))
    result = worker_readiness()
    assert result["status"] == "blocked" and not result["integration_verified"]
    assert "FileNotFoundError" in result["blockers"][0]


def test_delta_checkpoint_roundtrip_and_corruption(tmp_path):
    from types import SimpleNamespace
    import torch
    from vla_visualizer.adapters.smolvla import CHECKPOINT_FILES, file_sha256
    from vla_visualizer.execution.checkpoints import save_delta, load_trained_checkpoint

    policy = torch.nn.Module()
    policy.model = torch.nn.Module()
    policy.model.action_out_proj = torch.nn.Linear(720, 32)
    policy.model.frozen = torch.nn.Linear(2, 2)
    source = tmp_path / "base"
    source.mkdir()
    for name in CHECKPOINT_FILES:
        (source / name).write_bytes(b"focused unit-test base; not integration evidence")
    hashes = {name: file_sha256(source / name) for name in CHECKPOINT_FILES}
    expected = policy.model.action_out_proj.weight.detach().clone()
    directory = tmp_path / "checkpoint"
    save_delta(directory, policy, hashes, {})
    with torch.no_grad():
        policy.model.action_out_proj.weight.zero_()
    adapter = SimpleNamespace(_policy=policy, _snapshot_path=source)
    assert load_trained_checkpoint(adapter, directory)["kind"] == "delta"
    assert torch.equal(expected, policy.model.action_out_proj.weight)
    manifest = directory / "checkpoint_manifest.json"
    data = json.loads(manifest.read_text())
    data["base_artifact_sha256"] = {}
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="All base-artifact hashes"):
        load_trained_checkpoint(adapter, directory)


def test_dataset_real_frames_mask_and_ui_no_model_work(monkeypatch):
    from vla_visualizer.data.so100 import catalog, observation, target_chunk

    if "0" not in catalog()["episodes"]:
        pytest.skip("Requires a locally prepared episode")
    first = observation(0, 0)
    later = observation(0, 120)
    for camera in first.images:
        assert first.images[camera].tobytes() != later.images[camera].tobytes()
    assert first.state != later.state
    assert later.provenance["timestamp_seconds"] == pytest.approx(4.0)
    _, mask = target_chunk(0, 453)
    assert mask.tolist() == [False] + [True] * 49

    def forbidden(*args, **kwargs):
        raise AssertionError("Exploration, hover or replay attempted a model job")

    monkeypatch.setattr(JobManager, "submit", forbidden)
    app = AppTest.from_file(ROOT / "app.py", default_timeout=40).run()
    app.selectbox(key="education_scenario").select("SO100 · dataset episode/frame exploration").run()
    assert not app.exception
    app.slider(key="dataset_frame").set_value(120).run()
    assert app.session_state["education_observation"].provenance["frame_index"] == 120
    assert not app.exception
    app.button(key="saved_full_loop").click().run()
    app.selectbox(key="_edu_stage").select("loop_next").run()
    app.button(key="edu_next").click().run()
    assert app.session_state["edu_stage"] == "action_state"
    assert app.session_state["edu_flow"] == 1
    app.selectbox(key="_edu_flow").select(9).run()
    app.selectbox(key="_edu_stage").select("loop_next").run()
    app.button(key="edu_next").click().run()
    assert app.session_state["edu_stage"] == "actions"
    app.radio(key="_edu_view").set_value("Recorded invocation order").run()
    app.button(key="edu_reset").click().run()
    app.button(key="edu_next").click().run()
    assert app.session_state["edu_stage"] == "top_connector"
    assert not app.exception
