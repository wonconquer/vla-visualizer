"""State lifecycle, bounded evaluation admission, and English presentation regressions."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import re

from streamlit.testing.v1 import AppTest

from vla_visualizer.config import ROOT
from vla_visualizer.execution.jobs import JobConfig, JobManager, atomic_json
from vla_visualizer.ui.education_state import ensure_selection, widget_value


def test_persistent_state_survives_widget_cleanup_and_partial_lists():
    state = {}
    assert ensure_selection(state, []) == [] and state["edu_stage"] is None
    ensure_selection(state, ["a", "b"], [4, 9], identity="one")
    assert state["edu_stage"] == "a" and state["edu_flow"] == 4
    state["edu_stage"] = "b"
    key = widget_value(state, "edu_stage")
    del state[key]
    ensure_selection(state, ["a", "b"], [4, 9], identity="one")
    assert state["edu_stage"] == "b"
    ensure_selection(state, ["train"], identity="two")
    ensure_selection(state, ["a", "b"], [4, 9], identity="one")
    assert state["edu_stage"] == "b"
    state["edu_stage"] = "invalid"
    state["edu_flow"] = 999
    ensure_selection(state, ["b"], [9], identity="one")
    assert state["edu_stage"] == "b" and state["edu_flow"] == 9


def test_ui_state_loss_mode_switch_and_empty_stages(monkeypatch):
    import vla_visualizer.ui.workflow as workflow

    def forbidden(*args, **kwargs):
        raise AssertionError("Visualization-only interaction submitted a job")

    monkeypatch.setattr(JobManager, "submit", forbidden)
    app = AppTest.from_file(ROOT / "app.py", default_timeout=40).run()
    assert not app.exception
    del app.session_state["edu_stage"]
    app.run()
    assert not app.exception
    app.button(key="saved_full_loop").click().run()
    snapshot = app.session_state["education_run"].snapshot
    app.selectbox(key="_edu_stage").select("velocity").run()
    app.selectbox(key="workflow_mode").select("Training").run()
    app.button(key="saved_training").click().run()
    app.selectbox(key="_edu_stage").select("update").run()
    # A token pin event must not reset training to its forward stage.
    with monkeypatch.context() as patch:
        patch.setattr(
            workflow,
            "_pipeline",
            lambda **kwargs: {"token": {"index": 0}, "nonce": "pin-regression"},
        )
        app.run()
        assert app.session_state["edu_stage"] == "update"
    app.selectbox(key="workflow_mode").select("Simulation").run()
    app.selectbox(key="workflow_mode").select("Explore / Inference").run()
    assert not app.exception and app.session_state["edu_stage"] == "velocity"
    assert app.session_state["education_run"].snapshot == snapshot
    app.session_state["edu_stage"] = "unknown-stage"
    app.run()
    assert not app.exception
    assert app.session_state["edu_stage"] in workflow.STAGES
    with monkeypatch.context() as patch:
        patch.setattr(workflow, "STAGES", {})
        app.run()
        assert not app.exception
        assert any("No educational stages" in e.value for e in app.info)
    app.run()
    assert not app.exception
    for element in [*app.title, *app.caption, *app.info, *app.warning, *app.button]:
        value = getattr(element, "label", None) or getattr(element, "value", "")
        if isinstance(value, str):
            assert not re.search("[\uac00-\ud7a3]", value), value


def test_evaluation_capacity_and_concurrent_dedup(tmp_path, monkeypatch):
    manager = JobManager(tmp_path)
    owner = "session"
    parent = "1" * 32
    d = manager.directory(parent)
    d.mkdir()
    atomic_json(
        d / "status.json",
        {
            "run_id": parent,
            "kind": "training",
            "state": "completed",
            "owner_hash": sha256(owner.encode()).hexdigest(),
        },
    )
    atomic_json(d / "config.json", {"kind": "training"})
    atomic_json(d / "result.json", {"saved_weight_sha256": "unit-fixture-hash"})
    monkeypatch.setenv("VLA_MAX_JOBS", "1")
    launches = []
    monkeypatch.setattr(
        "vla_visualizer.execution.jobs.subprocess.Popen", lambda *a, **kw: launches.append(a)
    )
    config = JobConfig(kind="evaluate", checkpoint_job=parent)
    with ThreadPoolExecutor(max_workers=2) as pool:
        children = list(pool.map(lambda _: manager.submit(config, owner), range(2)))
    assert children[0] == children[1] and len(launches) == 1
    child = children[0]
    child_dir = manager.directory(child)
    saved = json.loads((child_dir / "config.json").read_text())
    assert saved["parent_training_run_id"] == parent
    assert saved["checkpoint_sha256"] == "unit-fixture-hash"
    status = json.loads((child_dir / "status.json").read_text())
    status["state"] = "completed"
    atomic_json(child_dir / "status.json", status)
    (child_dir / "replay.json").write_text("{}")
    assert manager.submit(config, owner) == child and len(launches) == 1
    assert manager.inventory()["primary_limit"] == 1
    assert manager.inventory()["evaluation_limit"] == 3
    assert (d / "result.json").exists()


def test_hover_is_frontend_only_and_loop_routing_has_clearance():
    html = (ROOT / "vla_visualizer/ui/pipeline_component/index.html").read_text()
    pin = html.split("function pinEvent()", 1)[1].split("function inspect", 1)[0]
    assert "setComponentValue" not in pin
    assert 'id="hover-preview"' in html and "showPreview" in html
    assert "scrollIntoView" in html
    assert 'id="loop-boundary"' in html
    assert "M1267 600V635H867V445" in html
    assert "M1344 560H1410V710H1382" in html
    assert 'x="984" y="153"' in html
    assert not re.search("[\uac00-\ud7a3]", html)


def test_completed_checkpoint_view_does_not_require_a_baseline_run(tmp_path, monkeypatch):
    """Rendering fixture only; actual fresh-worker checkpoint evidence is validated separately."""
    import vla_visualizer.ui.workflow as workflow

    manager = JobManager(tmp_path)
    owner = "fixture-session"
    parent = "1" * 32
    child = "2" * 32
    training = json.loads((ROOT / "assets/education/training_capture.json").read_text())
    training["result"]["run_id"] = parent
    directory = manager.directory(parent)
    directory.mkdir()
    for key, name in [
        ("audit", "training_audit.json"),
        ("metrics", "metrics.json"),
        ("result", "result.json"),
        ("visualization", "training_visualization.json"),
    ]:
        atomic_json(directory / name, training[key])
    for run, kind in [(parent, "training"), (child, "evaluate")]:
        d = manager.directory(run)
        d.mkdir(exist_ok=True)
        atomic_json(
            d / "status.json",
            {
                "run_id": run,
                "kind": kind,
                "state": "completed",
                "owner_hash": sha256(owner.encode()).hexdigest(),
            },
        )
        atomic_json(d / "config.json", {"kind": kind, "seed": 0, "device": "cpu"})
    raw = json.loads((ROOT / "assets/education/full_loop_replay.json").read_text())
    raw["capture"]["metadata"].update(
        weight_sha256="unit-rendering-fixture", parent_training_run_id=parent
    )
    atomic_json(manager.directory(child) / "replay.json", raw)
    monkeypatch.setattr(workflow, "manager", lambda: manager)
    app = AppTest.from_file(ROOT / "app.py", default_timeout=40)
    app.session_state["job_owner"] = owner
    app.session_state["active_job"] = child
    app.run()
    app.selectbox(key="workflow_mode").select("Training").run()
    assert not app.exception
    assert "education_run" not in app.session_state
    assert (
        app.session_state["education_evaluation"].read()["capture"]["metadata"][
            "parent_training_run_id"
        ]
        == parent
    )
    app.radio(key="_training_result_view").set_value("Training process").run()
    app.radio(key="_training_result_view").set_value("Checkpoint inference").run()
    assert not app.exception


def test_simulator_policy_load_exception_cleans_both_boundaries(tmp_path, monkeypatch):
    """Focused exception fixture; never evidence of actual simulator integration."""
    import numpy as np
    import pytest
    from vla_visualizer.simulation.rollout import run_rollout

    closed = []

    class EnvironmentFixture:
        controller_metadata = {"physics_timestep_seconds": 0.002}

        def reset(self, seed):
            return {"pixels": {"image": np.zeros((4, 4, 3), dtype=np.uint8)}}, {}

        def simulator_time(self):
            return 0.5

        def close(self):
            closed.append("environment")

    class PolicyFixture:
        def __init__(self, *args):
            pass

        def load(self):
            raise RuntimeError("injected policy-load failure")

        def close(self):
            closed.append("policy")

    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.setattr(
        "vla_visualizer.simulation.rollout.preflight", lambda: {"blockers": [], "versions": {}}
    )
    monkeypatch.setattr(
        "vla_visualizer.simulation.environment.LiberoEnvironment", EnvironmentFixture
    )
    monkeypatch.setattr("vla_visualizer.simulation.policy.LiberoPolicy", PolicyFixture)
    with pytest.raises(RuntimeError, match="injected policy-load"):
        run_rollout(
            tmp_path,
            {"device": "cpu", "seed": 0, "action_steps": 10, "max_steps": 20, "max_seconds": 60},
        )
    assert closed == ["environment", "policy"]
    record = json.loads((tmp_path / "rollout.json").read_text())
    assert record["stages"]["E_policy"]["status"] == "FAIL"
    assert record["stages"]["K_cleanup"]["status"] == "PASS"
    assert record["cleanup_completed"] and record["video"] is None
    assert not record["simulator_integration_success"]


def test_video_requires_policy_stepping(tmp_path, monkeypatch):
    from vla_visualizer.simulation.video import encode_rollout

    monkeypatch.setattr(
        "vla_visualizer.simulation.video.subprocess.run",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("No video should be generated")),
    )
    assert (
        encode_rollout(tmp_path, {"frames": [{}, {}], "simulator_integration_success": False})
        is None
    )


def test_unknown_spatial_order_stays_unavailable():
    from vla_visualizer.inference.replay import load_run
    from vla_visualizer.ui.education_views import pipeline_visuals

    raw = json.loads((ROOT / "assets/education/full_loop_replay.json").read_text())
    camera = next(iter(raw["view_spec"]["camera_to_native"]))
    raw["capture"]["intermediates"]["vision"][camera]["ordering"] = "Not recorded"
    data = load_run(json.dumps(raw)).read()
    visual = pipeline_visuals(data, "", 0)
    assert visual["vision_top"]["grid"] is None
    assert visual["vision_top"]["geometry"] is None


def test_training_images_belong_to_the_recorded_experiment():
    from vla_visualizer.ui.workflow import training_observation
    from vla_visualizer.data.so100 import catalog
    import pytest

    if "0" not in catalog()["episodes"]:
        pytest.skip("Prepared episode 0 required for recorded-image lookup")
    item = catalog()["episodes"]["0"]
    record = {
        "audit": {
            "observation_episode": 0,
            "observation_frame": 453,
            "parquet_sha256": item["source_sha256"][item["parquet"]],
            "recorded_instruction": "Pick up the cube and place it in the box.",
        },
        "config": {"observation_mode": "dataset"},
    }
    actual = training_observation(record)
    assert actual.provenance["frame_index"] == 453
