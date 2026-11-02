"""Exercise UI selection through worker inference and replay; browser acceptance is separate."""

import os
import time

import pytest
from streamlit.testing.v1 import AppTest

from vla_visualizer.config import ROOT


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("ECE525_INTEGRATED_REAL") != "1", reason="Requires the pinned model cache"
)
def test_actual_ui_selection_and_replay(monkeypatch):
    from vla_visualizer.execution.jobs import TERMINAL, read_json, atomic_json
    from vla_visualizer.ui.workflow import manager
    from vla_visualizer.inference.replay import select_records

    destination = ROOT / "outputs/integration/ui_jobs"
    monkeypatch.setenv("VLA_ARTIFACT_DIR", str(destination))
    manager.clear()
    app = AppTest.from_file(ROOT / "app.py", default_timeout=45).run()
    app.selectbox(key="education_scenario").select("SO100 · dataset episode/frame exploration").run()
    app.slider(key="dataset_frame").set_value(120).run()
    instruction = "Move the cube into the box."
    app.selectbox(key="instruction_example").select(instruction).run()
    app.button(key="start_job").click().run()
    assert not app.exception
    job = app.session_state["active_job"]
    owner = app.session_state["job_owner"]
    app.run()
    assert not app.exception
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        status = manager().status(job, owner)
        if status["state"] in TERMINAL:
            break
        time.sleep(0.25)
    assert status["state"] == "completed", status
    app.run()
    assert not app.exception
    directory = manager().directory(job)
    config = read_json(directory / "config.json")
    capture = read_json(directory / "result.json")
    replay = app.session_state["education_run"]
    data = replay.read()
    token = select_records(capture, ["language_tokenization"])[0]
    embedded = select_records(capture, ["language"])[0]
    assert config["instruction"] == data["manifest"]["instruction"] == instruction
    assert token["context"]["task"] == [instruction + "\n"]
    assert token["outputs"]["token_ids"]["values"] == embedded["inputs"]["tensor"]["values"]
    assert capture["metadata"]["observation_provenance"]["frame_index"] == 120
    count = len(list(destination.glob("*/status.json")))
    snapshot = replay.snapshot
    for stage in [
        "language_embedding",
        "state_project",
        "action_state",
        "expert",
        "velocity",
        "solver_update",
        "loop_next",
        "actions",
    ]:
        app.selectbox(key="_edu_stage").select(stage).run()
        assert not app.exception
    app.button(key="edu_reset").click().run()
    app.button(key="edu_play").click().run()
    app.session_state["edu_tick"] = 0.0
    app.run()
    app.button(key="edu_pause").click().run()
    assert app.session_state["education_run"].snapshot == snapshot
    assert len(list(destination.glob("*/status.json"))) == count
    app.selectbox(key="workflow_mode").select("Training").run()
    app.button(key="saved_training").click().run()
    for stage in ["forward", "objective", "loss", "backward", "update", "next"]:
        app.selectbox(key="_edu_stage").select(stage).run()
        assert not app.exception
    assert app.session_state["education_training"]["result"]["frozen_parameters_preserved"]
    atomic_json(
        ROOT / "outputs/integration/actual_ui_validation.json",
        {
            "status": "PASS_REAL_WORKER_AND_APPTEST_ONLY",
            "job_run_id": job,
            "directory": str(directory),
            "instruction": instruction,
            "native_task": token["context"]["task"],
            "episode": 0,
            "frame": 120,
            "replay_additional_jobs": 0,
            "snapshot_unchanged": True,
            "training_stages_rendered": 6,
            "browser_acceptance": "BLOCKED; AppTest is not browser visual acceptance",
        },
    )
    manager.clear()
