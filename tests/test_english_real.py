"""Opt-in real English UI inference, tail-mask training, and fresh-worker reconstruction."""

import os
from pathlib import Path
import time

import pytest
from streamlit.testing.v1 import AppTest

from vla_visualizer.config import ROOT
from vla_visualizer.execution.jobs import JobConfig, TERMINAL, atomic_json, read_json


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("ECE525_ENGLISH_REAL") != "1",
    reason="Requires the actual pinned model and prepared episode 0",
)
def test_actual_english_workflows(monkeypatch):
    from vla_visualizer.ui.workflow import manager
    from vla_visualizer.inference.replay import select_records

    out = Path(os.environ.get("ECE525_VALIDATION_DIR", str(ROOT / "outputs/english_revision")))
    out.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VLA_ARTIFACT_DIR", str(out / "variant_jobs"))
    manager.clear()

    def wait(job, owner):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            status = manager().status(job, owner)
            if status["state"] in TERMINAL:
                assert status["state"] == "completed", status
                return status
            time.sleep(0.25)
        raise AssertionError("Real worker did not finish within the validation deadline")

    app = AppTest.from_file(ROOT / "app.py", default_timeout=45).run()
    assert not app.exception
    app.selectbox(key="education_scenario").select(
        "SO100 · dataset episode/frame exploration"
    ).run()
    app.slider(key="dataset_frame").set_value(120).run()
    instruction = "Move the cube into the box."
    app.selectbox(key="instruction_example").select(instruction).run()
    app.button(key="start_job").click().run()
    assert not app.exception
    job = app.session_state["active_job"]
    owner = app.session_state["job_owner"]
    wait(job, owner)
    app.run()
    assert not app.exception
    capture = read_json(manager().directory(job) / "result.json")
    token = select_records(capture, ["language_tokenization"])[0]
    embedded = select_records(capture, ["language"])[0]
    assert token["context"]["task"] == [instruction + "\n"]
    assert token["outputs"]["token_ids"]["values"] == embedded["inputs"]["tensor"]["values"]
    assert capture["metadata"]["observation_provenance"]["frame_index"] == 120
    snapshot = app.session_state["education_run"].snapshot
    count = len(list(manager().root.glob("*/status.json")))
    app.selectbox(key="_edu_stage").select("loop_next").run()
    app.button(key="edu_next").click().run()
    assert not app.exception
    assert app.session_state["edu_stage"] == "action_state" and app.session_state["edu_flow"] == 1
    app.selectbox(key="instruction_example").select(
        "Pick up the cube and place it in the box."
    ).run()
    assert app.session_state["education_run"].snapshot == snapshot
    assert len(list(manager().root.glob("*/status.json"))) == count
    # A single true target at the last frame; padding must not become supervision.
    training = manager().submit(
        JobConfig(
            kind="training",
            training_steps=1,
            observation_mode="dataset",
            episode_index=0,
            frame_index=453,
        ),
        owner,
    )
    training_status = wait(training, owner)
    td = manager().directory(training)
    audit = read_json(td / "training_audit.json")
    trained = read_json(td / "result.json")
    assert audit["padding_mask"] == [[False] + [True] * 49]
    assert audit["recorded_instruction"] == "Pick up the cube and place it in the box."
    assert trained["checkpoint_reload_verified"] and trained["frozen_parameters_preserved"]
    evaluation = manager().submit(JobConfig(kind="evaluate", checkpoint_job=training), owner)
    evaluation_status = wait(evaluation, owner)
    evaluated = read_json(manager().directory(evaluation) / "result.json")
    assert evaluated["metadata"]["observation_provenance"]["frame_index"] == 453
    assert evaluated["metadata"]["post_training_probe_comparison"]["max_absolute_difference"] == 0
    assert (
        manager().submit(JobConfig(kind="evaluate", checkpoint_job=training), owner) == evaluation
    )
    atomic_json(
        out / "english_real_workflows.json",
        {
            "status": "PASS_REAL_WORKERS_AND_APPTEST",
            "scope": "Not browser visual acceptance",
            "instruction": instruction,
            "native_task": token["context"]["task"],
            "tokenizer_model_ids_equal": True,
            "inference_run_id": job,
            "replay_additional_jobs": 0,
            "training_run_id": training,
            "evaluation_run_id": evaluation,
            "training_status": training_status,
            "evaluation_status": evaluation_status,
            "tail_valid_targets": 1,
            "tail_masked_targets": 49,
            "dataset_instruction_preserved": True,
            "fresh_worker_max_absolute_difference": 0,
            "repeated_evaluation_reused": True,
        },
    )
    manager.clear()
