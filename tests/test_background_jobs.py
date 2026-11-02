"""Test actual background SO100 inference and cancellation separately from browser acceptance."""

from hashlib import sha256
import json
import os
from pathlib import Path
import time

import pytest
from streamlit.testing.v1 import AppTest

from vla_visualizer.config import ROOT
from vla_visualizer.execution.jobs import TERMINAL, read_json


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("ECE525_BACKGROUND_REAL") != "1", reason="Requires actual cache-backed model jobs")
def test_actual_background_ui_and_cancel(monkeypatch):
    from vla_visualizer.ui.workflow import manager
    validation_dir = Path(os.environ.get('ECE525_VALIDATION_DIR', ROOT / 'outputs/jobs_validation'))
    output = validation_dir / 'ui_jobs'
    output.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VLA_ARTIFACT_DIR", str(output))
    manager.clear()
    app = AppTest.from_file(ROOT / "app.py", default_timeout=60).run()
    assert not app.exception
    app.button(key="start_job").click().run()
    assert not app.exception
    job = app.session_state["active_job"]
    owner = app.session_state["job_owner"]
    seen = []
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        status = manager().status(job, owner)
        seen.append(status["state"])
        if status["state"] in TERMINAL:
            break
        time.sleep(.5)
        app.run()
        assert not app.exception
    assert status["state"] == "completed", status
    app.run()
    assert not app.exception and not app.error
    directory = manager().directory(job)
    payload = read_json(directory / "result.json")
    assert payload["metadata"]["native_action_shape"] == [1, 50, 6]
    assert payload["intermediates"]["hook_count_after"] == 0
    replay_hash = sha256((directory / "replay.json").read_bytes()).hexdigest()
    count = len(list(output.glob("*/status.json")))
    for stage in ("state_normalize", "language_tokens", "prefix", "conditioning", "actions"):
        app.selectbox(key="_edu_stage").select(stage).run()
        assert not app.exception
    app.slider(key="_edu_action").set_value(49).run()
    assert not app.exception
    assert sha256((directory / "replay.json").read_bytes()).hexdigest() == replay_hash
    assert len(list(output.glob("*/status.json"))) == count
    app.button(key="start_job").click().run()
    cancelled = app.session_state["active_job"]
    assert cancelled != job
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if manager().status(cancelled, owner)["progress"].get("phase") == "Loading model":
            break
        time.sleep(.1)
    assert manager().status(cancelled, owner)["state"] == "running"
    app.run()
    app.button(key="cancel_job").click().run()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        cancelled_status = manager().status(cancelled, owner)
        if cancelled_status["state"] in TERMINAL:
            break
        time.sleep(.1)
    assert cancelled_status["state"] == "cancelled"
    app.run()
    assert not app.exception
    report = {
        "status": "PASS", "scope": "Real AppTest/background worker execution; not browser visual acceptance.",
        "run_id": job, "cancelled_run_id": cancelled, "states_seen": sorted(set(seen)),
        "job_status": status, "cancel_status": cancelled_status,
        "replay_operations_additional_jobs": 0, "replay_sha256": replay_hash,
        "ui_tree_screenshot": None,
    }
    (validation_dir / 'ui_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
