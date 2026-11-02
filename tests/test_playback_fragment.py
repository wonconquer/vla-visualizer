"""Exercise actual Streamlit 1.50 fragment callbacks, including delayed widget echoes.

This is server-side regression coverage, not real-browser visual acceptance.
AppTest normally recreates fragment storage and runs the entire application. This
fixture retains storage and submits fragment RerunData to cover the timer path.
"""

from dataclasses import replace
from time import monotonic

import pytest
from streamlit.runtime.fragment import MemoryFragmentStorage
from streamlit.runtime.scriptrunner_utils.script_requests import ScriptRequests
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.local_script_runner import LocalScriptRunner

from vla_visualizer.config import ROOT
from vla_visualizer.execution.jobs import JobManager
from vla_visualizer.inference.runner import InferenceRunner


@pytest.fixture
def fragment_driver(monkeypatch):
    storage = MemoryFragmentStorage()
    driver = {"fragment": None, "runner": None}

    class TimerRunner(LocalScriptRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._fragment_storage = storage
            self._requests = ScriptRequests()
            driver["runner"] = self

        def request_rerun(self, data):
            if driver["fragment"]:
                data = replace(data, fragment_id_queue=[driver["fragment"]], is_auto_rerun=True)
            return super().request_rerun(data)

    monkeypatch.setattr("streamlit.testing.v1.app_test.LocalScriptRunner", TimerRunner)
    return driver


@pytest.mark.parametrize("button", ["saved_evidence", "saved_full_loop"])
@pytest.mark.parametrize("view", ["Information flow", "Recorded invocation order"])
def test_timer_continues_past_wrist_despite_delayed_widget_echo(
    fragment_driver, monkeypatch, button, view
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Playback attempted a model load, inference, or job submission")

    monkeypatch.setattr(JobManager, "submit", forbidden)
    monkeypatch.setattr(InferenceRunner, "load", forbidden)
    monkeypatch.setattr(InferenceRunner, "run_inference", forbidden)
    app = AppTest.from_file(ROOT / "app.py", default_timeout=40).run()
    app.button(key=button).click().run()
    snapshot = app.session_state["education_run"].snapshot
    app.slider(key="_edu_action").set_value(49).run()
    if view != "Information flow":
        app.radio(key="_edu_view").set_value(view).run()
    app.button(key="edu_play").click().run()
    fragment_driver["fragment"] = next(
        msg.auto_rerun.fragment_id
        for msg in fragment_driver["runner"].forward_msgs()
        if msg.WhichOneof("type") == "auto_rerun" and msg.auto_rerun.interval == 1.5
    )

    def tick():
        app.session_state["edu_tick"] = monotonic() - 2
        app.run()
        assert not app.exception
        assert app.session_state["edu_playing"]
        assert app.session_state["edu_action"] == 49

    tick()
    assert app.session_state["edu_stage"] == "top_connector"
    tick()
    assert app.session_state["edu_stage"] == "wrist_vision"
    delayed = app._tree.get_widget_states()
    tick()
    assert app.session_state["edu_stage"] == "wrist_connector"
    # The browser reports the older displayed dropdown/slider after the timer
    # advanced. Previously this called commit_selection and stopped playback.
    app._run(delayed)
    assert not app.exception
    assert app.session_state["edu_playing"]
    assert app.session_state["edu_stage"] == "wrist_connector"
    assert app.button(key="edu_play").disabled
    assert app.selectbox(key="_edu_stage").disabled
    tick()
    assert app.session_state["edu_stage"] == (
        "language_tokens" if view == "Information flow" else "language_embedding"
    )
    assert app.session_state["education_run"].snapshot == snapshot
    app.button(key="edu_pause").click().run()
    assert not app.session_state["edu_playing"]
    assert app.session_state["edu_pause_reason"] == "Pause button"
    if view == "Information flow":
        app.selectbox(key="_edu_stage").select("wrist_vision").run()
        assert app.session_state["edu_stage"] == "wrist_vision"


def test_delayed_flow_widget_cannot_stop_an_active_iteration():
    from vla_visualizer.ui.education_state import commit_widget

    state = {"edu_playing": True, "edu_flow": 1, "_edu_flow": 0}
    assert not commit_widget(state, "edu_flow")
    assert state["edu_flow"] == 1 and state["edu_playing"]
    state["edu_playing"] = False
    assert commit_widget(state, "edu_flow")
    assert state["edu_flow"] == 0
