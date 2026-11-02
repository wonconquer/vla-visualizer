"""Real artifact mapping and model-free Streamlit replay regression tests."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
import torch

from vla_visualizer.adapters.smolvla import SmolVLAAdapter
from vla_visualizer.inference.replay import (
    ReplayError,
    action_rows,
    language_rows,
    load_run,
    make_run,
    norm_rows,
    prefix_rows,
    state_rows,
)
from vla_visualizer.inference.runner import InferenceRunner
from vla_visualizer.ui.inspection import autoplay_tick
from vla_visualizer.visualization.inspection import inspection_dot

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def captured():
    return json.loads((ROOT / "docs/phase3_integration_validation.json").read_text())


@pytest.fixture
def spec():
    return SmolVLAAdapter().get_inspection_spec()


def test_real_capture_mapping_and_plotted_values(captured, spec):
    run = make_run(captured, spec)
    data = run.read()
    assert len(data["manifest"]["records"]) == 23
    assert data["capture"]["actions"] == captured["actions"]
    cameras = {r["camera"] for r in data["manifest"]["records"] if r["camera"]}
    assert cameras == {"observation.images.top", "observation.images.wrist"}
    assert {r["flow_step"] for r in data["manifest"]["records"] if r["flow_step"] is not None} == {
        0,
        4,
        9,
    }
    assert [row["positions"] for row in prefix_rows(captured, spec)] == [64, 64, 48, 1]
    assert [row["start inclusive"] for row in prefix_rows(captured, spec)] == [0, 64, 128, 176]
    tokens = language_rows(captured)
    assert len(tokens) == 12 and all(row["valid"] for row in tokens)
    all_tokens = language_rows(captured, reveal_padding=True)
    assert len(all_tokens) == 48 and sum(row["padding"] for row in all_tokens) == 36
    assert all_tokens[-1]["token_id"] == 2 and all_tokens[-1]["token"] == "<|im_end|>"
    rows = state_rows(captured, spec)
    assert [row["joint"] for row in rows] == spec["joint_names"]
    assert rows[0]["raw measured"] == 0.3515625
    plotted = action_rows(captured)
    for index, action in enumerate(captured["actions"]):
        assert [plotted[index][name] for name in spec["joint_names"]] == action
    vision = next(r for r in captured["intermediates"]["stages"] if r["stage_name"] == "vision")
    assert [r["L2 norm"] for r in norm_rows(vision["outputs"]["tensor"])] == vision["outputs"][
        "tensor"
    ]["token_norms"]
    before = run.snapshot
    data["capture"]["actions"][0][0] = 999
    assert run.snapshot == before and run.read()["capture"]["actions"][0][0] != 999
    assert "cached prefix context" in inspection_dot(spec, "prefix")


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "shape",
        "norm_length",
        "mask",
        "module",
        "seed",
        "record_mapping",
        "nan_action",
        "statistic",
        "camera_invocation",
        "instruction",
    ],
)
def test_invalid_data_is_rejected(captured, spec, mutation):
    data = make_run(captured, spec).read()
    capture = data["capture"]
    if mutation == "schema":
        data["manifest"]["schema_version"] = 99
    elif mutation == "shape":
        capture["intermediates"]["stages"][0]["outputs"]["normalized_state"]["shape"] = [1, 7]
    elif mutation == "norm_length":
        capture["intermediates"]["stages"][5]["outputs"]["tensor"]["token_norms"].pop()
    elif mutation == "mask":
        capture["intermediates"]["stages"][3]["outputs"]["attention_mask"]["values"][0] = 0
    elif mutation == "module":
        capture["intermediates"]["stages"][5]["module_name"] = "unverified.module"
    elif mutation == "seed":
        data["manifest"]["seed"] = 42
    elif mutation == "record_mapping":
        data["manifest"]["records"][5]["camera"] = "observation.images.wrist"
    elif mutation == "nan_action":
        capture["actions"][0][0] = float("nan")
    elif mutation == "statistic":
        capture["intermediates"]["stages"][0]["outputs"]["normalized_state"]["std"] = -1
    elif mutation == "camera_invocation":
        record = capture["intermediates"]["stages"][5]
        record["context"]["call_index"] = 1
    elif mutation == "instruction":
        data["manifest"]["instruction"] = "Different unverified instruction"
    with pytest.raises(ReplayError):
        load_run(json.dumps(data))


def test_optional_stages_and_values_remain_unavailable(captured, spec):
    capture = deepcopy(captured)
    capture["intermediates"]["stages"] = [
        r
        for r in capture["intermediates"]["stages"]
        if r["stage_name"] not in {"language", "connector", "vision"}
    ]
    capture["intermediates"]["language"] = []
    run = make_run(capture, spec)
    assert prefix_rows(run.read()["capture"], spec) == []
    assert language_rows(run.read()["capture"]) == []
    assert norm_rows({}) == []
    assert not run.read()["images"]


def test_autoplay_is_single_bounded_explanatory_step():
    assert autoplay_tick(10, 0, 2, 7, False) == (2, False, 0)
    assert autoplay_tick(1, 0, 2, 7, True) == (2, True, 0)
    assert autoplay_tick(100, 0, 2, 7, True) == (3, True, 100)  # no catch-up loop
    assert autoplay_tick(100, 0, 5, 7, True) == (6, False, 100)


def test_saved_replay_controls_never_execute_inference_or_register_hooks(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Replay attempted model execution/hook registration")

    monkeypatch.setattr(InferenceRunner, "load", forbidden)
    monkeypatch.setattr(InferenceRunner, "run_inference", forbidden)
    monkeypatch.setattr(torch.nn.Module, "register_forward_hook", forbidden)
    app = AppTest.from_file(ROOT / "app_baseline.py", default_timeout=30).run()
    app.button(key="load_saved_capture").click().run()
    assert not app.exception and not app.error
    run = app.session_state["replay_run"]
    assert len(run.read()["images"]) == 2
    identity, snapshot = run.run_id, run.snapshot
    for stage in ("camera_0", "camera_1", "language", "state", "prefix", "generation", "actions"):
        app.selectbox(key="replay_stage").set_value(stage).run()
        assert not app.exception and not app.error
    app.selectbox(key="replay_stage").set_value("language").run()
    assert any(frame.value.shape[0] == 12 for frame in app.dataframe)
    app.checkbox(key="replay_padding").check().run()
    assert any(frame.value.shape[0] == 48 for frame in app.dataframe)
    for step in (4, 9, 0):
        app.selectbox(key="replay_flow").set_value(step).run()
    app.slider(key="replay_action").set_value(17).run()
    app.button(key="replay_reset").click().run()
    assert app.selectbox(key="replay_stage").value == "camera_0"
    assert (
        app.selectbox(key="replay_flow").value == 0 and app.slider(key="replay_action").value == 17
    )
    app.button(key="replay_next").click().run()
    assert app.selectbox(key="replay_stage").value == "camera_1"
    app.button(key="replay_previous").click().run()
    app.button(key="replay_play").click().run()
    assert app.button(key="replay_play").label == "Pause"
    app.button(key="replay_play").click().run()
    assert app.button(key="replay_play").label == "Play"
    assert not app.session_state["replay_timer_enabled"]
    # Changing pending input never relabels a completed capture.
    app.text_area(key="instruction").set_value("Pending instruction only").run()
    for _ in range(4):
        app.run()
    assert app.session_state["replay_run"].run_id == identity
    assert app.session_state["replay_run"].snapshot == snapshot
    assert "last_run" not in app.session_state
    assert any("Pending inputs differ" in warning.value for warning in app.warning)
    assert not app.exception


def test_failed_saved_capture_preserves_last_valid_result(monkeypatch):
    app = AppTest.from_file(ROOT / "app_baseline.py", default_timeout=30).run()
    app.button(key="load_saved_capture").click().run()
    snapshot = app.session_state["replay_run"].snapshot
    original = Path.read_text

    def reject(path, *args, **kwargs):
        if path.name == "phase3_integration_validation.json":
            return '{"invalid":true}'
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject)
    app.button(key="load_saved_capture").click().run()
    assert not app.exception and app.error
    assert app.session_state["replay_run"].snapshot == snapshot
