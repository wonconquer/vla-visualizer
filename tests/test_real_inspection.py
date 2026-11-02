"""Opt-in actual pretrained capture, semantic checks and same-seed regression."""

import json
import os
from pathlib import Path

import pytest
import torch

from examples.smolvla_inspection import inspect_real_sample
from vla_visualizer.contracts import LoadOptions
from vla_visualizer.adapters.smolvla_inspection import POINTS


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("ECE525_REAL_SMOKE") != "1", reason="Needs cached checkpoint; opt in")
def test_real_inspection_preserves_actions_and_captures_executed_stages() -> None:
    report = inspect_real_sample(LoadOptions(), repeats=3)
    intermediate = report["intermediates"]
    stages = intermediate["stages"]
    def named(name):
        return [stage for stage in stages if stage["stage_name"] == name]
    assert len(stages) == 23
    assert set(intermediate["executed_hook_counts"]) == set(POINTS.values())
    for camera in ("observation.images.top", "observation.images.wrist"):
        layout = intermediate["vision"][camera]
        assert layout["patch_grid"] == (32, 32)
        assert layout["token_count"] == 1024 and layout["embedding_dim"] == 768
        assert layout["connector_grid"] == (8, 8)
        assert layout["connector_token_count"] == 64 and layout["connector_embedding_dim"] == 960
        assert layout["patch_group_size"] == (4, 4)
        assert not layout["vision_attentions_returned"]
        assert layout["resize_geometry"]["resized_hw"] == (384, 512)
        assert layout["resize_geometry"]["padding_top_left"] == (128, 0)
    for stage in named("vision"):
        assert stage["inputs"]["tensor"]["shape"] == (1, 3, 512, 512)
        assert stage["outputs"]["tensor"]["shape"] == (1, 1024, 768)
        assert len(stage["outputs"]["tensor"]["token_norms"]) == 1024
        assert stage["inputs"]["tensor"]["minimum"] >= -1
        assert stage["inputs"]["tensor"]["maximum"] <= 1
    tokenized = named("language_tokenization")[0]
    assert tokenized["outputs"]["token_ids"]["shape"] == (1, 48)
    assert tokenized["outputs"]["attention_mask"]["shape"] == (1, 48)
    assert tokenized["context"]["valid_token_count"] == 12
    assert len(intermediate["language"][0]["tokens"]) == 48
    assert named("language")[0]["outputs"]["tensor"]["shape"] == (1, 48, 960)
    state = named("robot_state_normalization")[0]
    padded = named("state")[0]["inputs"]["tensor"]["values"]
    normalized = state["outputs"]["normalized_state"]["values"]
    assert padded[:6] == normalized and padded[6:] == (0,) * 26
    assert named("state")[0]["outputs"]["tensor"]["shape"] == (1, 960)
    stats = report["metadata"]["normalization_audit"]["processors"]["preprocessor"]["available_tensors"]
    expected = (torch.tensor(state["inputs"]["raw_state"]["values"]) - torch.tensor(
        stats["observation.state.mean"]["values"])) / (torch.tensor(
        stats["observation.state.std"]["values"]) + 1e-8)
    torch.testing.assert_close(torch.tensor(normalized), expected)
    assert named("multimodal_prefix")[0]["inputs"]["tensor"]["shape"] == (1, 177, 960)
    for stage_name in ("flow_state", "time_conditioned_actions", "action_velocity"):
        selected = named(stage_name)
        assert [stage["context"]["flow_step_index"] for stage in selected] == [0, 4, 9]
        assert intermediate["executed_hook_counts"][POINTS[stage_name]] == 10
    assert named("flow_state")[0]["inputs"]["tensor"]["preview"] == named("initial_noise")[0]["outputs"]["noise"]["preview"]
    for stage in named("action_velocity"):
        assert stage["inputs"]["tensor"]["shape"] == (1, 50, 720)
        assert stage["outputs"]["tensor"]["shape"] == (1, 50, 32)
    assert named("normalized_action_chunk")[0]["outputs"]["normalized_action"]["shape"] == (1, 50, 6)
    assert named("action_postprocessing")[0]["outputs"]["action"]["shape"] == (1, 50, 6)
    assert not intermediate["attention_available"]
    for check in report["validation"]["checks"]:
        assert check["max_action_absolute_difference"] == 0
        assert check["hook_count_before"] == check["hook_count_after"] == 0
        assert check["record_count"] == 23
    for stage in stages:
        for summary in (*stage["inputs"].values(), *stage["outputs"].values()):
            assert summary["device"] == "cpu"
            assert len(summary["preview"]) <= 4 and len(summary["token_norms"]) <= 1024
            if summary["numel"] > 64:
                assert summary["values"] is None
    assert report["metadata"]["action_postprocessing_verified"]
    json.dumps(report, allow_nan=False)
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / "tests/fixtures/reference_actions.json").read_text())
    torch.testing.assert_close(torch.tensor(report["actions"]), torch.tensor(reference["actions"]),
                               rtol=0, atol=1e-6)
    validation_dir = root / os.environ.get("ECE525_VALIDATION_DIR", "outputs")
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "inspection_validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
