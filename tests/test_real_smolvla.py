"""Opt-in actual Streamlit→adapter→pretrained-model→action smoke test."""

import os
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
import torch


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("ECE525_REAL_SMOKE") != "1",
    reason="Set ECE525_REAL_SMOKE=1; needs cached weights",
)
def test_real_inference_from_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    # Everything below uses native processors, real recorded inputs, and actual weights.
    from vla_visualizer.adapters.smolvla import (
        CHECKPOINT,
        CHECKPOINT_FILES,
        REVISION,
        SmolVLAAdapter,
        inspect_checkpoint_normalization,
        require_verified_normalization,
    )
    from vla_visualizer.contracts import CaptureOptions, LoadOptions
    from vla_visualizer.adapters import registry
    from vla_visualizer.inference.observations import read_observation

    adapters = []
    original_factory = registry.create_adapter

    def track_real_adapter(model_id):
        adapter = original_factory(model_id)
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(registry, "create_adapter", track_real_adapter)

    path = SmolVLAAdapter._snapshot(
        CHECKPOINT,
        REVISION,
        tuple(name for name in CHECKPOINT_FILES if name != "model.safetensors"),
        LoadOptions(),
    )
    require_verified_normalization(inspect_checkpoint_normalization(Path(path)))
    app = AppTest.from_file(root / "app_baseline.py", default_timeout=600).run()
    try:
        app.checkbox(key="use_sample").check().run()
        app.button[0].click().run()  # Explicit local-only load.
        assert not app.exception and not app.error
        adapter = next(item for item in adapters if item._policy is not None)
        assert adapter._loading_keys == {"tensor_count": 500, "missing": [], "unexpected": []}
        assert not adapter._policy.training
        assert all(not p.requires_grad for p in adapter._policy.parameters())
        observation = read_observation(root / "assets/sample_inputs/so100_observation.json")
        prepared = adapter.preprocess(observation)
        assert prepared["observation.state"].shape == (1, 6)
        assert prepared["task"] == [observation.instruction + "\n"]
        assert prepared["observation.language.tokens"].shape == (1, 48)
        assert prepared["observation.language.attention_mask"].shape == (1, 48)
        assert all(prepared[key].shape == (1, 3, 480, 640) for key in [
            "observation.images.camera1", "observation.images.camera2",
        ])
        with torch.inference_mode():
            images, masks = adapter._policy.prepare_images(prepared)
            assert len(images) == len(masks) == 2
            assert all(image.shape == (1, 3, 512, 512) and image.isfinite().all()
                       and image.min() >= -1 and image.max() <= 1 for image in images)
            assert adapter._policy.prepare_state(prepared).shape == (1, 32)
        calls = {"reset": 0, "predict": 0}
        original_reset = adapter._policy.reset
        original_predict = adapter._policy.predict_action_chunk

        def reset():
            calls["reset"] += 1
            original_reset()

        def predict(*args, **kwargs):
            assert torch.is_inference_mode_enabled()
            calls["predict"] += 1
            return original_predict(*args, **kwargs)

        monkeypatch.setattr(adapter._policy, "reset", reset)
        monkeypatch.setattr(adapter._policy, "predict_action_chunk", predict)
        runner = app.session_state.filtered_state  # UI must not save a model in per-session state.
        assert all(not isinstance(value, torch.nn.Module) for value in runner.values())
        app.button[2].click().run()
        assert not app.exception and not app.error
        result = app.session_state["last_run"][2]
        assert result.metadata["parameter_count"] == 450046176
        assert result.metadata["native_action_shape"] == [1, 50, 6]
        assert result.metadata["frozen"] is True
        assert result.metadata["eval_mode"] is True
        assert result.metadata["state_normalization_verified"] is True
        assert result.metadata["action_postprocessing_verified"] is True
        assert result.metadata["strict_loading"] is True
        assert result.metadata["action_chunk_shape"] == [1, 50, 6]
        assert result.metadata["action_output_device"] == "cpu"
        assert result.metadata["action_output_dtype"] == "torch.float32"
        # The actual predicted normalized values must be inverse-normalized with
        # the published action statistics, not merely return a finite tensor.
        stats = result.metadata["normalization_audit"]["processors"]["postprocessor"]["available_tensors"]
        expected = torch.tensor(result.metadata["normalized_first_action"]) * torch.tensor(
            stats["action.std"]["values"]
        ) + torch.tensor(stats["action.mean"]["values"])
        torch.testing.assert_close(torch.tensor(result.actions[0]), expected)
        assert torch.isfinite(torch.tensor(result.actions)).all()
        assert len(app.dataframe) == 1
        assert app.dataframe[0].value.shape == (50, 7)  # Step + six action dimensions.
        assert "Real inference completed" in app.success[0].value
        first = result.actions
        app.button[2].click().run()
        assert not app.exception
        assert app.session_state["last_run"][2].actions == first
        assert calls == {"reset": 2, "predict": 2}
        # A new seed changes actual flow-matching noise. This extra native call
        # makes cached/stale output reuse distinguishable from reproducibility.
        adapter._options = replace(adapter._options, seed=1)
        changed = adapter.run_inference(adapter.preprocess(observation), CaptureOptions())
        assert changed.actions != first
        assert calls == {"reset": 3, "predict": 3}
        report = asdict(result)
        report["validation"] = {
            "strict_checkpoint_loading": "PASS", "real_sample_preprocessing": "PASS",
            "actual_pretrained_inference": "PASS", "streamlit_result_display": "PASS",
            "same_seed_repeat_matches": True, "different_seed_output_changes": True,
            "independent_native_reset_and_predict_calls": calls,
            "native_image_shapes": [list(image.shape) for image in images],
            "native_padded_state_shape": [1, 32],
        }
        validation_dir = root / os.environ.get("ECE525_VALIDATION_DIR", "outputs")
        validation_dir.mkdir(parents=True, exist_ok=True)
        (validation_dir / "inference_validation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        # Editing inputs clears previous actions; nothing is shown as a fresh prediction.
        app.checkbox(key="use_sample").uncheck().run()
        assert "last_run" not in app.session_state
        assert not app.dataframe
    finally:
        app.button[1].click().run()  # Release the cached process-wide model.
    assert not app.exception
    assert app.button[2].disabled
