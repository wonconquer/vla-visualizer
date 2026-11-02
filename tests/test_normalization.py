"""Checkpoint audit and native processor math; no pretrained inference is fabricated."""

import json
import base64
from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file
from streamlit.testing.v1 import AppTest
import torch

from vla_visualizer.adapters.smolvla import (
    NormalizationCompatibilityError,
    SmolVLAAdapter,
    inspect_checkpoint_normalization,
    require_verified_normalization,
)
from vla_visualizer.contracts import LoadOptions
from vla_visualizer.inference.observations import read_observation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkpoint_metadata(tmp_path: Path) -> Path:
    """Reproduce the tiny native artifacts from the captured audit, outside the Hub cache."""
    recorded = json.loads((ROOT / "tests/fixtures/base_normalization.json").read_text())
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "normalization_mapping": recorded["normalization_method"],
                "input_features": recorded["model_input_features"],
                "output_features": recorded["model_output_features"],
            }
        )
    )
    for role, data in recorded["processors"].items():
        (tmp_path / data["config_file"]).write_text(
            json.dumps(
                {
                    "steps": [
                        {
                            "registry_name": "normalizer_processor"
                            if role == "preprocessor"
                            else "unnormalizer_processor",
                            "config": {"features": data["features"], "norm_map": data["norm_map"]},
                            "state_file": data["state_file"],
                        }
                    ]
                }
            )
        )
        # These values are original published statistics, not fabricated substitutes.
        save_file(
            {
                key: torch.tensor(item["values"], dtype=torch.float32)
                for key, item in data["available_tensors"].items()
            },
            tmp_path / data["state_file"],
        )
    return tmp_path


@pytest.fixture
def approved_metadata(tmp_path: Path) -> Path:
    """Exact tiny published artifacts, independent of downloads and the model cache."""
    record = json.loads((ROOT / "tests/fixtures/smolvla_native_artifacts.json").read_text())
    for filename, data in record["files_base64"].items():
        (tmp_path / filename).write_bytes(base64.b64decode(data))
    return tmp_path


def test_approved_native_artifacts_pass_guard_with_constant_metadata(
    approved_metadata: Path,
) -> None:
    report = inspect_checkpoint_normalization(approved_metadata)
    require_verified_normalization(report)
    assert report["verified"] is True
    for item in report["processors"].values():
        assert item["available_tensors"]["task_index.std"]["values"] == [0.0]
        assert not item["missing_keys"] and not item["invalid_tensors"]
    assert report["provenance"]["dataset_revision"] == (
        "87169e502b742f989b96c952bff7af55e248d467"
    )


@pytest.mark.parametrize("key", ["observation.state.std", "action.std"])
def test_canonical_active_zero_std_is_rejected(approved_metadata: Path, key: str) -> None:
    path = approved_metadata / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    tensors = load_file(path)
    tensors[key][0] = 0
    save_file(tensors, path)
    report = inspect_checkpoint_normalization(approved_metadata)
    assert key in report["processors"]["preprocessor"]["invalid_tensors"]
    with pytest.raises(NormalizationCompatibilityError):
        require_verified_normalization(report)


def test_same_shapes_cannot_approve_changed_processor(approved_metadata: Path) -> None:
    path = approved_metadata / "policy_preprocessor.json"
    processor = json.loads(path.read_text())
    processor["steps"][-1]["config"]["norm_map"]["STATE"] = "IDENTITY"
    path.write_text(json.dumps(processor))
    report = inspect_checkpoint_normalization(approved_metadata)
    assert not report["processors"]["preprocessor"]["missing_keys"]
    assert "policy_preprocessor.json" in report["artifact_hash_mismatches"]
    with pytest.raises(NormalizationCompatibilityError):
        require_verified_normalization(report)


def test_exact_missing_keys_and_multiple_profiles(checkpoint_metadata: Path) -> None:
    report = inspect_checkpoint_normalization(checkpoint_metadata)
    assert report["processors"]["preprocessor"]["missing_keys"] == [
        "observation.state.mean",
        "observation.state.std",
        "action.mean",
        "action.std",
    ]
    assert report["processors"]["postprocessor"]["missing_keys"] == ["action.mean", "action.std"]
    for item in report["processors"].values():
        assert item["dataset_prefixes"] == ["so100", "so100-blue", "so100-red"]
        assert len(item["available_tensors"]) == 6
        assert not item["invalid_tensors"]
        for tensor in item["available_tensors"].values():
            assert tensor["shape"] == [6] and tensor["finite"]
    assert report["verified"] is False


def test_no_silent_native_processor_skip_is_accepted(checkpoint_metadata: Path) -> None:
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
    from lerobot.processor.converters import (
        batch_to_transition,
        transition_to_batch,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    report = inspect_checkpoint_normalization(checkpoint_metadata)
    tensors = load_file(checkpoint_metadata / report["processors"]["preprocessor"]["state_file"])
    features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
        "action": PolicyFeature(FeatureType.ACTION, (6,)),
    }
    modes = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }
    pre = NormalizerProcessorStep(features=features, norm_map=modes)
    post = UnnormalizerProcessorStep(features=features, norm_map=modes)
    pre.load_state_dict(tensors)
    post.load_state_dict(tensors)
    state = torch.tensor(
        read_observation(ROOT / "assets/sample_inputs/so100_observation.json").state
    )
    native_state = transition_to_batch(pre(batch_to_transition({"observation.state": state})))
    torch.testing.assert_close(native_state["observation.state"], state, rtol=0, atol=0)
    numeric_probe = torch.linspace(-1, 1, 300).reshape(1, 50, 6)
    native_actions = transition_to_policy_action(post(policy_action_to_transition(numeric_probe)))
    torch.testing.assert_close(native_actions, numeric_probe, rtol=0, atol=0)
    # Native silent identity is reproduced, but it can never enter our inference path.
    with pytest.raises(NormalizationCompatibilityError, match="No dataset-prefix"):
        require_verified_normalization(report)


def test_load_stops_before_weights_or_backbone(
    checkpoint_metadata: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def local_metadata(repo, revision, files, options):
        calls.append((repo, files))
        assert "model.safetensors" not in files
        return str(checkpoint_metadata)

    monkeypatch.setattr(SmolVLAAdapter, "_snapshot", staticmethod(local_metadata))
    adapter = SmolVLAAdapter()
    with pytest.raises(NormalizationCompatibilityError, match="observation.state.mean"):
        adapter.load(LoadOptions())
    assert len(calls) == 1
    assert adapter._policy is None
    assert adapter._preprocessor is None and adapter._postprocessor is None


@pytest.mark.parametrize("corruption", ["nonfinite", "zero_std", "wrong_shape"])
def test_reject_invalid_statistic_tensors(checkpoint_metadata: Path, corruption: str) -> None:
    path = checkpoint_metadata / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    tensors = load_file(path)
    key = "so100.buffer.action.std"
    if corruption == "nonfinite":
        tensors[key][0] = float("nan")
    elif corruption == "zero_std":
        tensors[key][0] = 0
    else:
        tensors[key] = tensors[key][:5]
    save_file(tensors, path)  # Only the temporary fixture, never a cached checkpoint.
    report = inspect_checkpoint_normalization(checkpoint_metadata)
    assert key in report["processors"]["postprocessor"]["invalid_tensors"]
    with pytest.raises(NormalizationCompatibilityError, match="invalid="):
        require_verified_normalization(report)


def test_candidate_native_state_and_action_math() -> None:
    """Validate approved native statistics numerically without model construction."""
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
    from lerobot.processor.converters import (
        batch_to_transition,
        transition_to_batch,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    record = json.loads((ROOT / "tests/fixtures/native_statistics.json").read_text())
    stats = record["normalization"]["pre"]["tensors"]
    tensors = {key: torch.tensor(item["values"]) for key, item in stats.items()}
    features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
        "action": PolicyFeature(FeatureType.ACTION, (6,)),
    }
    modes = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }
    pre = NormalizerProcessorStep(features=features, norm_map=modes)
    post = UnnormalizerProcessorStep(features=features, norm_map=modes)
    pre.load_state_dict(tensors)
    post_tensors = {
        key: torch.tensor(item["values"])
        for key, item in record["normalization"]["post"]["tensors"].items()
    }
    post.load_state_dict(post_tensors)
    state = torch.tensor(
        read_observation(ROOT / "assets/sample_inputs/so100_observation.json").state
    ).unsqueeze(0)
    normalized = transition_to_batch(pre(batch_to_transition({"observation.state": state})))
    expected = (state - tensors["observation.state.mean"]) / (
        tensors["observation.state.std"] + 1e-8
    )
    torch.testing.assert_close(normalized["observation.state"], expected)
    assert not torch.equal(normalized["observation.state"], state)
    restored = transition_to_batch(post(batch_to_transition(normalized)))["observation.state"]
    torch.testing.assert_close(restored, state)
    normalized_actions = torch.linspace(-1, 1, 300).reshape(1, 50, 6)
    physical = transition_to_policy_action(post(policy_action_to_transition(normalized_actions)))
    expected_actions = normalized_actions * post_tensors["action.std"] + post_tensors["action.mean"]
    torch.testing.assert_close(physical, expected_actions)
    roundtrip = transition_to_policy_action(pre(policy_action_to_transition(physical)))
    torch.testing.assert_close(roundtrip, normalized_actions)
    assert physical.shape == (1, 50, 6) and physical.isfinite().all()
    assert normalized["observation.state"].shape == (1, 6) and expected.isfinite().all()


def test_streamlit_reports_blocker_without_actions(
    checkpoint_metadata: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        SmolVLAAdapter, "_snapshot", staticmethod(lambda *args: str(checkpoint_metadata))
    )
    app = AppTest.from_file(ROOT / "app_baseline.py", default_timeout=30).run()
    app.checkbox(key="use_sample").check().run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Normalization blocked" in error.value for error in app.error)
    assert app.button[2].disabled
    assert "last_run" not in app.session_state
    assert not app.dataframe
