"""Verify input validation and lifecycle failures without loading model weights."""

from pathlib import Path
from dataclasses import replace

from PIL import Image
import pytest

from vla_visualizer.adapters import registry
from vla_visualizer.adapters.smolvla import SmolVLAAdapter
from vla_visualizer.contracts import LoadOptions, Observation
from vla_visualizer.inference.observations import observation_fingerprint, read_observation
from vla_visualizer.inference.runner import InferenceRunner
from vla_visualizer.ui.inputs import parse_state

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("state", ["", "1,2", "1,2,3,4,5,nan", "1,2,3,4,5,inf"])
def test_reject_missing_or_nonfinite_state(state: str) -> None:
    with pytest.raises(ValueError):
        parse_state(state, 6)


def test_reject_invalid_camera_and_missing_state() -> None:
    adapter = SmolVLAAdapter()
    with pytest.raises(ValueError, match="six measured"):
        adapter.validate_observation(Observation({}, "Move the cube"))
    with pytest.raises(ValueError, match="supported key"):
        adapter.validate_observation(
            Observation(
                {"unsupported_camera": Image.new("RGB", (2, 2))},
                "Move the cube",
                (1.0,) * 6,
            )
        )


def test_observation_fingerprint_tracks_state_and_instruction() -> None:
    sample = read_observation(ROOT / "assets/sample_inputs/so100_observation.json")
    changed = Observation(sample.images, sample.instruction + " now", sample.state)
    assert observation_fingerprint(sample) != observation_fingerprint(changed)
    changed_state = Observation(
        sample.images, sample.instruction, tuple(x + 1 for x in sample.state)
    )
    assert observation_fingerprint(sample) != observation_fingerprint(changed_state)


def test_real_curated_sample_requires_two_views_and_pinned_domain() -> None:
    sample = read_observation(ROOT / "assets/sample_inputs/so100_observation.json")
    adapter = SmolVLAAdapter()
    adapter.validate_observation(sample)
    assert set(sample.images) == {"observation.images.top", "observation.images.wrist"}
    assert all(image.size == (640, 480) for image in sample.images.values())
    with pytest.raises(ValueError, match="both required"):
        adapter.validate_observation(replace(sample, images={"observation.images.top": next(
            iter(sample.images.values())
        )}))
    with pytest.raises(ValueError, match="dataset/revision"):
        adapter.validate_observation(replace(sample, provenance={**sample.provenance, "revision": "main"}))
    with pytest.raises(ValueError, match="state ordering"):
        adapter.validate_observation(replace(sample, provenance={
            **sample.provenance, "state_names": list(reversed(sample.provenance["state_names"])),
        }))


def test_failed_load_unloads_previous_and_failed_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []

    class LifecycleProbe:
        def __init__(self, fail: bool) -> None:
            self.fail = fail

        def load(self, options: LoadOptions) -> None:
            events.append("load_fail" if self.fail else "load_ok")
            if self.fail:
                raise RuntimeError("allocation failed")

        def unload(self) -> None:
            events.append("unload_fail" if self.fail else "unload_ok")

    probes = iter([LifecycleProbe(False), LifecycleProbe(True)])
    monkeypatch.setattr(registry, "create_adapter", lambda model_id: next(probes))
    runner = InferenceRunner()
    runner.load("smolvla", LoadOptions())
    assert runner.is_loaded("smolvla")
    with pytest.raises(RuntimeError, match="allocation failed"):
        runner.load("smolvla", LoadOptions())
    assert not runner.is_loaded("smolvla")
    assert events == ["load_ok", "unload_ok", "load_fail", "unload_fail"]


def test_snapshot_checks_incomplete_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *args, **kwargs: str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Incomplete cached snapshot"):
        SmolVLAAdapter._snapshot("repo", "revision", ("model.safetensors",), LoadOptions())
