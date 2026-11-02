"""Regression checks: the UI stays usable without loading a model or CUDA."""

from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest
import torch

from vla_visualizer.adapters.base import BaseVLAAdapter
from vla_visualizer.adapters.registry import get_model, list_models
from vla_visualizer.contracts import Observation
from vla_visualizer.inference.runner import InferenceRunner, ModelUnavailableError
from vla_visualizer.ui import inputs
from vla_visualizer.utils import images
from vla_visualizer.utils.device import detect_device


def test_base_adapter_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseVLAAdapter()


def test_catalog_distinguishes_planned_from_available() -> None:
    models = list_models()
    assert [model.model_id for model in models] == ["smolvla"]
    assert models[0].checkpoint_id == "Zekai-Chen/smolvla_so100_pickplace_ft_2cam"
    assert models[0].status == "available"
    assert list_models(available_only=True) == models
    with pytest.raises(KeyError, match="Unknown model ID"):
        get_model("not-a-model")


def test_backend_never_returns_fake_inference() -> None:
    observation = Observation(images={}, instruction="Move the block.")
    with pytest.raises(ModelUnavailableError, match="Load the model"):
        InferenceRunner().run_inference("smolvla", observation)


def test_device_detection_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def unexpected_gpu_query(index: int) -> str:
        raise AssertionError("An unavailable GPU must not be queried")

    monkeypatch.setattr(torch.cuda, "get_device_name", unexpected_gpu_query)
    info = detect_device()
    assert info.device == "cpu"
    assert info.cuda_available is False
    assert info.cuda_device_name is None


def test_device_detection_with_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Test GPU")
    info = detect_device()
    assert info.device == "cuda"
    assert info.cuda_available is True
    assert info.cuda_device_name == "Test GPU"


@pytest.mark.parametrize("fail_at", ["is_available", "get_device_name"])
def test_device_detection_failure_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch, fail_at: str
) -> None:
    def fail(*args: object) -> None:
        raise RuntimeError("Driver unavailable")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, fail_at, fail)
    info = detect_device()
    assert info.device == "cpu"
    assert info.cuda_available is False
    assert "Driver unavailable" in (info.detail or "")


def test_image_preview_decodes_real_image_bytes() -> None:
    contents = BytesIO()
    Image.new("RGBA", (3, 2), color=(255, 0, 0, 255)).save(contents, format="PNG")
    preview = inputs.decode_image(contents.getvalue())
    assert preview.mode == "RGB"
    assert preview.size == (3, 2)
    assert preview.getpixel((0, 0)) == (255, 0, 0)


def test_image_preview_rejects_invalid_bytes() -> None:
    with pytest.raises(ValueError, match="valid image"):
        inputs.decode_image(b"not an image")


def test_image_preview_rejects_excessive_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(images, "MAX_UPLOAD_BYTES", 4)
    with pytest.raises(ValueError, match="preview limit"):
        inputs.decode_image(b"12345")


def test_image_preview_rejects_excessive_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(images, "MAX_IMAGE_PIXELS", 4)
    contents = BytesIO()
    Image.new("RGB", (3, 2)).save(contents, format="PNG")
    with pytest.raises(ValueError, match="megapixel"):
        inputs.decode_image(contents.getvalue())


def test_streamlit_shell_renders_and_accepts_instruction() -> None:
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(root / "app_baseline.py", default_timeout=30).run()
    assert not app.exception
    assert app.selectbox(key="model_id").value == "smolvla"
    assert len(app.get("file_uploader")) == 2
    assert len(app.get("graphviz_chart")) == 1
    assert len(app.button) == 4  # Original lifecycle controls + model-free saved capture.
    assert app.button[0].disabled is False
    assert app.button[1].disabled and app.button[2].disabled
    assert any("No predicted actions" in element.value for element in app.info)
    assert any("data flow" in element.value.lower() for element in app.subheader)

    app.text_area(key="instruction").set_value("Pick up the block.").run()
    assert not app.exception
    assert app.text_area(key="instruction").value == "Pick up the block."
    assert app.button[1].disabled and app.button[2].disabled
