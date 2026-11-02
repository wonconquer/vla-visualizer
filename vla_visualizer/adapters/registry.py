"""Small model catalog that does not import or instantiate VLA implementations."""

from vla_visualizer.contracts import ModelSpec
from vla_visualizer.adapters.base import BaseVLAAdapter
from vla_visualizer.adapters.smolvla_config import MVP_CHECKPOINT

_MODELS = (
    ModelSpec(
        model_id="smolvla",
        display_name="SmolVLA",
        checkpoint_id=MVP_CHECKPOINT.repo_id,
        status="available",
        download_size_mb=907,
    ),
)


def list_models(*, available_only: bool = False) -> tuple[ModelSpec, ...]:
    """List planned/available models, or only fully integrated backends."""
    if available_only:
        return tuple(model for model in _MODELS if model.status == "available")
    return _MODELS


def get_model(model_id: str) -> ModelSpec:
    """Look up an exact catalog ID; never substitute a different model."""
    for model in _MODELS:
        if model.model_id == model_id:
            return model
    raise KeyError(f"Unknown model ID: {model_id}")


def create_adapter(model_id: str) -> BaseVLAAdapter:
    """Create a lightweight adapter without loading weights."""
    get_model(model_id)
    if model_id == "smolvla":
        from vla_visualizer.adapters.smolvla import SmolVLAAdapter

        return SmolVLAAdapter()
    raise KeyError(f"No adapter registered for {model_id}")
