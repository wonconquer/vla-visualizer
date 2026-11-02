"""Shared adapter contracts, independent of Streamlit and VLA libraries."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from PIL import Image


@dataclass(frozen=True)
class ModelSpec:
    """Catalog metadata; a planned model is not an integrated backend."""

    model_id: str
    display_name: str
    checkpoint_id: str
    status: Literal["planned", "available"] = "planned"
    download_size_mb: int = 0


@dataclass(frozen=True)
class Observation:
    """One observation; adapters map logical camera names to native input keys."""

    images: Mapping[str, Image.Image]
    instruction: str
    state: tuple[float, ...] | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InputSpec:
    """Verified requirements of a particular model/checkpoint."""

    camera_keys: tuple[str, ...]
    state_dim: int | None
    max_frames_per_camera: int = 1
    normalization_description: str = ""


@dataclass(frozen=True)
class ModelCapabilities:
    """Internal information an adapter has verified it can expose."""

    capture_kinds: tuple[str, ...] = ()
    attention_available: bool = False


@dataclass(frozen=True)
class Stage:
    """One architecture stage with optional verified module paths."""

    stage_id: str
    label: str
    module_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelStructure:
    """Architecture nodes/edges and their origin."""

    stages: tuple[Stage, ...]
    edges: tuple[tuple[str, str], ...]
    provenance: Literal["placeholder", "adapter"]


@dataclass(frozen=True)
class LoadOptions:
    """Lazy loading controls; downloads require an explicit opt-in."""

    device: str = "cpu"
    cache_dir: str | None = None
    local_files_only: bool = True
    seed: int = 0
    cpu_threads: int = 8


@dataclass(frozen=True)
class CaptureOptions:
    """Opt-in summaries. Module names select adapter-approved points only.

    max_tensors is the scalar preview length (the legacy field was unused).
    Full values are retained only for tensors with at most 64 elements.
    """

    module_names: tuple[str, ...] = ()
    max_tensors: int = 4
    enabled: bool = False
    max_token_norms: int = 1024
    max_records: int = 48
    extended: bool = False
    token_channels: int = 16

    def __post_init__(self) -> None:
        if not 1 <= self.token_channels <= 32:
            raise ValueError('Partial token-channel capture must select 1–32 channels.')
        if not 1 <= self.max_tensors <= 64:
            raise ValueError("Scalar preview length must be in [1, 64].")
        if not 1 <= self.max_token_norms <= 1024:
            raise ValueError("Token norm limit must be in [1, 1024].")
        if not 1 <= self.max_records <= 64:
            raise ValueError("Capture record limit must be in [1, 64].")


@dataclass(frozen=True)
class TensorSummary:
    """Detached measurements serialized as numbers, never retained tensors.

    std is population standard deviation. Token indices refer to flattened
    leading dimensions of a rank-three embedding tensor; norms use its last axis.
    """

    shape: tuple[int, ...]
    dtype: str
    device: str
    numel: int
    minimum: float
    maximum: float
    mean: float
    std: float
    l2_norm: float
    preview: tuple[float, ...]
    values: tuple[float, ...] | None = None
    token_indices: tuple[int, ...] = ()
    token_norms: tuple[float, ...] = ()
    token_norm_statistics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CapturedStage:
    """One measured semantic event; framework-independent visualization contract."""

    stage_name: str
    module_name: str
    description: str
    source: Literal["forward_hook", "processor_value", "model_return", "model_input"]
    inputs: Mapping[str, TensorSummary]
    outputs: Mapping[str, TensorSummary]
    context: Mapping[str, Any] = field(default_factory=dict)


# Native preprocessed batches remain opaque to the UI and visualization layer.
PreparedInputs = Mapping[str, Any]


@dataclass(frozen=True)
class InferenceResult:
    """One real run, with CPU action rows indexed by step and matching captures."""

    actions: tuple[tuple[float, ...], ...]
    intermediates: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
