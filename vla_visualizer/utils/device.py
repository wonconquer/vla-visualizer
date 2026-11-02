"""Detect usable PyTorch devices without modifying GPU configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DeviceInfo:
    """Device availability, distinct from whether a VLA model is loaded."""

    device: Literal["cpu", "cuda"]
    torch_build: str
    cuda_runtime: str | None
    cuda_available: bool
    cuda_device_name: str | None = None
    detail: str | None = None


def detect_device() -> DeviceInfo:
    """Prefer accessible CUDA; fall back to CPU if driver checks fail."""
    import torch

    try:
        if torch.cuda.is_available():
            return DeviceInfo(
                device="cuda",
                torch_build=torch.__version__,
                cuda_runtime=torch.version.cuda,
                cuda_available=True,
                cuda_device_name=torch.cuda.get_device_name(0),
            )
    except (RuntimeError, AssertionError, OSError) as exc:
        detail = f"CUDA detection failed; using CPU. {exc}"
    else:
        detail = "CUDA is not currently accessible; CPU is available."

    return DeviceInfo(
        device="cpu",
        torch_build=torch.__version__,
        cuda_runtime=torch.version.cuda,
        cuda_available=False,
        detail=detail,
    )
