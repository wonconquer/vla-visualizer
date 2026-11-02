"""Portable runtime paths and conservative, operator-controlled resource ceilings."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def path_setting(name: str, default: str) -> Path:
    path = Path(os.environ.get(name, default)).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def bounded_setting(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.environ.get(name, default))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name}: {minimum}–{maximum} is the supported range.")
    return value


def cache_dir() -> Path:
    return path_setting("VLA_MODEL_CACHE", ".cache/huggingface/hub")


def artifact_root() -> Path:
    return path_setting("VLA_ARTIFACT_DIR", "outputs/jobs")


def worker_python(kind: str) -> str:
    key = "VLA_SIM_PYTHON" if kind == "simulation" else "VLA_WORKER_PYTHON"
    return os.environ.get(key, sys.executable)


def sample_path() -> Path:
    return path_setting("VLA_SAMPLE", "assets/sample_inputs/so100_observation.json")


def training_data() -> Path:
    return path_setting("VLA_TRAINING_DATA", "assets/training/so100_episode0.parquet")
