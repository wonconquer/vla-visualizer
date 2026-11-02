"""Report file and package readiness without loading a model."""

import importlib.metadata
import json

from vla_visualizer.config import cache_dir, sample_path


def readiness() -> dict:
    from vla_visualizer.adapters.smolvla import (
        CHECKPOINT, REVISION, CHECKPOINT_FILES, BACKBONE, BACKBONE_REVISION, BACKBONE_FILES,
    )
    missing = []
    for repo, pin, files in ((CHECKPOINT, REVISION, CHECKPOINT_FILES), (BACKBONE, BACKBONE_REVISION, BACKBONE_FILES)):
        source = cache_dir() / ("models--" + repo.replace("/", "--")) / "snapshots" / pin
        missing.extend(f"{repo}/{name}" for name in files if not (source / name).is_file())
    from vla_visualizer.simulation.compatibility import worker_readiness
    return {
        "viewer_ready": sample_path().is_file(), "baseline_artifacts_ready": not missing,
        "baseline_missing": missing, "model_loaded": False,
        "normalization_and_weights_checked_on_load": True,
        "simulation": worker_readiness(), "streamlit": importlib.metadata.version("streamlit"),
        "scope": "File/package readiness only, not evidence of inference or task success.",
    }


if __name__ == "__main__":
    report = readiness()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["baseline_artifacts_ready"] else 2)
