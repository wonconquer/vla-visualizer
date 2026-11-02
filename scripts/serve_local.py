"""Launch locally with an existing project cache; never download automatically."""

import os

from vla_visualizer.config import ROOT
from vla_visualizer.adapters.smolvla import CHECKPOINT, CHECKPOINT_FILES, REVISION


def configure_environment() -> None:
    """Prefer the operator setting and project cache before checking a sibling cache."""
    if "VLA_MODEL_CACHE" in os.environ:
        return
    relative = f"models--{CHECKPOINT.replace('/', '--')}/snapshots/{REVISION}"
    candidates = (
        ROOT / ".cache/huggingface/hub",
        ROOT.parent / ".cache/huggingface/ece525/hub",
    )
    for cache in candidates:
        if all((cache / relative / name).is_file() for name in CHECKPOINT_FILES):
            os.environ["VLA_MODEL_CACHE"] = str(cache.resolve())
            return
    # Keep the portable default when no cache is prepared; the UI reports missing artifacts.


def main() -> None:
    configure_environment()
    from scripts.serve import main as serve

    serve()


if __name__ == "__main__":
    main()
