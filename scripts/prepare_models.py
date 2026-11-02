"""Prepare only pinned model artifacts; workers run offline from cache."""

import argparse
from pathlib import Path
import json

from vla_visualizer.adapters.smolvla import (
    SmolVLAAdapter, CHECKPOINT, REVISION, CHECKPOINT_FILES,
    BACKBONE, BACKBONE_REVISION, BACKBONE_FILES, file_sha256,
)
from vla_visualizer.config import cache_dir
from vla_visualizer.contracts import LoadOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair", choices=["baseline", "simulation"])
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/model_preparation.json"))
    args = parser.parse_args()
    model, revision, files = CHECKPOINT, REVISION, CHECKPOINT_FILES
    if args.pair == "simulation":
        from vla_visualizer.simulation.compatibility import MODEL, REVISION as SIM_REVISION
        model, revision, files = MODEL, SIM_REVISION, (*CHECKPOINT_FILES, "train_config.json")
    options = LoadOptions(cache_dir=str(cache_dir()), local_files_only=not args.allow_download)
    records = []
    for repo, pin, names in ((model, revision, files), (BACKBONE, BACKBONE_REVISION, BACKBONE_FILES)):
        path = Path(SmolVLAAdapter._snapshot(repo, pin, names, options))
        records.append({"repo": repo, "revision": pin, "files": {
            name: {"bytes": (path / name).stat().st_size, "sha256": file_sha256(path / name)} for name in names}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
