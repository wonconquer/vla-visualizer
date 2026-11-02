"""Prepare a reviewed minimal asset list for one LIBERO task."""

import argparse
import json
from pathlib import Path

from vla_visualizer.simulation.compatibility import (
    ASSETS_REPO, ASSETS_REVISION, TASK_NAME, assets_dir,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Reviewed JSON with task_name and files (exact relative paths). No verified minimal list is supplied yet.")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    files = source.get("files")
    if source.get("task_name") != TASK_NAME or not isinstance(files, list) or not 1 <= len(files) <= 2000:
        raise ValueError("A reviewed file list for the selected task is required.")
    if any(not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or any(c in name for c in "*?[]") for name in files):
        raise ValueError("Only exact relative file paths are allowed.")
    from huggingface_hub import snapshot_download, HfApi
    if args.allow_download:
        info = HfApi().get_paths_info(ASSETS_REPO, paths=files, repo_type="dataset", revision=ASSETS_REVISION)
        if len(info) != len(files) or any(not hasattr(item, "size") for item in info):
            raise ValueError("Every file size must be verified.")
        if sum(item.size for item in info) > 64 * 1024**2:
            raise ValueError("Task asset download exceeds the 64 MiB limit")
    directory = Path(snapshot_download(ASSETS_REPO, repo_type="dataset", revision=ASSETS_REVISION,
        allow_patterns=files, local_files_only=not args.allow_download, local_dir=assets_dir(), max_workers=2))
    from vla_visualizer.adapters.smolvla import file_sha256
    manifest = {"repo": ASSETS_REPO, "revision": ASSETS_REVISION, "task_name": TASK_NAME,
                "files": {name: file_sha256(directory / name) for name in files}}
    (directory / "task_assets.json").write_text(json.dumps(manifest, indent=2))
    print(directory / "task_assets.json")


if __name__ == "__main__":
    main()
