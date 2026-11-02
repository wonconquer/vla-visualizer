"""Prepare at most two pinned SO100 episodes, reusing resumable Hub cache."""

import argparse
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

from vla_visualizer.config import ROOT, cache_dir
from vla_visualizer.data.so100 import dataset_root
from vla_visualizer.adapters.smolvla import file_sha256
from vla_visualizer.adapters.smolvla_config import MVP_CHECKPOINT
from vla_visualizer.execution.jobs import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--reuse-source", type=Path, help="Existing smoke-source directory (optional)")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if any(i < 0 or i >= 50 for i in args.episodes) or len(args.episodes) > 2:
        raise ValueError("Prepare at most two pinned episodes per request.")
    root = dataset_root()
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "repo": MVP_CHECKPOINT.dataset,
        "revision": MVP_CHECKPOINT.dataset_revision,
        "purpose": "recorded observation exploration; not simulator assets",
        "estimated_bytes": 24 * 1024**2,
        "estimate_scope": "two episodes, approximate; actual sizes recorded per file",
        "files": [],
        "resumable": "HF content-addressed cache and .incomplete downloads",
        "episodes_requested": args.episodes,
    }
    attempt = root / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")
    attempt.parent.mkdir(exist_ok=True)
    atomic_json(attempt, report)
    metadata = json.loads((ROOT / "docs/smolvla_pairing.json").read_text())
    original = json.loads((ROOT / "assets/sample_inputs/so100_observation.json").read_text())
    source_files = {f["rfilename"]: f for f in metadata["sample"]["source_files"]}
    reuse = {}
    if args.reuse_source:
        reuse = {
            "data/chunk-000/episode_000000.parquet": args.reuse_source / "episode_000000.parquet",
            "videos/chunk-000/observation.images.top/episode_000000.mp4": args.reuse_source
            / "top.mp4",
            "meta/tasks.jsonl": args.reuse_source / "tasks.jsonl",
        }
    wrist = "videos/chunk-000/observation.images.wrist/episode_000000.mp4"
    cached = (
        cache_dir()
        / ("datasets--" + MVP_CHECKPOINT.dataset.replace("/", "--"))
        / "snapshots"
        / MVP_CHECKPOINT.dataset_revision
        / wrist
    )
    if cached.is_file():
        reuse[wrist] = cached
    names = ["meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"]
    for ep in args.episodes:
        names += [f"data/chunk-000/episode_{ep:06d}.parquet"] + [
            f"videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4"
            for cam in ("top", "wrist")
        ]
    for name in names:
        path = root / name
        entry = {
            "file": name,
            "expected_bytes": source_files.get(name, {}).get("size"),
            "local_path": str(path),
            "status": "pending",
            "downloaded_bytes": 0,
        }
        try:
            if not path.is_file() and name in reuse and reuse[name].is_file():
                expected = source_files.get(name, {}).get("lfs", {}).get("sha256")
                if expected and file_sha256(reuse[name]) != expected:
                    raise ValueError("Existing cache checksum mismatch")
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(reuse[name], path)
                entry["status"] = "reused_verified_cache"
            elif not path.is_file():
                from huggingface_hub import hf_hub_download

                downloaded = hf_hub_download(
                    MVP_CHECKPOINT.dataset,
                    name,
                    repo_type="dataset",
                    revision=MVP_CHECKPOINT.dataset_revision,
                    cache_dir=str(cache_dir()),
                    local_files_only=not args.allow_download,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(downloaded, path)
                entry.update(status="downloaded", downloaded_bytes=path.stat().st_size)
            else:
                entry["status"] = "already_prepared"
            entry.update(bytes=path.stat().st_size, sha256=file_sha256(path))
        except Exception as exc:
            entry.update(status="blocked", error=f"{type(exc).__name__}: {str(exc)[:800]}")
        report["files"].append(entry)
        atomic_json(root / "preparation_manifest.json", report)
        atomic_json(attempt, report)
    prepared = (
        json.loads((root / "catalog.json").read_text()).get("episodes", {})
        if (root / "catalog.json").exists()
        else {}
    )
    import pyarrow.parquet as pq

    for ep in args.episodes:
        parquet = f"data/chunk-000/episode_{ep:06d}.parquet"
        videos = {
            f"observation.images.{cam}": f"videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4"
            for cam in ("top", "wrist")
        }
        if all((root / name).is_file() for name in [parquet, *videos.values()]):
            rows = pq.read_table(root / parquet).to_pylist()
            assert all(
                r["episode_index"] == ep and r["frame_index"] == i for i, r in enumerate(rows)
            )
            prepared[str(ep)] = {
                "parquet": parquet,
                "videos": videos,
                "frames": len(rows),
                "timestamp_range": [rows[0]["timestamp"], rows[-1]["timestamp"]],
                "source_sha256": {
                    name: file_sha256(root / name) for name in [parquet, *videos.values()]
                },
            }
    tasks = (
        {
            str(x["task_index"]): x["task"]
            for x in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())
        }
        if (root / "meta/tasks.jsonl").exists()
        else {"0": original["instruction"]}
    )
    atomic_json(
        root / "catalog.json",
        {
            "repo": report["repo"],
            "revision": report["revision"],
            "remote_metadata": metadata["dataset"]["old_info"],
            "metadata_source": "preserved pinned Phase 2 audit; live refresh status in manifest",
            "episodes": prepared,
            "prepared_episode_count": len(prepared),
            "tasks": tasks,
        },
    )
    print(
        json.dumps(
            {"prepared": prepared, "manifest": str(root / "preparation_manifest.json")},
            ensure_ascii=False,
        )
    )
    return 0 if len(prepared) == len(args.episodes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
