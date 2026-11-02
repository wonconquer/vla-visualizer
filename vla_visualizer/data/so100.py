"""Read SO100 v2.1 episodes without migrating the source dataset."""

from io import BytesIO
import json
import subprocess

from PIL import Image

from vla_visualizer.config import ROOT, path_setting
from vla_visualizer.adapters.smolvla_config import MVP_CHECKPOINT
from vla_visualizer.contracts import Observation


def dataset_root():
    return path_setting(
        "VLA_DATASET_DIR", ".cache/datasets/so100/" + MVP_CHECKPOINT.dataset_revision
    )


def catalog():
    path = dataset_root() / "catalog.json"
    return (
        json.loads(path.read_text())
        if path.exists()
        else {"episodes": {}, "remote_metadata": None, "prepared_episode_count": 0}
    )


def episode_rows(episode):
    import pyarrow.parquet as pq

    item = catalog()["episodes"].get(str(episode))
    if not item:
        raise ValueError("This episode is not prepared locally.")
    return pq.read_table(dataset_root() / item["parquet"]).to_pylist()


def decode_frame(path, frame):
    # Select by decoded frame index rather than approximate keyframe seeking.
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        "2",
        "-i",
        str(path),
        "-vf",
        f"select=eq(n\\,{frame})",
        "-frames:v",
        "1",
        "-vsync",
        "0",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-threads",
        "2",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, timeout=45)
    if result.returncode or not result.stdout:
        raise RuntimeError(
            "Video frame decode failed: " + result.stderr.decode(errors="replace")[-1000:]
        )
    return Image.open(BytesIO(result.stdout)).convert("RGB")


def observation(episode, frame):
    from vla_visualizer.adapters.smolvla import file_sha256

    item = catalog()["episodes"].get(str(episode))
    rows = episode_rows(episode)
    for name, digest in item.get("source_sha256", {}).items():
        if file_sha256(dataset_root() / name) != digest:
            raise ValueError("Dataset source checksum changed: " + name)
    if not 0 <= frame < len(rows):
        raise ValueError("Frame index is outside this episode.")
    row = rows[frame]
    if row["episode_index"] != episode or row["frame_index"] != frame:
        raise ValueError("Parquet episode/frame indices do not match")
    original = json.loads((ROOT / "assets/sample_inputs/so100_observation.json").read_text())
    tasks = catalog()["tasks"]
    instruction = tasks[str(row["task_index"])]
    images = {
        key: decode_frame(dataset_root() / name, frame) for key, name in item["videos"].items()
    }
    provenance = {
        **original["provenance"],
        "episode_index": episode,
        "frame_index": frame,
        "timestamp_seconds": float(row["timestamp"]),
        "task_index": int(row["task_index"]),
        "sample_identity": f"dataset episode {episode} / frame {frame} / timestamp {row['timestamp']} / task_index {row['task_index']}",
        "recorded_instruction": instruction,
        "state_source": item["parquet"],
        "image_sources": item["videos"],
        "observation_source": "recorded dataset; not simulator",
        "sensor_acquisition_synchronization": "not independently measured; matched dataset frame/timestamp",
        "parquet_sha256": file_sha256(dataset_root() / item["parquet"]),
    }
    provenance.pop("image_sha256_by_camera", None)
    provenance.pop("image_sha256", None)
    return Observation(
        images=images,
        instruction=instruction,
        state=tuple(row["observation.state"]),
        provenance=provenance,
    )


def target_chunk(episode, frame, count=50):
    import torch

    rows = episode_rows(episode)
    if not 0 <= frame < len(rows):
        raise ValueError("Invalid target frame")
    valid = rows[frame : frame + count]
    if any(row["episode_index"] != episode for row in valid):
        raise ValueError("Future targets cross an episode boundary.")
    values = [row["action"] for row in valid]
    mask = [False] * len(values) + [True] * (count - len(values))
    values += [values[-1]] * (count - len(values))
    return torch.tensor(values, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)
