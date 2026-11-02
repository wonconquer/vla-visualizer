"""Immutable candidate and fail-closed source/runtime compatibility checks."""

import importlib.metadata as metadata
import importlib.util
from pathlib import Path

from vla_visualizer.config import cache_dir, path_setting

MODEL = "lerobot/smolvla_libero"
REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"
ASSETS_REPO = "lerobot/libero-assets"
ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
SUITE = "libero_spatial"
TASK_ID = 0
TASK_NAME = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
INSTRUCTION = TASK_NAME.replace("_", " ")
CAMERA_MAP = (
    ("observation.images.image", "observation.images.camera1"),
    ("observation.images.image2", "observation.images.camera2"),
)
STATE_NAMES = (
    "eef_x",
    "eef_y",
    "eef_z",
    "axis_angle_x",
    "axis_angle_y",
    "axis_angle_z",
    "gripper_left_qpos",
    "gripper_right_qpos",
)
ACTION_NAMES = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_axis_angle_x",
    "delta_axis_angle_y",
    "delta_axis_angle_z",
    "gripper",
)


def assets_dir() -> Path:
    return path_setting("VLA_LIBERO_ASSETS", ".cache/libero/assets")


def preflight() -> dict:
    required = {"lerobot": "0.6.1", "hf-libero": "0.1.4", "robosuite": "1.4.0", "mujoco": "3.3.7"}
    versions, blockers = {}, []
    for name, expected in required.items():
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
        if versions[name] != expected:
            blockers.append(f"{name}=={expected} required; installed: {versions[name]}")
    snapshot = cache_dir() / "models--lerobot--smolvla_libero" / "snapshots" / REVISION
    from vla_visualizer.adapters.smolvla import CHECKPOINT_FILES

    missing = [
        name for name in (*CHECKPOINT_FILES, "train_config.json") if not (snapshot / name).is_file()
    ]
    if missing:
        blockers.append("Pinned LIBERO checkpoint files missing: " + ", ".join(missing))
    if not (assets_dir() / "task_assets.json").is_file():
        blockers.append("The selected task has no verified minimal asset manifest.")
    if not importlib.util.find_spec("mujoco"):
        blockers.append("MuJoCo rendering probe cannot run because the dependency is missing.")
    return {
        "status": "blocked" if blockers else "prerequisites_present",
        "integration_verified": False,
        "task_success": None,
        "blockers": blockers,
        "model": MODEL,
        "revision": REVISION,
        "suite": SUITE,
        "task_id": TASK_ID,
        "instruction": INSTRUCTION,
        "versions": versions,
        "assets_revision": ASSETS_REVISION,
    }


def verify_assets() -> dict:
    import json
    from vla_visualizer.adapters.smolvla import file_sha256

    directory = assets_dir()
    manifest = json.loads((directory / "task_assets.json").read_text())
    if (
        manifest.get("repo") != ASSETS_REPO
        or manifest.get("revision") != ASSETS_REVISION
        or manifest.get("task_name") != TASK_NAME
        or not manifest.get("files")
    ):
        raise ValueError("Asset repository/revision/file-list validation failed")
    for name, digest in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_sha256(path) != digest:
            raise ValueError(f"Asset checksum validation failed: {name}")
    return manifest


def worker_readiness() -> dict:
    """Inspect prerequisites in the configured simulation worker environment."""
    import json
    import subprocess
    import sys
    from vla_visualizer.config import ROOT, worker_python

    interpreter = worker_python("simulation")
    if Path(interpreter).resolve() == Path(sys.executable).resolve():
        return preflight()
    command = [
        interpreter,
        "-c",
        "import json; from vla_visualizer.simulation.compatibility import preflight; print(json.dumps(preflight()))",
    ]
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10)
        if result.returncode:
            raise RuntimeError(result.stderr[-1500:])
        return {
            **json.loads(result.stdout),
            "readiness_scope": "configured simulator worker; no model loaded",
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "integration_verified": False,
            "task_success": None,
            "blockers": [f"Configured simulation worker readiness check failed: {type(exc).__name__}: {exc}"],
            "model": MODEL,
            "revision": REVISION,
            "suite": SUITE,
            "task_id": TASK_ID,
        }
