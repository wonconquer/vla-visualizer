"""Record actual simulator stage diagnostics without treating missing stages as success."""

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from vla_visualizer.config import ROOT, cache_dir
from vla_visualizer.execution.jobs import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/integration/simulation"
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    rollout = None
    args.output.mkdir(parents=True, exist_ok=True)
    required = {"lerobot": "0.6.1", "hf-libero": "0.1.4", "robosuite": "1.4.0", "mujoco": "3.3.7"}
    versions = {}
    for name in required:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    stages = {
        key: {"status": "NOT_RUN", "reason": "dependency/asset prerequisites not verified"}
        for key in [
            "A_dependency",
            "B_assets",
            "C_reset",
            "D_render",
            "E_checkpoint",
            "F_observation_normalization",
            "G_inference",
            "H_step",
            "I_repeated_inference",
            "J_cleanup",
            "J_stop",
            "K_cleanup",
        ]
    }
    command = [
        sys.executable,
        "-c",
        'import lerobot; import libero.libero; import robosuite; import mujoco; print("imports OK")',
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    (args.output / "A_dependency.log").write_text(result.stdout + result.stderr)
    stages["A_dependency"] = {
        "status": "PASS" if result.returncode == 0 and versions == required else "FAIL",
        "versions": versions,
        "required_versions": required,
        "command": command,
        "log": str(args.output / "A_dependency.log"),
    }
    from vla_visualizer.simulation.compatibility import preflight, assets_dir, REVISION

    report = preflight()
    manifest = assets_dir() / "task_assets.json"
    assets_verified = False
    asset_error = None
    if manifest.is_file():
        try:
            from vla_visualizer.simulation.compatibility import verify_assets

            verify_assets()
            assets_verified = True
        except (ValueError, OSError, ImportError) as exc:
            asset_error = str(exc)
    stages["B_assets"] = {
        "status": "PASS" if assets_verified else "FAIL",
        "asset_manifest_exists": manifest.is_file(),
        "asset_manifest": str(manifest),
        "initial_state_package_available": importlib.util.find_spec("libero") is not None,
        "hash_verification_error": asset_error,
    }
    snapshot = cache_dir() / "models--lerobot--smolvla_libero" / "snapshots" / REVISION
    stages["E_checkpoint"] = {
        "status": "FAIL" if not (snapshot / "model.safetensors").exists() else "NOT_RUN",
        "weights_present": (snapshot / "model.safetensors").exists(),
        "path": str(snapshot),
        "reason": "immutable checkpoint files unavailable"
        if not snapshot.exists()
        else "requires actual strict load",
    }
    if not args.quick:
        render = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.render_probe",
                "--output",
                str(args.output / "render_probe.json"),
            ],
            capture_output=True,
            text=True,
            timeout=65,
            cwd=ROOT,
        )
        (args.output / "D_render.log").write_text(render.stdout + render.stderr)
        backend_probe = {
            "status": "PASS" if render.returncode == 0 else "FAIL",
            "command": [sys.executable, "-m", "scripts.render_probe"],
            "log": str(args.output / "render_probe.json"),
        }
        stages["D_render"] = {
            "status": "NOT_RUN",
            "reason": "The backend probe is separate from a genuine initial task scene.",
            "backend_probe": backend_probe,
        }
        if not report["blockers"] and render.returncode == 0:
            # Only the existing bounded real worker can establish C/F/G/H/I/J.
            from vla_visualizer.simulation.rollout import run_rollout
            import uuid

            directory = args.output / uuid.uuid4().hex
            directory.mkdir()
            atomic_json(directory / "config.json", {"kind": "simulation"})
            try:
                run_rollout(
                    directory,
                    {
                        "device": "cpu",
                        "seed": 0,
                        "max_steps": 20,
                        "action_steps": 10,
                        "max_seconds": 300,
                    },
                )
                rollout = json.loads((directory / "rollout.json").read_text())
            except Exception:
                (directory / "traceback.log").write_text(traceback.format_exc())
            journal = directory / "stages.json"
            if journal.exists():
                actual = json.loads(journal.read_text())["stages"]
                aliases = {
                    "C_reset": "C_reset",
                    "D_render": "D_render",
                    "E_checkpoint": "E_policy",
                    "F_observation_normalization": "F_preprocessing",
                    "G_inference": "G_inference",
                    "H_step": "H_action_step",
                    "I_repeated_inference": "I_repeated_inference",
                    "J_stop": "J_stop",
                    "K_cleanup": "K_cleanup",
                    "J_cleanup": "K_cleanup",
                }
                for key, source in aliases.items():
                    stages[key] = {**actual[source], "log": str(journal)}
    ready = (
        all(stages[key]["status"] == "PASS" for key in ("A_dependency", "B_assets"))
        and not report["blockers"]
    )
    output = {
        "schema_version": 2,
        "status": "prerequisites_ready" if ready else "blocked",
        "ready_to_request_rollout": ready,
        "integration_verified": all(v["status"] == "PASS" for v in stages.values()),
        "stages": stages,
        "blockers": report["blockers"],
        "command_environment": {"MUJOCO_GL": os.environ.get("MUJOCO_GL")},
        "task_success": rollout.get("task_success") if rollout else None,
        "simulator_steps": rollout.get("control_steps", 0) if rollout else 0,
        "termination": rollout.get("termination") if rollout else None,
        "scope": "prerequisites only"
        if args.quick
        else "actual staged attempt; backend probe is not the task scene",
    }
    atomic_json(args.output / "stages.json", output)
    print(json.dumps(output, ensure_ascii=False))
    return 0 if output["integration_verified"] or (args.quick and ready) else 2


if __name__ == "__main__":
    raise SystemExit(main())
