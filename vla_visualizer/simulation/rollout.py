"""Bounded closed-loop implementation; absent dependencies never produce a fake rollout."""

import os
from pathlib import Path
import resource
import time

from vla_visualizer.execution.jobs import atomic_json
from vla_visualizer.execution.worker import progress, check_cancel
from vla_visualizer.simulation.compatibility import (
    MODEL,
    REVISION,
    SUITE,
    TASK_ID,
    INSTRUCTION,
    ACTION_NAMES,
    CAMERA_MAP,
    preflight,
)


def run_rollout(directory: Path, config: dict):
    stages = {
        key: {"status": "NOT_RUN"}
        for key in (
            "A_dependency",
            "B_assets",
            "C_reset",
            "D_render",
            "E_policy",
            "F_preprocessing",
            "G_inference",
            "H_action_step",
            "I_repeated_inference",
            "J_stop",
            "K_cleanup",
        )
    }

    def mark(key, status, **details):
        stages[key] = {"status": status, "timestamp_unix": time.time(), **details}
        atomic_json(
            directory / "stages.json",
            {
                "stages": stages,
                "scope": "Actual worker boundaries; readiness failures are not successful imports.",
            },
        )

    report = preflight()
    atomic_json(directory / "preflight.json", report)
    if report["blockers"]:
        mark(
            "A_dependency",
            "FAIL",
            reason="Readiness gate failed; see actual dependency-import diagnostics.",
            blockers=report["blockers"],
        )
        raise RuntimeError("LIBERO execution blocked: " + "; ".join(report["blockers"]))
    if os.environ.get("MUJOCO_GL") not in {"egl", "osmesa"}:
        raise RuntimeError(
            "Run a real rendering probe, then set MUJOCO_GL to the verified egl or osmesa backend."
        )
    import numpy as np
    from PIL import Image
    from vla_visualizer.simulation.environment import LiberoEnvironment
    from vla_visualizer.simulation.policy import LiberoPolicy

    (directory / "frames").mkdir()
    (directory / "captures").mkdir()
    environment, policy = LiberoEnvironment(), LiberoPolicy(config["device"], config["seed"])
    start = time.monotonic()
    record = {
        "schema_version": 1,
        "run_id": directory.name,
        "model": MODEL,
        "revision": REVISION,
        "environment": "LIBERO",
        "suite": SUITE,
        "task_id": TASK_ID,
        "seed": config["seed"],
        "instruction": INSTRUCTION,
        "versions": report["versions"],
        "control_hz": 20,
        "warmup_steps": 10,
        "replan_actions": config["action_steps"],
        "max_steps": config["max_steps"],
        "action_names": ACTION_NAMES,
        "image_transform": "raw HWC uint8 → BCHW float [0,1] → native H/W flip (180°) → checkpoint camera rename → native resize/pad 512² / [-1,1]",
        "camera_map": CAMERA_MAP,
        "requests": [],
        "frames": [],
        "steps": [],
        "inference_success": False,
        "simulator_integration_success": False,
        "task_success": None,
        "termination": "initializing",
        "cleanup_completed": False,
        "timing_scope": "inference=prediction+inverse-normalization+capture; env.step=physics+camera rendering+success; artifact encoding separate",
        "stages": stages,
        "video": None,
        "truncated": False,
    }
    total = 0
    current_stage = "C_reset"
    try:
        progress(directory, "LIBERO reset and headless camera rendering")
        observation, info = environment.reset(config["seed"])
        mark("A_dependency", "PASS", scope="Imports performed by the actual environment reset")
        mark("B_assets", "PASS", scope="Asset manifest hashes verified by environment.reset")
        mark("C_reset", "PASS")
        sim_start = environment.simulator_time()
        current_stage = "D_render"
        initial = "frames/initial_scene.png"
        Image.fromarray(np.ascontiguousarray(observation["pixels"]["image"][::-1, ::-1])).save(
            directory / initial
        )
        record["frames"].append({"file": initial, "control_step": 0, "simulator_time": sim_start})
        mark("D_render", "PASS", file=initial, backend=os.environ["MUJOCO_GL"])
        record["controller"] = environment.controller_metadata
        progress(
            directory,
            "Loading frozen LIBERO policy",
            latest_scene=initial,
            simulator_control_step=0,
            max_steps=config["max_steps"],
        )
        current_stage = "E_policy"
        policy.load()
        mark("E_policy", "PASS")
        record["policy"] = policy.metadata
        record["termination"] = "running"
        while total < config["max_steps"]:
            check_cancel(directory)
            if time.monotonic() - start >= config["max_seconds"]:
                record["termination"] = "time_limit"
                break
            request_index = len(record["requests"])
            inference_id = f"{directory.name}:inference:{request_index}"
            current_stage = "F_preprocessing"
            prepared, state, images = policy.prepare(observation, INSTRUCTION)
            mark("F_preprocessing", "PASS", observation_control_step=total)
            camera_files = {}
            for key, pixels in images.items():
                filename = f"frames/input_{request_index:03d}_{key.rsplit('.', 1)[-1]}.png"
                Image.fromarray(pixels).save(directory / filename)
                camera_files[key] = filename
            capture = request_index in {0, 1, 2}
            current_stage = "G_inference"
            progress(
                directory,
                "Running frozen policy inference",
                simulator_control_step=total,
                inference_request_index=request_index,
                max_steps=config["max_steps"],
                latest_scene=record["frames"][-1]["file"]
                if record["frames"]
                else camera_files[CAMERA_MAP[0][0]],
            )
            chunk, captured, seconds = policy.predict(prepared, state, capture=capture)
            mark("G_inference", "PASS", inference_request=request_index)
            if request_index >= 1 and total > 0:
                mark(
                    "I_repeated_inference",
                    "PASS",
                    inference_request=request_index,
                    observation_control_step=total,
                )
            record["inference_success"] = True
            capture_file = None
            if capture:
                capture_file = f"captures/inference_{request_index:03d}.json"
                atomic_json(
                    directory / capture_file,
                    {
                        "run_id": directory.name,
                        "inference_id": inference_id,
                        "capture": captured,
                        "record_ids": [
                            f"{inference_id}:capture:{i}" for i in range(len(captured["stages"]))
                        ],
                    },
                )
            request = {
                "id": inference_id,
                "index": request_index,
                "observation_control_step": total,
                "policy_input_frames": camera_files,
                "state_before_normalization": state,
                "state_after_normalization": prepared["observation.state"].detach().cpu().tolist(),
                "capture_file": capture_file,
                "predicted_action_chunk": chunk.tolist(),
                "executed": [],
                "inference_seconds": seconds,
                "timestamp_unix": time.time(),
            }
            record["requests"].append(request)
            for action_index, row in enumerate(chunk[: config["action_steps"]]):
                check_cancel(directory)
                if total >= config["max_steps"]:
                    break
                if time.monotonic() - start >= config["max_seconds"]:
                    record["termination"] = "time_limit"
                    break
                current_stage = "H_action_step"
                applied = policy.action(row)
                if applied.shape != (7,) or not np.isfinite(applied).all():
                    raise RuntimeError("Environment action processor output validation failed")
                before = environment.simulator_time()
                tick = time.monotonic()
                observation, reward, terminated, truncated, info = environment.step(applied)
                step_seconds = time.monotonic() - tick
                after = environment.simulator_time()
                total += 1
                mark("H_action_step", "PASS", control_step=total)
                record["simulator_integration_success"] = True
                step = {
                    "inference_id": inference_id,
                    "chunk_action_index": action_index,
                    "simulator_control_step": total,
                    "applied_action": applied.tolist(),
                    "simulator_time_range": [before, after],
                    "env_step_wall_seconds": step_seconds,
                    "physics_step_range_derived": [
                        round(before / record["controller"]["physics_timestep_seconds"]),
                        round(after / record["controller"]["physics_timestep_seconds"]),
                    ],
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "is_success": bool(info["is_success"]) if "is_success" in info else None,
                }
                request["executed"].append({"action_index": action_index, "control_step": total})
                record["steps"].append(step)
                # Same agentview camera as policy input, native orientation correction.
                # Every second control step: bounded 10 fps display at 20 Hz control.
                if total == 1 or total % 2 == 0 or terminated or truncated:
                    filename = f"frames/scene_{total:04d}.png"
                    tick = time.monotonic()
                    Image.fromarray(
                        np.ascontiguousarray(observation["pixels"]["image"][::-1, ::-1])
                    ).save(directory / filename)
                    record["frames"].append(
                        {
                            "file": filename,
                            "control_step": total,
                            "simulator_time": after,
                            "encoding_seconds": time.monotonic() - tick,
                        }
                    )
                    progress(
                        directory,
                        "Executing policy actions",
                        simulator_control_step=total,
                        inference_request_index=request_index,
                        latest_scene=filename,
                        max_steps=config["max_steps"],
                    )
                record["task_success"] = step["is_success"]
                atomic_json(directory / "rollout.json", record)
                if terminated or truncated:
                    record["termination"] = (
                        "success"
                        if record["task_success"]
                        else ("terminated" if terminated else "truncated")
                    )
                    break
            if record["termination"] != "running":
                break
        if record["termination"] == "running":
            record["termination"] = "step_limit"
        record["truncated"] = record["termination"] in ("step_limit", "time_limit", "truncated")
        current_stage = "J_stop"
        mark("J_stop", "PASS", reason=record["termination"], task_success=record["task_success"])
        record["simulator_seconds_excluding_warmup"] = environment.simulator_time() - sim_start
        atomic_json(
            directory / "result.json",
            {
                k: record[k]
                for k in (
                    "run_id",
                    "inference_success",
                    "simulator_integration_success",
                    "task_success",
                    "termination",
                )
            },
        )
    except Exception as exc:
        mark(current_stage, "FAIL", error=f"{type(exc).__name__}: {exc}")
        record["termination"] = type(exc).__name__
        record["error"] = str(exc)[:2000]
        raise
    finally:
        cleanup_errors = []
        for component in (environment, policy):
            try:
                component.close()
            except Exception as exc:
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        mark("K_cleanup", "FAIL" if cleanup_errors else "PASS", errors=cleanup_errors)
        record.update(
            {
                "cleanup_completed": not cleanup_errors,
                "wall_seconds": time.monotonic() - start,
                "control_steps": total,
                "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )
        if not (directory / "cancel").exists() and record["termination"] != "Cancelled":
            encoding_start = time.monotonic()
            try:
                from vla_visualizer.simulation.video import encode_rollout

                record["video"] = encode_rollout(directory, record)
            except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
                record["video_error"] = str(exc)
            record["video_encoding_seconds"] = time.monotonic() - encoding_start
        record["wall_seconds"] = time.monotonic() - start
        atomic_json(directory / "rollout.json", record)
        if cleanup_errors and "error" not in record:
            raise RuntimeError("Simulator cleanup failed: " + "; ".join(cleanup_errors))
