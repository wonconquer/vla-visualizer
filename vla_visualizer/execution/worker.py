"""Explicit worker dispatch. Selector/replay controls never import policy code."""

from dataclasses import asdict, replace
from pathlib import Path
import signal
import sys
import time

from vla_visualizer.config import cache_dir, sample_path
from vla_visualizer.execution.jobs import atomic_json, read_json


class Cancelled(RuntimeError):
    pass


_termination_requested = False


def check_cancel(directory: Path) -> None:
    if _termination_requested or (directory / "cancel").exists():
        raise Cancelled("The user cancelled this job.")


def progress(directory: Path, phase: str, **values) -> None:
    check_cancel(directory)
    atomic_json(
        directory / "progress.json", {"phase": phase, "updated_unix": time.time(), **values}
    )


def baseline(directory: Path, config: dict) -> None:
    from vla_visualizer.contracts import CaptureOptions, LoadOptions
    from vla_visualizer.inference.observations import read_observation
    from vla_visualizer.inference.replay import make_run
    from vla_visualizer.inference.runner import InferenceRunner

    runner = InferenceRunner()
    try:
        progress(directory, "Loading model")
        runner.load(
            "smolvla",
            LoadOptions(
                device=config["device"],
                seed=config["seed"],
                cache_dir=str(cache_dir()),
            ),
        )
        if config.get("observation_mode") == "dataset":
            from vla_visualizer.data.so100 import observation as dataset_observation

            observation = dataset_observation(config["episode_index"], config["frame_index"])
        else:
            observation = read_observation(sample_path())
        if config.get("instruction") is not None:
            from vla_visualizer.adapters.education_catalog import validate_instruction

            validate_instruction(config["instruction"], "baseline")
            provenance = dict(observation.provenance)
            if config["instruction"] != observation.instruction:
                provenance.update(
                    {
                        "recorded_instruction": observation.instruction,
                        "instruction_origin": "Educational sentence variation; task success not verified",
                    }
                )
            observation = replace(
                observation, instruction=config["instruction"], provenance=provenance
            )
        progress(directory, "Running actual SO100 dataset inference")
        result = runner.run_inference(
            "smolvla", observation, CaptureOptions(enabled=True, extended=True)
        )
        payload = asdict(result)
        payload["metadata"]["job_run_id"] = directory.name
        atomic_json(directory / "result.json", payload)
        replay = make_run(result, runner.get_inspection_spec("smolvla"), observation)
        (directory / "replay.json").write_text(replay.snapshot)
        atomic_json(
            directory / "links.json",
            {
                "job_run_id": directory.name,
                "capture_run_id": replay.run_id,
                "result": "result.json",
                "replay": "replay.json",
            },
        )
        progress(directory, "Completed", inference_requests=1, simulator_steps=0, task_success=None)
    finally:
        runner.unload()


def main(directory: Path) -> int:
    def cancel_signal(signum, frame):
        # Raising asynchronously during native-library initialization is unsafe.
        # Poll at Python boundaries; the supervisor enforces a hard process deadline.
        global _termination_requested
        _termination_requested = True

    signal.signal(signal.SIGTERM, cancel_signal)
    config = read_json(directory / "config.json")
    import importlib.metadata
    import os

    versions = {}
    for name in (
        "streamlit",
        "torch",
        "lerobot",
        "transformers",
        "hf-libero",
        "robosuite",
        "mujoco",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    atomic_json(
        directory / "runtime.json",
        {
            "python_executable": sys.executable,
            "pid": os.getpid(),
            "versions": versions,
            "requested_device": config["device"],
            "model_cache": str(cache_dir()),
            "output_directory": str(directory),
            "render_backend": os.environ.get("MUJOCO_GL"),
        },
    )
    try:
        if config["kind"] == "baseline":
            baseline(directory, config)
        elif config["kind"] == "simulation":
            from vla_visualizer.simulation.rollout import run_rollout

            run_rollout(directory, config)
        else:
            from vla_visualizer.execution.training import train, evaluate

            (train if config["kind"] == "training" else evaluate)(directory, config)
        return 0
    except Exception as exc:
        import traceback

        traceback.print_exc()
        partial_path = directory / "rollout.json"
        partial = read_json(partial_path) if partial_path.exists() else {}
        atomic_json(
            directory / "error.json",
            {
                "type": type(exc).__name__,
                "message": str(exc)[:3000],
                "run_id": directory.name,
                "inference_success": partial.get("inference_success", False),
                "simulator_integration_success": partial.get(
                    "simulator_integration_success", False
                ),
                "task_success": partial.get("task_success"),
            },
        )
        print(f"{type(exc).__name__}: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    sys.modules["vla_visualizer.execution.worker"] = sys.modules[__name__]
    sys.exit(main(Path(sys.argv[1])))
