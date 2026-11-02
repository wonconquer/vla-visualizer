"""Own one worker process group, lease and deadline independently of UI reruns."""

import os
from pathlib import Path
import signal
import resource
import shutil
import subprocess
import sys
import time

from vla_visualizer.config import ROOT, bounded_setting
from vla_visualizer.execution.jobs import atomic_json, read_json


def group_rss_kib(pid: int) -> int:
    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(stat[2]) == pid:  # process group
                status = (entry / "status").read_text()
                line = next((x for x in status.splitlines() if x.startswith("VmRSS:")), "VmRSS: 0")
                total += int(line.split()[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return total


def supervise(directory: Path, python: str) -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    status = read_json(directory / "status.json")
    config = read_json(directory / "config.json")
    start = time.monotonic()
    process = None
    peak_rss = 0
    peak_bytes = 0
    reason = None
    env = {
        **os.environ,
        "VLA_SUPERVISED": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
    }
    try:
        with (directory / "worker.log").open("wb") as log:
            process = subprocess.Popen(
                [python, "-m", "vla_visualizer.execution.worker", str(directory)],
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            atomic_json(
                directory / "status.json",
                {
                    **status,
                    "state": "running",
                    "started_unix": time.time(),
                },
            )
            from vla_visualizer.execution.storage import usage, budgets

            budget = config.get("artifact_budget", budgets(config["kind"]))
            while process.poll() is None:
                peak_rss = max(peak_rss, group_rss_kib(process.pid))
                categories = usage(directory)
                size = sum(categories.values())
                peak_bytes = max(peak_bytes, size)
                if (directory / "cancel").exists():
                    reason = "cancelled"
                elif time.monotonic() - start > config["max_seconds"]:
                    reason = "timed_out"
                elif (
                    any(categories[key] > budget[key] for key in budget)
                    or peak_rss > bounded_setting("VLA_MAX_RSS_MB", 16384, 4096, 32768) * 1024
                ):
                    reason = "resource_limit"
                    atomic_json(
                        directory / "resource_limit.json",
                        {
                            "phase": "runtime_storage_or_rss",
                            "used_by_category": categories,
                            "budget_by_category": budget,
                            "rss_kib": peak_rss,
                        },
                    )
                elif (directory / "worker.log").stat().st_size > 512 * 1024:
                    reason = "resource_limit"
                if reason:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break
                time.sleep(0.25)
            if reason is None:
                reason = "completed" if process.returncode == 0 else "failed"
    except Exception as exc:
        reason = "failed"
        status["error"] = str(exc)[:1000]
    finally:
        if config["kind"] == "simulation":
            (directory / ".policy_rollout.staging.mp4").unlink(missing_ok=True)
        if process is not None:
            # Descendants are owned exclusively by this worker's process group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if (
            reason != "completed"
            and config["kind"] == "training"
            and not (directory / "result.json").exists()
        ):
            checkpoint = directory / "checkpoint"
            if checkpoint.exists():
                shutil.rmtree(checkpoint)
                atomic_json(
                    directory / "partial_checkpoint_cleanup.json",
                    {
                        "reason": "Removed only this interrupted job's incomplete checkpoint; metrics/error/status preserved",
                    },
                )
            staging = directory / ".checkpoint.staging"
            if staging.exists():
                shutil.rmtree(staging)
        atomic_json(
            directory / "status.json",
            {
                **status,
                "state": reason,
                "finished_unix": time.time(),
                "wall_seconds": time.monotonic() - start,
                "peak_group_rss_kib_sampled": peak_rss,
                "returncode": process.returncode if process else None,
                "peak_artifact_bytes_sampled": peak_bytes,
                "artifact_bytes": sum(
                    p.stat().st_size for p in directory.rglob("*") if p.is_file()
                ),
            },
        )


if __name__ == "__main__":
    # fd remains open for the full worker lifetime, including exception cleanup.
    lease_fd = int(sys.argv[2])
    try:
        supervise(Path(sys.argv[1]), sys.argv[3])
    finally:
        os.close(lease_fd)
