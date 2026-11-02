"""Run one bounded model job from the project root."""

import argparse
import json
from pathlib import Path
import time
import uuid

from vla_visualizer.execution.jobs import JobConfig, JobManager, TERMINAL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["baseline", "simulation", "training", "evaluate"])
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--training-steps", type=int, default=2)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-job")
    parser.add_argument("--owner", help="Session capability for evaluating your completed training checkpoint")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--instruction", help="Supported inference instruction; training retains the dataset annotation")
    parser.add_argument("--observation-mode", choices=["canonical", "dataset"], default="canonical")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    manager = JobManager()
    owner = args.owner or uuid.uuid4().hex
    config = JobConfig(
        kind=args.kind,
        max_seconds=args.seconds,
        max_steps=args.steps,
        action_steps=args.action_steps,
        training_steps=args.training_steps,
        seed=args.seed,
        device=args.device,
        checkpoint_job=args.checkpoint_job,
        instruction=args.instruction,
        observation_mode=args.observation_mode,
        episode_index=args.episode,
        frame_index=args.frame,
    )
    job_id = manager.submit(config, owner)
    receipt = {"run_id": job_id, "owner": owner, "directory": str(manager.directory(job_id))}
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    last = None
    try:
        while True:
            status = manager.status(job_id, owner)
            if status != last:
                print(json.dumps(status, ensure_ascii=False), flush=True)
                last = status
            if status["state"] in TERMINAL:
                return 0 if status["state"] == "completed" else 2
            time.sleep(1)
    except KeyboardInterrupt:
        manager.cancel(job_id, owner)
        print("Cancellation requested. The supervisor will clean up worker resources.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
