"""Check hook/profiler cleanup and cancellation of an actual training worker."""

import json
import sys
import time
import uuid

import torch

from vla_visualizer.config import ROOT, cache_dir, sample_path
from vla_visualizer.contracts import CaptureOptions, LoadOptions
from vla_visualizer.execution.jobs import JobManager, JobConfig, TERMINAL, atomic_json, model_lease


def main():
    from vla_visualizer.adapters.smolvla import SmolVLAAdapter
    from vla_visualizer.execution.training import training_batch
    from vla_visualizer.execution.training_capture import TrainingCapture
    from vla_visualizer.inference.observations import read_observation

    adapter = SmolVLAAdapter()
    output = ROOT / "outputs/phase5_integrated/cleanup.json"
    report = {}

    def fail(*args, **kwargs):
        raise RuntimeError("Injected head failure in the actual model forward path")

    with model_lease():
        try:
            adapter.load(LoadOptions(device="cpu", seed=0, cache_dir=str(cache_dir())))
            observation = read_observation(sample_path())
            policy = adapter._policy
            head = policy.model.action_out_proj
            handle = head.register_forward_hook(fail)
            try:
                try:
                    adapter.run_inference(
                        adapter.preprocess(observation), CaptureOptions(enabled=True, extended=True)
                    )
                    raise AssertionError("The expected exception was not raised.")
                except RuntimeError as exc:
                    assert "Injected head failure" in str(exc)
                assert sys.getprofile() is None
                assert sum(len(m._forward_hooks) for m in policy.modules()) == 1
                policy.requires_grad_(False)
                head.requires_grad_(True)
                policy.train()
                batch, _ = training_batch(adapter)
                try:
                    with TrainingCapture(policy):
                        policy(dict(batch))
                    raise AssertionError("The expected training exception was not raised.")
                except RuntimeError as exc:
                    assert "Injected head failure" in str(exc)
                assert sys.getprofile() is None
                assert sum(len(m._forward_hooks) for m in policy.modules()) == 1
            finally:
                handle.remove()
            policy.requires_grad_(False).eval()
            result = adapter.run_inference(
                adapter.preprocess(observation), CaptureOptions(enabled=True, extended=True)
            )
            reference = json.loads((ROOT / "docs/smolvla_matched_cpu_result.json").read_text())[
                "actions"
            ]
            torch.testing.assert_close(
                torch.tensor(result.actions), torch.tensor(reference), rtol=0, atol=1e-6
            )
            report["actual_exception_cleanup"] = {
                "status": "PASS",
                "scope": "Injected head failure during actual inference and training forwards",
                "forward_hooks_after": 0,
                "profile_restored": True,
                "subsequent_canonical_unchanged": True,
            }
        finally:
            adapter.unload()
    manager = JobManager()
    owner = uuid.uuid4().hex
    job = manager.submit(JobConfig(kind="training", training_steps=2), owner)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = manager.status(job, owner)
        if status["state"] == "running":
            break
        time.sleep(0.1)
    manager.cancel(job, owner)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = manager.status(job, owner)
        if status["state"] in TERMINAL:
            break
        time.sleep(0.2)
    assert status["state"] == "cancelled", status
    directory = manager.directory(job)
    assert (
        not (directory / "checkpoint").exists() and not (directory / ".checkpoint.staging").exists()
    )
    report["actual_background_cancellation"] = {
        "status": "PASS",
        "job_run_id": job,
        "final_state": status["state"],
        "scope": "Cancellation shortly after worker start; not a mid-optimizer cancellation test",
        "checkpoint_or_staging_retained": False,
        "diagnostic_log_retained": (directory / "worker.log").exists(),
    }
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
