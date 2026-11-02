"""Bounded inspection and same-seed action regression using cached artifacts."""

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import platform
import resource
from statistics import median

import torch

from vla_visualizer.contracts import CaptureOptions, LoadOptions
from vla_visualizer.inference.observations import read_observation, observation_fingerprint
from vla_visualizer.inference.runner import InferenceRunner


def rss_bytes() -> int:
    """Linux current resident set, distinct from cumulative process high-water mark."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux VmRSS is unavailable.")


def contains_tensor(value) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return any(contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(contains_tensor(item) for item in value)
    return False


def inspect_real_sample(options: LoadOptions, repeats: int = 3) -> dict:
    """Alternate baseline/capture on the existing runner; reject any output change."""
    if not 2 <= repeats <= 5:
        raise ValueError("Use 2–5 repetitions for a bounded regression check.")
    root = Path(__file__).resolve().parents[1]
    observation = read_observation(root / "assets/sample_inputs/so100_observation.json")
    runner = InferenceRunner()
    timings, rss, peaks, checks = [], [], [], []
    try:
        runner.load("smolvla", options)
        warm = runner.run_inference("smolvla", observation)
        for repeat in range(repeats):
            pair = []
            for capture in (CaptureOptions(), CaptureOptions(enabled=True)):
                before = rss_bytes()
                result = runner.run_inference("smolvla", observation, capture)
                rss.append({"repeat": repeat, "capture": capture.enabled,
                            "before_bytes": before, "after_bytes": rss_bytes()})
                peaks.append(result.metadata["process_peak_rss_bytes"])
                pair.append(result)
            baseline, inspected = pair
            actual = torch.tensor(inspected.actions)
            reference = torch.tensor(baseline.actions)
            difference = (actual - reference).abs().max().item()
            torch.testing.assert_close(actual, reference, rtol=0, atol=1e-6)
            torch.testing.assert_close(reference, torch.tensor(warm.actions), rtol=0, atol=1e-6)
            captures = inspected.intermediates
            if contains_tensor(captures) or captures["hook_count_after"] != captures["hook_count_before"]:
                raise RuntimeError("Capture retained tensors or hooks.")
            if not torch.isfinite(actual).all() or inspected.metadata["action_chunk_shape"] != [1, 50, 6]:
                raise RuntimeError("Action regression failed.")
            if not all(inspected.metadata[key] for key in (
                "state_normalization_verified", "action_postprocessing_verified", "strict_loading",
                "frozen", "eval_mode",
            )):
                raise RuntimeError("Verified inference safeguards did not pass.")
            checks.append({"repeat": repeat, "max_action_absolute_difference": difference,
                           "hook_count_before": captures["hook_count_before"],
                           "hook_count_after": captures["hook_count_after"],
                           "record_count": len(captures["stages"])})
            timings.append({"baseline_seconds": baseline.metadata["inference_seconds"],
                            "inspection_seconds": inspected.metadata["inference_seconds"]})
        payload = asdict(inspected)
        plain_captures = json.dumps(payload["intermediates"], allow_nan=False)
        base_median = median(item["baseline_seconds"] for item in timings)
        capture_median = median(item["inspection_seconds"] for item in timings)
        payload["validation"] = {
            "checks": checks, "timings": timings,
            "baseline_median_seconds": base_median, "inspection_median_seconds": capture_median,
            "observed_overhead_seconds": capture_median - base_median,
            "observed_overhead_percent": (capture_median / base_median - 1) * 100,
            "capture_json_bytes": len(plain_captures.encode()),
            "rss_samples": rss, "process_high_water_samples_bytes": peaks,
            "memory_limit": "VmRSS is process resident memory; high water is cumulative and includes loading; neither isolates temporary capture allocations",
            "retained_tensors": False,
            "python": platform.python_version(), "processor": platform.processor(),
            "torch_build": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "observation_sha256": observation_fingerprint(observation),
        }
        return payload
    finally:
        runner.unload()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/smolvla_inspection.json"))
    args = parser.parse_args()
    payload = inspect_real_sample(LoadOptions(
        device=args.device, cpu_threads=args.threads, seed=args.seed, local_files_only=True,
    ), repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.output), "validation": payload["validation"]}, indent=2))


if __name__ == "__main__":
    main()
