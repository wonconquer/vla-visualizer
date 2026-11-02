"""Opt-in real inference: python -m examples.smolvla_inference --device cpu."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import resource

from vla_visualizer.contracts import LoadOptions
from vla_visualizer.adapters.smolvla import (
    CHECKPOINT,
    CHECKPOINT_FILES,
    REVISION,
    SmolVLAAdapter,
    NormalizationCompatibilityError,
    inspect_checkpoint_normalization,
)
from vla_visualizer.inference.observations import observation_fingerprint, read_observation
from vla_visualizer.inference.runner import InferenceRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--observation", type=Path, default=root / "assets/sample_inputs/so100_observation.json"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--allow-download", action="store_true", help="Download approved checkpoint ~907 MB if uncached"
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=root / "outputs/smolvla_smoke.json")
    parser.add_argument(
        "--audit-normalization",
        action="store_true",
        help="Inspect cached processor statistics without constructing a policy",
    )
    args = parser.parse_args()
    if args.audit_normalization:
        path = SmolVLAAdapter._snapshot(
            CHECKPOINT,
            REVISION,
            tuple(name for name in CHECKPOINT_FILES if name != "model.safetensors"),
            LoadOptions(cache_dir=args.cache_dir, local_files_only=not args.allow_download),
        )
        report = inspect_checkpoint_normalization(Path(path))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Normalization verified: {report['verified']}; report: {args.output}")
        return
    observation = read_observation(args.observation)
    runner = InferenceRunner()
    try:
        runner.load(
            "smolvla",
            LoadOptions(
                device=args.device,
                cache_dir=args.cache_dir,
                local_files_only=not args.allow_download,
                cpu_threads=args.threads,
                seed=args.seed,
            ),
        )
        print("PASS: authenticated checkpoint loaded strictly in frozen/eval mode.", flush=True)
        result = runner.run_inference("smolvla", observation)
        payload = asdict(result)
        payload["metadata"].update(
            {
                "observation_sha256": observation_fingerprint(observation),
                "observation_manifest": str(args.observation),
                "python": platform.python_version(),
                "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "shape": result.metadata["action_shape"],
                    "first_action": result.actions[0],
                    "inference_seconds": result.metadata["inference_seconds"],
                    "saved_to": str(args.output),
                },
                indent=2,
            )
        )
    except NormalizationCompatibilityError as exc:
        parser.exit(2, f"{exc}\nNo inference performed; no output file written.\n")
    finally:
        runner.unload()


if __name__ == "__main__":
    main()
