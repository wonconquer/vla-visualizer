"""Validate canonical inference, extended capture, instructions and samples."""

from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path
import platform
import resource
from statistics import median
import sys

import torch

from vla_visualizer.config import ROOT, cache_dir, sample_path
from vla_visualizer.contracts import CaptureOptions, LoadOptions
from vla_visualizer.execution.jobs import atomic_json, model_lease
from vla_visualizer.inference.observations import read_observation
from vla_visualizer.inference.replay import make_run, select_records


def main():
    from vla_visualizer.adapters.smolvla import SmolVLAAdapter
    from vla_visualizer.data.so100 import observation, target_chunk
    from vla_visualizer.execution.training import training_batch
    from vla_visualizer.ui.pipeline_evidence import enrich
    from vla_visualizer.ui.execution_flow import call_events

    output = ROOT / "outputs/phase5_integrated/actual"
    output.mkdir(parents=True, exist_ok=True)
    adapter = SmolVLAAdapter()
    canonical = read_observation(sample_path())
    reference = torch.tensor(
        json.loads((ROOT / "docs/smolvla_matched_cpu_result.json").read_text())["actions"]
    )
    report = {
        "scope": "Actual CPU inference and capture validation; separate from training, simulation and browser acceptance.",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "hardware": platform.processor(),
        "gpu_used": False,
        "checks": [],
    }
    with model_lease():
        try:
            adapter.load(LoadOptions(device="cpu", seed=0, cache_dir=str(cache_dir())))
            report["load_seconds"] = adapter._load_seconds
            adapter.run_inference(adapter.preprocess(canonical), CaptureOptions())
            timings = []
            for repeat in range(3):
                results = []
                for options in (CaptureOptions(), CaptureOptions(enabled=True, extended=True)):
                    rng_before = torch.get_rng_state().clone()
                    profiler_before = sys.getprofile()
                    result = adapter.run_inference(adapter.preprocess(canonical), options)
                    assert torch.equal(torch.get_rng_state(), rng_before)
                    assert sys.getprofile() is profiler_before
                    torch.testing.assert_close(
                        torch.tensor(result.actions), reference, rtol=0, atol=1e-6
                    )
                    results.append(result)
                plain, full = results
                delta = float(
                    (torch.tensor(plain.actions) - torch.tensor(full.actions)).abs().max()
                )
                assert delta == 0
                timings.append(
                    {
                        "plain_seconds": plain.metadata["inference_seconds"],
                        "extended_seconds": full.metadata["inference_seconds"],
                    }
                )
                ext = full.intermediates["extended_capture"]
                assert len(ext["full_loop"]) == 10 and full.intermediates["hook_count_after"] == 0
                errors = []
                for k, event in enumerate(ext["full_loop"]):
                    x = torch.tensor(event["x_before"])
                    v = torch.tensor(event["velocity"])
                    after = torch.tensor(event["x_after"])
                    error = float((after - (x + event["dt"] * v)).abs().max())
                    assert error == 0
                    errors.append(error)
                    if k < 9:
                        assert event["x_after"] == ext["full_loop"][k + 1]["x_before"]
                report["checks"].append(
                    {
                        "repeat": repeat,
                        "canonical_max_abs_difference": float(
                            (torch.tensor(full.actions) - reference).abs().max()
                        ),
                        "capture_output_difference": delta,
                        "hook_count_after": 0,
                        "rng_unchanged": True,
                        "profile_restored": True,
                        "euler_max_abs_errors": errors,
                    }
                )
            replay = make_run(full, adapter.get_inspection_spec(), canonical)
            (output / "full_loop_replay.json").write_text(replay.snapshot)
            atomic_json(output / "full_loop_capture.json", asdict(full))
            data = replay.read()
            hover = enrich(data, {}, 4)
            for key, grid in [("vision_top", [32, 32]), ("connector_wrist", [8, 8])]:
                assert hover[key]["grid"] == grid
                assert len(hover[key]["features"]["values"]) == grid[0] * grid[1]
                assert len(hover[key]["features"]["values"][0]) == 16
            report["timings"] = timings
            report["plain_median_seconds"] = median(t["plain_seconds"] for t in timings)
            report["extended_median_seconds"] = median(t["extended_seconds"] for t in timings)
            report["overhead_seconds"] = (
                report["extended_median_seconds"] - report["plain_median_seconds"]
            )
            report["replay_bytes"] = len(replay.snapshot.encode())
            report["call_events"] = call_events(data)
            report["function_timeline"] = ext["function_timeline"]
            report["prefix_processing"] = ext["prefix_processing"]
            report["instructions"] = []
            for sentence in ["Place the cube into the box.", "Move the cube into the box."]:
                selected = replace(canonical, instruction=sentence)
                actual = adapter.run_inference(
                    adapter.preprocess(selected), CaptureOptions(enabled=True)
                )
                captured = asdict(actual)
                tokens = select_records(captured, ["language_tokenization"])[0]
                embedding = select_records(captured, ["language"])[0]
                assert tokens["context"]["task"] == [sentence + "\n"]
                assert (
                    tokens["outputs"]["token_ids"]["values"]
                    == embedding["inputs"]["tensor"]["values"]
                )
                result_replay = make_run(actual, adapter.get_inspection_spec(), selected)
                assert result_replay.read()["manifest"]["instruction"] == sentence
                report["instructions"].append(
                    {
                        "selected": sentence,
                        "native_task": tokens["context"]["task"],
                        "processor_model_token_ids_equal": True,
                        "replay_sentence_equal": True,
                        "shape": actual.metadata["native_action_shape"],
                        "task_success": None,
                    }
                )
            frame = observation(0, 120)
            actual = adapter.run_inference(adapter.preprocess(frame), CaptureOptions(enabled=True))
            (output / "frame120_replay.json").write_text(
                make_run(actual, adapter.get_inspection_spec(), frame).snapshot
            )
            for camera, image in frame.images.items():
                image.save(output / ("frame120_" + camera.rsplit(".", 1)[-1] + ".png"))
            targets, mask = target_chunk(0, 453)
            assert int((~mask).sum()) == 1 and bool(mask[1:].all())
            batch, _ = training_batch(
                adapter, {"observation_mode": "dataset", "episode_index": 0, "frame_index": 453}
            )
            assert torch.equal(batch["action_is_pad"].cpu()[0], mask)
            report["dataset"] = {
                "prepared_episodes": [0],
                "tested_frame": 120,
                "timestamp": frame.provenance["timestamp_seconds"],
                "tail_frame": 453,
                "tail_valid_targets": 1,
                "tail_padding_targets": 49,
                "native_normalization_verified": True,
                "two_episode_requirement": "BLOCKED; episode 1 download failed",
            }
            import lerobot.policies.smolvla.modeling_smolvla as smol
            from lerobot.policies.common.flow_matching import euler_integrate
            from transformers.models.idefics3.modeling_idefics3 import (
                Idefics3Connector,
                Idefics3VisionEmbeddings,
            )

            functions = [
                smol.VLAFlowMatching.embed_prefix,
                smol.VLAFlowMatching.embed_suffix,
                smol.VLAFlowMatching.sample_actions,
                smol.VLAFlowMatching.denoise_step,
                smol.VLAFlowMatching.forward,
                smol.SmolVLAPolicy.forward,
                euler_integrate,
                Idefics3Connector.pixel_shuffle,
                Idefics3VisionEmbeddings.forward,
            ]
            report["native_sources"] = [
                {
                    "function": f.__qualname__,
                    "file": inspect.getsourcefile(f),
                    "line": inspect.getsourcelines(f)[1],
                    "sha256": hashlib.sha256(
                        Path(inspect.getsourcefile(f)).read_bytes()
                    ).hexdigest(),
                }
                for f in functions
            ]
            report["peak_process_rss_bytes"] = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            )
            report["status"] = "PASS_REAL_INFERENCE_ONLY"
            atomic_json(output / "validation.json", report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
        finally:
            adapter.unload()


if __name__ == "__main__":
    main()
