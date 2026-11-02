"""Tiny real SO100 supervised experiment. Original checkpoint remains immutable."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import shutil
from time import perf_counter

from vla_visualizer.config import cache_dir, sample_path, training_data
from vla_visualizer.execution.jobs import atomic_json
from vla_visualizer.execution.worker import check_cancel, progress

PARQUET_SHA256 = "35d98dcc0bf846749cf1b674011c0fcd4215510a197f7f9f3b3d1a08fec788e9"


def frozen_digest(policy):
    digest = hashlib.sha256()
    for name, param in policy.named_parameters():
        if name not in {"model.action_out_proj.weight", "model.action_out_proj.bias"}:
            digest.update(name.encode())
            digest.update(param.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def availability() -> dict:
    path = training_data()
    valid = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == PARQUET_SHA256
    return {
        "data_available": valid,
        "scope": "SO100 episode 0 / frame 0, 50 recorded targets; at most 3 output-projection optimizer steps",
        "blocker": None
        if valid
        else "SO100 episode-0 Parquet with the required checksum is missing.",
        "improved_performance_verified": False,
    }


def load_base(config):
    from vla_visualizer.adapters.smolvla import SmolVLAAdapter
    from vla_visualizer.contracts import LoadOptions

    adapter = SmolVLAAdapter()
    adapter.load(
        LoadOptions(device=config["device"], seed=config["seed"], cache_dir=str(cache_dir()))
    )
    return adapter


def training_batch(adapter, config=None):
    config = config or {}
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from vla_visualizer.inference.observations import read_observation

    if config.get("observation_mode") != "dataset" and not availability()["data_available"]:
        raise RuntimeError(availability()["blocker"])
    rows = (
        pq.read_table(training_data()).to_pylist()
        if config.get("observation_mode") != "dataset"
        else None
    )
    observation = read_observation(sample_path())
    if config.get("observation_mode") == "dataset":
        from vla_visualizer.data.so100 import observation as dataset_observation, target_chunk

        observation = dataset_observation(config["episode_index"], config["frame_index"])
        targets, mask = target_chunk(config["episode_index"], config["frame_index"])
    else:
        targets = torch.tensor([row["action"] for row in rows[:50]], dtype=torch.float32)
        mask = torch.zeros(50, dtype=torch.bool)
    if config.get("observation_mode") != "dataset":
        # Keep this smoke dataset tied to the verified, synchronized canonical images.
        from vla_visualizer.config import ROOT

        canonical = json.loads((ROOT / "assets/sample_inputs/so100_observation.json").read_text())
        manifest = json.loads(sample_path().read_text())
        if observation.instruction != canonical["instruction"]:
            raise ValueError("Training smoke instruction differs from the canonical task.")
        for key, name in manifest["images"].items():
            if (
                hashlib.sha256((sample_path().parent / name).read_bytes()).hexdigest()
                != canonical["provenance"]["image_sha256_by_camera"][key]
            ):
                raise ValueError("Training camera observation differs from verified frame 0.")
        adapter.validate_observation(observation)
        if len(rows) < 50 or any(
            row["episode_index"] != 0 or row["frame_index"] != i or row["task_index"] != 0
            for i, row in enumerate(rows[:50])
        ):
            raise ValueError("50 consecutive episode-0 targets are required.")
    adapter.validate_observation(observation)
    if config.get("observation_mode") != "dataset" and not np.array_equal(
        np.asarray(rows[0]["observation.state"], dtype=np.float32),
        np.asarray(observation.state, dtype=np.float32),
    ):
        raise ValueError("Observation images and target state do not match.")
    raw = {"observation.state": torch.tensor(observation.state), "task": observation.instruction}
    for key, image in observation.images.items():
        raw[key] = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
    # Explicit batch avoids the native batch processor interpreting T as batch.
    raw = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v] for k, v in raw.items()}
    raw["action"] = targets.unsqueeze(0)
    raw["action_is_pad"] = mask.unsqueeze(0)
    batch = adapter._preprocessor(raw)
    stats = adapter._normalization["processors"]["preprocessor"]["available_tensors"]
    expected = (
        targets.to(adapter._options.device)
        - torch.tensor(stats["action.mean"]["values"], device=adapter._options.device)
    ) / (torch.tensor(stats["action.std"]["values"], device=adapter._options.device) + 1e-8)
    torch.testing.assert_close(batch["action"][0], expected)
    assert batch["action"].shape == (1, 50, 6) and (~batch["action_is_pad"]).any()
    assert torch.isfinite(batch["action"]).all()
    return batch, observation


def train(directory: Path, config: dict) -> None:
    import torch
    from vla_visualizer.execution.checkpoints import save_delta, load_trained_checkpoint
    from vla_visualizer.adapters.smolvla import CHECKPOINT_FILES, file_sha256
    from vla_visualizer.adapters.smolvla_config import MVP_CHECKPOINT
    from vla_visualizer.contracts import CaptureOptions

    adapter = None
    start = perf_counter()
    try:
        progress(directory, "Loading immutable training base")
        adapter = load_base(config)
        source = Path(adapter._snapshot_path)
        hashes_before = {name: file_sha256(source / name) for name in CHECKPOINT_FILES}
        batch, observation = training_batch(adapter, config)
        policy = adapter._policy
        policy.requires_grad_(False)
        policy.model.action_out_proj.requires_grad_(True)
        params = {n: p for n, p in policy.named_parameters() if p.requires_grad}
        before = {n: p.detach().cpu().clone() for n, p in params.items()}
        count = sum(p.numel() for p in params.values())
        frozen_before = frozen_digest(policy)
        audit = {
            "base_checkpoint": MVP_CHECKPOINT.repo_id,
            "base_revision": MVP_CHECKPOINT.revision,
            "dataset": MVP_CHECKPOINT.dataset,
            "dataset_revision": MVP_CHECKPOINT.dataset_revision,
            "parquet_sha256": observation.provenance.get("parquet_sha256", PARQUET_SHA256),
            "observation_frame": config.get("frame_index", 0),
            "observation_episode": config.get("episode_index", 0),
            "recorded_instruction": observation.instruction,
            "target_frame_range_inclusive": [
                config.get("frame_index", 0),
                config.get("frame_index", 0) + int((~batch["action_is_pad"]).sum()) - 1,
            ],
            "target_shape": [1, 50, 6],
            "target_kind": "Actual recorded actions; predictions are not used as targets",
            "padding_mask": batch["action_is_pad"].detach().cpu().tolist(),
            "normalization": "Verified native mean/std from the immutable SO100 checkpoint",
            "trainable_modules": list(params),
            "trainable_parameters": count,
            "optimizer": "AdamW, lr=1e-5, weight_decay=0",
            "batch_size": 1,
            "max_steps": config["training_steps"],
            "planning_memory_gib": 8,
            "output": "checkpoint/",
            "evaluation_scope": "Same-observation inference smoke; not a performance evaluation",
        }
        atomic_json(directory / "training_audit.json", audit)
        optimizer = torch.optim.AdamW(params.values(), lr=1e-5, weight_decay=0)
        torch.manual_seed(config["seed"])
        policy.train()
        metrics = []
        from vla_visualizer.execution.training_capture import TrainingCapture, linear_probe

        capture = TrainingCapture(policy)
        training_events = []
        objective = {
            "name": "native conditional flow matching velocity MSE",
            "x_t": "t * noise + (1-t) * normalized_recorded_actions",
            "target_velocity": "noise - normalized_recorded_actions",
            "loss": "MSE(target_velocity, predicted_velocity), valid time/action dimensions only",
            "time_sampling": "Beta(1.5,1.0) * 0.999 + 0.001 (native source)",
            "noise_and_time_values": "actual time and bounded noise/target summaries in each step native_objective",
            "normalized_recorded_targets": batch["action"].detach().cpu().tolist(),
            "valid_target_count": int((~batch["action_is_pad"]).sum().item()),
            "source": "LeRobot 0.6.1 VLAFlowMatching.forward / SmolVLAPolicy.forward",
        }
        if config["device"] == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step in range(config["training_steps"]):
            check_cancel(directory)
            tick = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            capture.begin_step()
            with capture:
                loss, details = policy(dict(batch))
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss.")
                loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params.values()):
                raise RuntimeError("Measured gradient validation failed")
            grad = torch.nn.utils.clip_grad_norm_(list(params.values()), 10.0)
            if not torch.isfinite(grad) or grad <= 0:
                raise RuntimeError("No finite nonzero gradient update.")
            step_before = {name: param.detach().cpu().clone() for name, param in params.items()}
            probe_before = linear_probe(policy.model.action_out_proj, capture.fixed_hidden)
            optimizer.step()
            updates = {
                name: {
                    "max_abs": (param.detach().cpu() - step_before[name]).abs().max().item(),
                    "l2": torch.linalg.vector_norm(param.detach().cpu() - step_before[name]).item(),
                }
                for name, param in params.items()
            }
            training_events.append(
                {
                    "optimizer_step": step + 1,
                    "forward": capture.event,
                    "parameter_update": updates,
                    "fixed_head_probe": capture.probe_record(
                        policy.model.action_out_proj, probe_before
                    ),
                }
            )
            row = {
                "optimizer_step": step + 1,
                "loss": loss.item(),
                "gradient_norm": grad.item(),
                "seconds": perf_counter() - tick,
                "native_metrics": details,
            }
            metrics.append(row)
            atomic_json(directory / "metrics.json", {"records": metrics})
            atomic_json(
                directory / "training_visualization.json",
                {
                    "schema_version": 1,
                    "run_id": directory.name,
                    "objective": objective,
                    "trainable_modules": list(params),
                    "trainable_parameters": count,
                    "frozen_scope": "all modules except model.action_out_proj",
                    "steps": training_events,
                    "summary_only": True,
                    "hook_count_after_step": sum(len(m._forward_hooks) for m in policy.modules()),
                },
            )
            progress(
                directory,
                "Executing optimizer update",
                optimizer_step=step + 1,
                max_steps=config["training_steps"],
                loss=loss.item(),
            )
        changed = {n: float((p.detach().cpu() - before[n]).abs().max()) for n, p in params.items()}
        if not any(v > 0 for v in changed.values()):
            raise RuntimeError("Trainable weights did not change.")
        frozen_after = frozen_digest(policy)
        if frozen_after != frozen_before:
            raise RuntimeError("Frozen parameters changed.")
        policy.requires_grad_(False).eval()
        checkpoint = directory / ".checkpoint.staging"
        progress(directory, "Saving delta checkpoint without copying base weights")
        checkpoint_manifest = save_delta(
            checkpoint, policy, hashes_before, {**audit, "job_config": config}
        )
        trained = adapter.run_inference(adapter.preprocess(observation), CaptureOptions())
        trained_actions = torch.tensor(trained.actions)
        adapter.unload()
        del optimizer, params, policy, batch
        progress(directory, "Reloading base and strictly validating delta reconstruction")
        adapter = load_base(config)
        load_trained_checkpoint(adapter, checkpoint)
        reloaded = adapter.run_inference(adapter.preprocess(observation), CaptureOptions())
        torch.testing.assert_close(
            torch.tensor(reloaded.actions), trained_actions, rtol=0, atol=1e-6
        )
        preserved = all(
            file_sha256(source / name) == digest for name, digest in hashes_before.items()
        )
        if not preserved:
            raise RuntimeError("Base checkpoint immutability check failed")
        checkpoint.rename(directory / "checkpoint")
        atomic_json(
            directory / "result.json",
            {
                "run_id": directory.name,
                "training_executed": True,
                "metrics": metrics,
                "parameter_max_abs_update": changed,
                "checkpoint_reload_verified": True,
                "base_preserved": preserved,
                "base_hashes": hashes_before,
                "frozen_parameters_preserved": True,
                "frozen_sha256_before": frozen_before,
                "frozen_sha256_after": frozen_after,
                "saved_weight_sha256": checkpoint_manifest["delta_sha256"],
                "checkpoint_kind": "delta",
                "standalone": False,
                "hash_scope": "trained output projection only; immutable base pinned separately",
                "reload_max_abs_action_difference": (
                    torch.tensor(reloaded.actions) - trained_actions
                )
                .abs()
                .max()
                .item(),
                "evaluation_actions": reloaded.actions,
                "improved_performance_verified": False,
                "training_wall_seconds": perf_counter() - start,
                "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                if config["device"] == "cuda"
                else None,
                "load_seconds": adapter._load_seconds,
            },
        )
        progress(
            directory,
            "Training and checkpoint reconstruction verified",
            optimizer_step=len(metrics),
        )
    finally:
        if adapter is not None:
            adapter.unload()
        if not (directory / "result.json").exists() and (directory / "checkpoint").exists():
            shutil.rmtree(directory / "checkpoint")
        if (directory / ".checkpoint.staging").exists():
            shutil.rmtree(directory / ".checkpoint.staging")


def evaluate(directory: Path, config: dict) -> None:
    from vla_visualizer.execution.checkpoints import load_trained_checkpoint
    from vla_visualizer.contracts import CaptureOptions
    from vla_visualizer.inference.observations import read_observation

    checkpoint = Path(config["checkpoint_directory"])
    report = json.loads((checkpoint.parent / "result.json").read_text())
    adapter = None
    try:
        progress(directory, "Loading trained checkpoint for evaluation")
        adapter = load_base(config)
        loaded = load_trained_checkpoint(adapter, checkpoint)
        if loaded["payload_sha256"] != report["saved_weight_sha256"]:
            raise ValueError("Trained checkpoint checksum changed.")
        if config.get("checkpoint_sha256", loaded["payload_sha256"]) != loaded["payload_sha256"]:
            raise ValueError("Checkpoint identity changed after evaluation admission.")
        source_config = json.loads((checkpoint.parent / "config.json").read_text())
        if source_config.get("observation_mode") == "dataset":
            from vla_visualizer.data.so100 import observation as dataset_observation

            observation = dataset_observation(
                source_config["episode_index"], source_config["frame_index"]
            )
        else:
            observation = read_observation(sample_path())
        result = adapter.run_inference(
            adapter.preprocess(observation), CaptureOptions(enabled=True, extended=True)
        )
        payload = asdict(result)
        payload["metadata"]["base_checkpoint"] = payload["metadata"]["checkpoint"]
        payload["metadata"]["base_revision"] = payload["metadata"]["revision"]
        payload["metadata"]["checkpoint"] = f"local training job {config['checkpoint_job']}"
        payload["metadata"]["revision"] = report["saved_weight_sha256"]
        payload["metadata"].update(
            {
                "trained_checkpoint_job": config["checkpoint_job"],
                "checkpoint_kind": loaded["kind"],
                "weight_sha256": report["saved_weight_sha256"],
                "evaluation_job_run_id": directory.name,
                "parent_training_run_id": config["checkpoint_job"],
                "immutable_base_checkpoint": report.get(
                    "base_checkpoint", payload["metadata"]["base_checkpoint"]
                ),
                "immutable_base_revision": payload["metadata"]["base_revision"],
                "scope": "SO100 fixed-observation inference; simulator/task performance not evaluated",
            }
        )
        comparison = {
            "conditions_match": source_config["seed"] == config["seed"]
            and source_config["device"] == config["device"],
            "rtol": 0,
            "atol": 1e-6,
            "max_absolute_difference": None,
        }
        if comparison["conditions_match"] and report.get("evaluation_actions"):
            import torch

            expected = torch.tensor(report["evaluation_actions"])
            actual = torch.tensor(result.actions)
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
            comparison["max_absolute_difference"] = float((actual - expected).abs().max())
        payload["metadata"]["post_training_probe_comparison"] = comparison
        atomic_json(directory / "result.json", payload)
        from vla_visualizer.inference.replay import make_run

        spec = dict(adapter.get_inspection_spec())
        spec.update(
            checkpoint=payload["metadata"]["checkpoint"], revision=payload["metadata"]["revision"]
        )
        replay = make_run(
            payload,
            spec,
            observation,
            origin="fresh-worker reconstruction of trained checkpoint; saved playback",
        )
        (directory / "replay.json").write_text(replay.snapshot)
        atomic_json(
            directory / "links.json",
            {
                "job_run_id": directory.name,
                "capture_run_id": replay.run_id,
                "parent_training_run_id": config["checkpoint_job"],
                "checkpoint_sha256": report["saved_weight_sha256"],
                "immutable_base_checkpoint": payload["metadata"]["base_checkpoint"],
                "immutable_base_revision": payload["metadata"]["base_revision"],
                "replay": "replay.json",
                "result": "result.json",
            },
        )
        progress(directory, "Trained checkpoint inference completed", inference_requests=1)
    finally:
        if adapter is not None:
            adapter.unload()
