"""Independent LIBERO policy; never imports SO100 statistics or action semantics."""

from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from vla_visualizer.config import cache_dir
from vla_visualizer.simulation.compatibility import MODEL, REVISION, CAMERA_MAP, STATE_NAMES


class LiberoPolicy:
    def __init__(self, device="cpu", seed=0):
        self.device, self.seed = device, seed
        self.policy = None
        self.request_index = 0

    def load(self):
        import torch
        from vla_visualizer.adapters.smolvla import (
            BACKBONE, BACKBONE_FILES, BACKBONE_REVISION, CHECKPOINT_FILES, SmolVLAAdapter, file_sha256,
        )
        from vla_visualizer.contracts import LoadOptions
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from safetensors.torch import load_file

        tick = perf_counter()
        options = LoadOptions(device=self.device, cache_dir=str(cache_dir()))
        source = Path(SmolVLAAdapter._snapshot(MODEL, REVISION, (*CHECKPOINT_FILES, "train_config.json"), options))
        config = SmolVLAConfig.from_pretrained(source)
        if tuple(config.image_features) != tuple(dst for _, dst in CAMERA_MAP) or config.input_features["observation.state"].shape != (8,) or config.action_feature.shape != (7,):
            raise ValueError("LIBERO checkpoint feature schema is incompatible with the native processor")
        training = json.loads((source / "train_config.json").read_text())
        if training["dataset"]["repo_id"] != "lerobot/libero" or training["env"]["control_mode"] != "relative":
            raise ValueError("Checkpoint training domain/control mode mismatch")
        self.stats = {}
        for processor_file in ("policy_preprocessor.json", "policy_postprocessor.json"):
            processor = json.loads((source / processor_file).read_text())
            step = next(s for s in processor["steps"] if "normalizer" in s["registry_name"])
            tensors = load_file(str(source / step["state_file"]))
            for feature, width in (("action", 7), ("observation.state", 8)):
                if processor_file.startswith("policy_post") and feature != "action":
                    continue
                for stat in ("mean", "std"):
                    tensor = tensors[f"{feature}.{stat}"]
                    if tensor.shape != (width,) or not tensor.isfinite().all() or (stat == "std" and not (tensor > 0).all()):
                        raise ValueError("Checkpoint normalization statistics failed validation")
                    if feature in self.stats and stat in self.stats[feature]:
                        if self.stats[feature][stat] != tensor.tolist():
                            raise ValueError("Pre/post action statistics differ.")
                    self.stats.setdefault(feature, {})[stat] = tensor.tolist()
        backbone = SmolVLAAdapter._snapshot(BACKBONE, BACKBONE_REVISION, BACKBONE_FILES, options)
        config.device, config.load_vlm_weights, config.vlm_model_name = self.device, False, backbone
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        torch.set_num_threads(8)
        self.policy = SmolVLAPolicy.from_pretrained(source, config=config, strict=True).float().requires_grad_(False).eval()
        self.policy.reset()
        self.pre, self.post = make_pre_post_processors(config, str(source), preprocessor_overrides={
            "device_processor": {"device": self.device}, "tokenizer_processor": {"tokenizer_name": backbone},
        })
        self.env_pre, self.env_post = make_env_pre_post_processors(LiberoEnv(), config)
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.metadata = {
            "model": MODEL, "revision": REVISION, "artifact_sha256": {n: file_sha256(source / n) for n in (*CHECKPOINT_FILES, "train_config.json")},
            "load_seconds": perf_counter() - tick, "state_names": STATE_NAMES,
            "normalization_source": "Native processor statistics from the immutable checkpoint",
            "training_dataset": training["dataset"]["repo_id"],
            "training_dataset_revision_recorded": training["dataset"].get("revision"),
            "camera_order": CAMERA_MAP, "statistics": self.stats,
        }

    def prepare(self, raw, instruction):
        from lerobot.envs.utils import preprocess_observation
        processed = self.env_pre(preprocess_observation(batch_observation(raw)))
        raw_state = processed["observation.state"].clone()
        # Retain actual post-env-processor camera pixels, before model resize/pad.
        images = {key: tensor[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
                  for key, tensor in processed.items() if key.startswith("observation.images.")}
        processed["task"] = [instruction]
        prepared = self.pre(processed)
        import torch
        expected = (raw_state.to(self.device) - torch.tensor(self.stats["observation.state"]["mean"], device=self.device)) / (torch.tensor(self.stats["observation.state"]["std"], device=self.device) + 1e-8)
        torch.testing.assert_close(prepared["observation.state"], expected)
        return prepared, raw_state[0].tolist(), images

    def predict(self, prepared, raw_state, capture=False):
        import torch
        from vla_visualizer.contracts import CaptureOptions
        from vla_visualizer.adapters.smolvla_inspection import SmolVLAInspection

        policy = self.policy
        # Source-reviewed modules are checked against the actual loaded model.
        # Missing/bypassed points fail rather than inventing hidden values.
        inspection = SmolVLAInspection(
            policy, CaptureOptions(enabled=True), CAMERA_MAP,
            state_description="LIBERO 8D eef position / axis-angle / gripper qpos native normalization",
            action_description="LIBERO relative OSC_POSE 7D native inverse normalization",
        ) if capture else None
        hook_before = sum(len(m._forward_hooks) for m in policy.modules())
        if self.device == "cuda":
            torch.cuda.synchronize()
        tick = perf_counter()
        with torch.inference_mode():
            noise = torch.randn((1, policy.config.chunk_size, policy.config.max_action_dim), generator=self.generator, device=self.device)
            if inspection:
                inspection.capture_inputs(prepared, raw_state, noise)
            with inspection.hooks if inspection else nullcontext():
                normalized = policy.predict_action_chunk(dict(prepared), noise=noise)
            actions = self.post(normalized)
            expected = normalized.cpu() * torch.tensor(self.stats["action"]["std"]) + torch.tensor(self.stats["action"]["mean"])
            torch.testing.assert_close(actions.cpu(), expected, rtol=1e-5, atol=1e-6)
            if actions.shape != (1, 50, 7) or not actions.isfinite().all():
                raise RuntimeError("Invalid LIBERO 50-by-7 action chunk.")
            captured = inspection.finish(normalized, actions) if inspection else {}
        if self.device == "cuda":
            torch.cuda.synchronize()
        if sum(len(m._forward_hooks) for m in policy.modules()) != hook_before:
            raise RuntimeError("Capture hook cleanup failed")
        self.request_index += 1
        return actions[0].cpu(), asdict_capture(captured), perf_counter() - tick

    def action(self, row):
        return self.env_post({"action": row.unsqueeze(0)})["action"][0].cpu().numpy()

    def close(self):
        import gc
        self.policy = None
        gc.collect()


def asdict_capture(value):
    from dataclasses import is_dataclass
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: asdict_capture(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [asdict_capture(v) for v in value]
    return value


def batch_observation(value):
    """Native env processor expects batched nested state as in official vector eval."""
    import numpy as np
    if isinstance(value, dict):
        return {key: batch_observation(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return np.expand_dims(value, axis=0)
    return value
