"""Pinned SmolVLA integration with explicit normalization compatibility checks."""

from importlib.metadata import version
from contextlib import nullcontext
import gc
import math
import json
import hashlib
from pathlib import Path
import resource
from time import perf_counter

from vla_visualizer.adapters.base import BaseVLAAdapter
from vla_visualizer.adapters.smolvla_config import MVP_CHECKPOINT, SmolVLACheckpoint
from vla_visualizer.contracts import (
    CaptureOptions,
    InferenceResult,
    InputSpec,
    LoadOptions,
    ModelCapabilities,
    ModelStructure,
    Observation,
    PreparedInputs,
    Stage,
)

CHECKPOINT = MVP_CHECKPOINT.repo_id
REVISION = MVP_CHECKPOINT.revision
BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
BACKBONE_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
BACKBONE_FILES = (
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "generation_config.json",
    "merges.txt",
    "vocab.json",
)


class NormalizationCompatibilityError(RuntimeError):
    """Checkpoint statistics cannot be mapped without unverified assumptions."""


def file_sha256(path: Path) -> str:
    """Hash artifacts without retaining a full checkpoint in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_checkpoint_normalization(
    directory: Path, checkpoint: SmolVLACheckpoint = MVP_CHECKPOINT
) -> dict:
    """Audit the selected revision's native artifacts without loading model weights."""
    from safetensors.torch import load_file

    config = json.loads((directory / "config.json").read_text())
    report = {
        "checkpoint": checkpoint.repo_id,
        "revision": checkpoint.revision,
        "lerobot_version": version("lerobot"),
        "normalization_method": config["normalization_mapping"],
        "model_input_features": config.get("input_features", {}),
        "model_output_features": config.get("output_features", {}),
        "processors": {},
        "verified": False,
        "semantic_blockers": [],
    }
    for role, filename, features in (
        ("preprocessor", "policy_preprocessor.json", ("observation.state", "action")),
        ("postprocessor", "policy_postprocessor.json", ("action",)),
    ):
        processor = json.loads((directory / filename).read_text())
        step = next(s for s in processor["steps"] if "normalizer" in s["registry_name"])
        state_file = directory / step["state_file"]
        tensors = load_file(state_file, device="cpu")
        required = [f"{feature}.{stat}" for feature in features for stat in ("mean", "std")]
        invalid = []
        available = {}
        for key, tensor in tensors.items():
            finite = bool(tensor.isfinite().all())
            # Native files also contain constant metadata (task_index.std=0).
            # Positive std is required for the state/action features actually normalized.
            active_std = any(key.endswith(f"{feature}.std") for feature in features)
            positive = bool((tensor > 0).all()) if active_std else None
            available[key] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "values": tensor.tolist(),
                "finite": finite,
                "positive_std": positive,
            }
            if not finite or positive is False:
                invalid.append(key)
            # Suffix inspection is diagnostic only; it never selects a profile.
            for feature in features:
                if key.endswith(f"{feature}.mean") or key.endswith(f"{feature}.std"):
                    expected_shape = step["config"]["features"][feature]["shape"]
                    if list(tensor.shape) != expected_shape:
                        invalid.append(key)
        for feature in features:
            dimensions = step["config"]["features"][feature]["shape"]
            for stat in ("mean", "std"):
                key = f"{feature}.{stat}"
                if key in tensors and list(tensors[key].shape) != dimensions:
                    invalid.append(key)
        report["processors"][role] = {
            "config_file": filename,
            "state_file": step["state_file"],
            "state_file_sha256": hashlib.sha256(state_file.read_bytes()).hexdigest(),
            "features": step["config"]["features"],
            "norm_map": step["config"]["norm_map"],
            "required_keys": required,
            "missing_keys": [key for key in required if key not in tensors],
            "available_tensors": available,
            "invalid_tensors": sorted(set(invalid)),
            "dataset_prefixes": sorted(
                {key.split(".buffer.")[0] for key in tensors if ".buffer." in key}
            ),
        }
    # Shape/key checks alone cannot approve semantics. Only the explicitly audited
    # checkpoint pin and its authenticated native processor artifacts are accepted.
    mismatches = [
        name for name, expected in checkpoint.artifact_sha256
        if not (directory / name).is_file() or file_sha256(directory / name) != expected
    ]
    report["artifact_hash_mismatches"] = mismatches
    if mismatches:
        report["semantic_blockers"].append(
            "Artifacts do not match the approved checkpoint/dataset/processor pairing."
        )
    report["verified"] = not mismatches and all(
        not item["missing_keys"] and not item["invalid_tensors"]
        for item in report["processors"].values()
    )
    if report["verified"]:
        report["provenance"] = {
            "dataset": checkpoint.dataset,
            "dataset_revision": checkpoint.dataset_revision,
            "joint_names": checkpoint.joint_names,
            "coordinate_convention": checkpoint.coordinate_convention,
            "statistics_source": "native checkpoint processor files; no override",
            "pairing_evidence": "docs/DATA.md and assets/sample_inputs/dataset_metadata.json",
        }
    return report


def require_verified_normalization(report: dict) -> None:
    """Fail explicitly rather than choosing a prefix or allowing identity fallback."""
    missing = {
        role: item["missing_keys"]
        for role, item in report["processors"].items()
        if item["missing_keys"]
    }
    invalid = {
        role: item["invalid_tensors"]
        for role, item in report["processors"].items()
        if item["invalid_tensors"]
    }
    if missing or invalid or report["verified"] is not True:
        raise NormalizationCompatibilityError(
            f"Normalization blocked for {report['checkpoint']}@{report['revision']}: "
            f"missing={missing}; invalid={invalid}. "
            "State statistics and the action-profile semantics must be verified together. "
            "No dataset-prefix or reference-statistics fallback is allowed. "
            "See docs/DATA.md (upstream LeRobot #4415)."
        )


class SmolVLAAdapter(BaseVLAAdapter):
    """Load only after checkpoint normalization has a verified semantic mapping."""

    def __init__(self, checkpoint: SmolVLACheckpoint = MVP_CHECKPOINT) -> None:
        self.checkpoint = checkpoint
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._options = LoadOptions()
        self._load_seconds = 0.0
        self._normalization = None
        self._snapshot_path = None
        self._state_normalization_verified = False
        self._raw_state = ()
        self._raw_images = {}

    def get_input_spec(self) -> InputSpec:
        return InputSpec(
            camera_keys=tuple(source for source, _ in self.checkpoint.camera_map),
            state_dim=len(self.checkpoint.joint_names),
            normalization_description=(
                "Native state/action statistics verified against the pinned dataset. "
                + self.checkpoint.coordinate_convention
            ),
        )

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(capture_kinds=("activation_summary", "token_norms", "flow_states"))

    def get_inspection_spec(self) -> dict:
        from vla_visualizer.adapters.smolvla_view_spec import inspection_spec

        return inspection_spec(self.checkpoint)

    def get_structure(self) -> ModelStructure:
        stages = (
            Stage("image", "RGB camera image(s)"),
            Stage("preprocess", "CHW [0,1] → resize/pad 512² → [-1,1]"),
            Stage(
                "vision",
                "SmolVLM vision encoder",
                ("model.vlm_with_expert.vlm.model.vision_model",),
            ),
            Stage("features", "Visual connector", ("model.vlm_with_expert.vlm.model.connector",)),
            Stage("instruction", "Instruction → newline → tokenizer (48 tokens)"),
            Stage(
                "language",
                "Language embeddings",
                ("model.vlm_with_expert.vlm.model.text_model.embed_tokens",),
            ),
            Stage("state", "6D state → normalize → pad 32 → projection", ("model.state_proj",)),
            Stage(
                "fusion",
                "16-layer VLM prefix / KV cache",
                ("model.vlm_with_expert.vlm.model.text_model",),
            ),
            Stage(
                "decoder",
                "Action expert · 10 flow-matching steps",
                ("model.vlm_with_expert.lm_expert",),
            ),
            Stage(
                "action",
                "Action projection → slice 6D → unnormalize · 50×6",
                ("model.action_out_proj",),
            ),
        )
        return ModelStructure(
            stages=stages,
            edges=(
                ("image", "preprocess"),
                ("preprocess", "vision"),
                ("vision", "features"),
                ("features", "fusion"),
                ("instruction", "language"),
                ("language", "fusion"),
                ("state", "fusion"),
                ("fusion", "decoder"),
                ("decoder", "action"),
            ),
            provenance="adapter",
        )

    @staticmethod
    def _snapshot(repo: str, revision: str, files: tuple[str, ...], options: LoadOptions) -> str:
        from huggingface_hub import snapshot_download

        from vla_visualizer.config import cache_dir

        cache = options.cache_dir or str(cache_dir())
        path = snapshot_download(
            repo,
            revision=revision,
            cache_dir=cache,
            allow_patterns=list(files),
            local_files_only=options.local_files_only,
            max_workers=2,
        )
        missing = [name for name in files if not (Path(path) / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete cached snapshot for {repo}: {missing}. Enable downloads when loading."
            )
        return path

    def load(self, options: LoadOptions) -> None:
        import torch

        if version("lerobot") != "0.6.1":
            raise RuntimeError("This adapter requires the verified LeRobot 0.6.1 implementation.")
        if options.device not in {"cpu", "cuda"}:
            raise ValueError("Select cpu or cuda.")
        if options.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select CPU; no driver repair is attempted.")
        if not 1 <= options.cpu_threads <= 32:
            raise ValueError("CPU threads must be between 1 and 32.")
        if self._policy is not None:
            raise RuntimeError("Unload the current model before loading again.")
        start = perf_counter()
        self._options = options
        try:
            # Fail before weights whenever the pin, native statistics or domain approval differ.
            checkpoint = self._snapshot(
                self.checkpoint.repo_id,
                self.checkpoint.revision,
                tuple(name for name in CHECKPOINT_FILES if name != "model.safetensors"),
                options,
            )
            self._normalization = inspect_checkpoint_normalization(Path(checkpoint), self.checkpoint)
            require_verified_normalization(self._normalization)

            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

            checkpoint = self._snapshot(
                self.checkpoint.repo_id, self.checkpoint.revision, CHECKPOINT_FILES, options
            )
            self._snapshot_path = checkpoint
            if file_sha256(Path(checkpoint) / "model.safetensors") != self.checkpoint.weights_sha256:
                raise RuntimeError("Checkpoint weight hash differs from the approved revision.")
            backbone = self._snapshot(BACKBONE, BACKBONE_REVISION, BACKBONE_FILES, options)
            config = SmolVLAConfig.from_pretrained(checkpoint)
            config.device = options.device
            # Full VLA weights include the vision/text backbone; strict loading checks all keys.
            config.load_vlm_weights = False
            config.vlm_model_name = backbone
            if options.device == "cpu":
                torch.set_num_threads(options.cpu_threads)
            self._policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=True)
            from safetensors import safe_open

            state_dict = self._policy.state_dict()
            with safe_open(str(Path(checkpoint) / "model.safetensors"), framework="pt") as weights:
                checkpoint_keys = set(weights.keys())
                missing = set(state_dict) - checkpoint_keys
                unexpected = checkpoint_keys - set(state_dict)
                if missing or unexpected:
                    raise RuntimeError(
                        f"Strict checkpoint schema mismatch: missing={sorted(missing)}, "
                        f"unexpected={sorted(unexpected)}"
                    )
            self._loading_keys = {
                "tensor_count": len(checkpoint_keys), "missing": [], "unexpected": [],
            }
            self._policy.float().requires_grad_(False).eval()
            self._preprocessor, self._postprocessor = make_pre_post_processors(
                config,
                checkpoint,
                preprocessor_overrides={
                    "device_processor": {"device": options.device},
                    "tokenizer_processor": {"tokenizer_name": backbone},
                },
            )
            modules = dict(self._policy.named_modules())
            for stage in self.get_structure().stages:
                for name in stage.module_paths:
                    if name not in modules:
                        raise RuntimeError(f"Verified architecture module is missing: {name}")
            if (
                tuple(config.image_features) != tuple(dst for _, dst in self.checkpoint.camera_map)
                or config.action_feature.shape != (len(self.checkpoint.joint_names),)
                or config.input_features["observation.state"].shape
                != (len(self.checkpoint.joint_names),)
            ):
                raise RuntimeError("Pinned checkpoint input/output schema changed unexpectedly.")
            self._load_seconds = perf_counter() - start
        except Exception:
            self.unload()
            raise

    def unload(self) -> None:
        import torch

        self._policy = self._preprocessor = self._postprocessor = None
        self._state_normalization_verified = False
        self._raw_state = ()
        self._raw_images = {}
        gc.collect()
        if self._options.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def validate_observation(self, observation: Observation) -> None:
        spec = self.get_input_spec()
        if not observation.instruction.strip():
            raise ValueError("A nonempty instruction is required.")
        if observation.state is None or len(observation.state) != spec.state_dim:
            raise ValueError(
                "Provide six measured robot-state values in the checkpoint's state convention."
            )
        if not all(math.isfinite(value) for value in observation.state):
            raise ValueError("Robot-state values must be finite.")
        if set(observation.images) != set(spec.camera_keys):
            raise ValueError(
                f"Provide both required cameras with supported keys: {spec.camera_keys}"
            )
        if observation.provenance:
            if (
                observation.provenance.get("dataset") != self.checkpoint.dataset
                or observation.provenance.get("revision") != self.checkpoint.dataset_revision
            ):
                raise ValueError("Observation dataset/revision differs from the approved pairing.")
            expected_names = [f"main_{name}" for name in self.checkpoint.joint_names]
            if observation.provenance.get("state_names") != expected_names:
                raise ValueError("Observation state ordering differs from the approved pairing.")
        for image in observation.images.values():
            if image.width * image.height > 16_000_000 or min(image.size) < 1:
                raise ValueError("Image dimensions exceed the supported limit.")

    def preprocess(self, observation: Observation) -> PreparedInputs:
        import numpy as np
        import torch

        self.validate_observation(observation)
        self._state_normalization_verified = False
        if self._policy is None:
            raise RuntimeError("Load the model before preprocessing.")
        raw = {
            "observation.state": torch.tensor(observation.state, dtype=torch.float32),
            "task": observation.instruction.strip(),
        }
        for name, image in observation.images.items():
            pixels = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
            raw[name] = torch.from_numpy(pixels).permute(2, 0, 1).float().div(255)
        prepared = self._preprocessor(raw)
        tensors = self._normalization["processors"]["preprocessor"]["available_tensors"]
        mean = torch.tensor(
            tensors["observation.state.mean"]["values"], device=self._options.device,
            dtype=torch.float32,
        )
        std = torch.tensor(
            tensors["observation.state.std"]["values"], device=self._options.device,
            dtype=torch.float32,
        )
        # Verification only: the native processor performs the actual transform.
        expected = (torch.tensor(
            observation.state, device=self._options.device, dtype=torch.float32
        ) - mean) / (
            std + 1e-8
        )
        state = prepared["observation.state"]
        if state.shape != (1, len(observation.state)) or not torch.allclose(state[0], expected):
            raise RuntimeError("Native state normalization was skipped or changed unexpectedly.")
        for _, native_key in self.checkpoint.camera_map:
            if prepared[native_key].ndim != 4:
                raise RuntimeError("Native camera preprocessing did not produce BCHW tensors.")
        if prepared["task"] != [observation.instruction.strip() + "\n"]:
            raise RuntimeError("Native task formatting differs from the verified processor.")
        for value in prepared.values():
            if isinstance(value, torch.Tensor):
                if value.device.type != self._options.device or not value.isfinite().all():
                    raise RuntimeError("Preprocessed tensors have an invalid device or value.")
                if value.is_floating_point() and value.dtype != torch.float32:
                    raise RuntimeError("Inference requires float32 preprocessed tensors.")
        self._state_normalization_verified = True
        self._normalized_state = state.detach().cpu().tolist()
        self._observation_provenance = dict(observation.provenance)
        self._raw_state = tuple(observation.state)
        self._raw_images = {
            key: {"size_wh": image.size, "mode": image.mode}
            for key, image in observation.images.items()
        }
        return prepared

    def run_inference(self, inputs: PreparedInputs, capture: CaptureOptions) -> InferenceResult:
        import torch

        if self._policy is None:
            raise RuntimeError("Load the model before inference.")
        if not self._state_normalization_verified:
            raise RuntimeError("Verified native preprocessing is required before inference.")
        inspection = None
        if capture.enabled or capture.module_names:
            from vla_visualizer.adapters.smolvla_inspection import SmolVLAInspection

            inspection = SmolVLAInspection(self._policy, capture, self.checkpoint.camera_map)
            hook_count_before = sum(len(module._forward_hooks) for module in self._policy.modules())
        device = self._options.device
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        with torch.inference_mode():
            generator = torch.Generator(device=device).manual_seed(self._options.seed)
            config = self._policy.config
            noise = torch.randn(
                (1, config.chunk_size, config.max_action_dim),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            self._policy.reset()
            if inspection is not None:
                inspection.capture_inputs(inputs, self._raw_state, noise)
            with inspection.hooks if inspection is not None else nullcontext():
                with inspection.trace if inspection is not None and capture.extended else nullcontext():
                    normalized = self._policy.predict_action_chunk(dict(inputs), noise=noise)
            if inspection is not None:
                hook_count_after = sum(len(module._forward_hooks) for module in self._policy.modules())
                if hook_count_after != hook_count_before:
                    raise RuntimeError("Inspection left persistent hooks after inference.")
            actions = self._postprocessor(normalized).detach()
            tensors = self._normalization["processors"]["postprocessor"]["available_tensors"]
            mean = torch.tensor(tensors["action.mean"]["values"], device=device, dtype=torch.float32)
            std = torch.tensor(tensors["action.std"]["values"], device=device, dtype=torch.float32)
            expected = (normalized * std + mean).cpu()
            if not torch.allclose(actions, expected, rtol=1e-5, atol=1e-6):
                raise RuntimeError("Native action inverse normalization was skipped or changed.")
            intermediates = inspection.finish(normalized, actions) if inspection is not None else {}
            if inspection is not None:
                intermediates["raw_images"] = self._raw_images
                intermediates["state_names"] = self.checkpoint.joint_names
                intermediates["hook_count_before"] = hook_count_before
                intermediates["hook_count_after"] = hook_count_after
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = perf_counter() - start
        if (
            tuple(actions.shape) != (1, config.chunk_size, config.action_feature.shape[0])
            or not torch.isfinite(actions).all()
            or actions.dtype != torch.float32 or actions.device.type != "cpu"
        ):
            raise RuntimeError(f"Invalid action chunk returned: shape={tuple(actions.shape)}")
        metadata = {
            "checkpoint": self.checkpoint.repo_id,
            "revision": self.checkpoint.revision,
            "snapshot_path": self._snapshot_path,
            "strict_loading": True,
            "loading_keys": self._loading_keys,
            "state_normalization_verified": self._state_normalization_verified,
            "normalized_state": self._normalized_state,
            "action_postprocessing_verified": True,
            "normalized_first_action": normalized[0, 0].cpu().tolist(),
            "action_chunk_shape": list(actions.shape),
            "action_output_dtype": str(actions.dtype),
            "action_output_device": str(actions.device),
            "action_names": self.checkpoint.joint_names,
            "coordinate_convention": self.checkpoint.coordinate_convention,
            "observation_provenance": self._observation_provenance,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "backbone": BACKBONE,
            "backbone_revision": BACKBONE_REVISION,
            "device": device,
            "dtype": str(next(self._policy.parameters()).dtype),
            "seed": self._options.seed,
            "load_seconds": self._load_seconds,
            "inference_seconds": elapsed,
            "native_action_shape": list(normalized.shape),
            "action_shape": list(actions.shape[1:]),
            "action_scale": "verified checkpoint scale, unnormalized",
            "normalization_verified": self._normalization["verified"],
            "normalization_audit": self._normalization,
            "denoising_steps": config.num_steps,
            "frozen": all(not p.requires_grad for p in self._policy.parameters()),
            "eval_mode": not self._policy.training,
            "parameter_count": sum(p.numel() for p in self._policy.parameters()),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in self._policy.parameters()),
            "input_shapes": {
                key: list(value.shape)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            },
            "attention_available": False,
            "intermediate_capture": "bounded measured summaries" if inspection is not None else "disabled",
            "versions": {
                name: version(name)
                for name in (
                    "lerobot",
                    "torch",
                    "torchvision",
                    "transformers",
                    "huggingface-hub",
                    "safetensors",
                    "accelerate",
                    "numpy",
                    "Pillow",
                    "streamlit",
                    "graphviz",
                )
            },
        }
        if device == "cuda":
            metadata["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        return InferenceResult(
            actions=tuple(tuple(row) for row in actions[0].tolist()),
            intermediates=intermediates, metadata=metadata
        )
