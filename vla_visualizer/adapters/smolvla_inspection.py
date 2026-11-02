"""Verified LeRobot 0.6.1 SmolVLA capture points, without modifying model methods."""

from collections import Counter
import base64
from dataclasses import asdict
from io import BytesIO

import torch

from vla_visualizer.contracts import CaptureOptions
from vla_visualizer.utils.hooks import CaptureBuffer, SafeForwardHooks


ROOT = "model.vlm_with_expert.vlm.model"
POINTS = {
    "vision": f"{ROOT}.vision_model",
    "connector": f"{ROOT}.connector",
    "language": f"{ROOT}.text_model.embed_tokens",
    "state": "model.state_proj",
    "multimodal_prefix": f"{ROOT}.text_model.layers.0.input_layernorm",
    "flow_state": "model.action_in_proj",
    "time_conditioned_actions": "model.action_time_mlp_out",
    "action_velocity": "model.action_out_proj",
}
DESCRIPTIONS = {
    "vision": "SigLIP transformer patch-token sequence, after final vision normalization",
    "connector": "Pixel-shuffle patch grouping and projection into text embedding width",
    "language": "Learned token embeddings before sqrt(hidden_size) scaling",
    "state": "Normalized state zero-padded to max_state_dim, projected into one prefix token",
    "multimodal_prefix": "Concatenated/scaled visual and language tokens plus state at first VLM layer",
    "flow_state": "Current x_t (padded normalized action space), projected into action expert width",
    "time_conditioned_actions": "Final action/time MLP output: action-expert suffix input tokens",
    "action_velocity": "Final expert hidden representations projected to flow velocity v_t",
}


class SmolVLAInspection:
    """One bounded capture session. Whole expert/VLM .forward calls bypass hooks."""

    def __init__(
        self,
        policy,
        options: CaptureOptions,
        camera_map,
        *,
        state_description="Native verified six-joint state normalization",
        action_description="Native verified inverse normalization to absolute targets",
    ) -> None:
        unknown = set(options.module_names) - set(POINTS.values())
        if unknown:
            raise ValueError(f"Only approved inspection modules are supported: {sorted(unknown)}")
        self.policy = policy
        self.state_description = state_description
        self.action_description = action_description
        self.buffer = CaptureBuffer(options)
        self.calls = Counter()
        self.options = options
        self.token_features = []
        self.processed_images = {}
        self.observed_order = []
        self.auxiliary = {}
        self.steps = tuple(
            sorted({0, (policy.config.num_steps - 1) // 2, policy.config.num_steps - 1})
        )
        self.camera_map = camera_map
        self.vision_layout = {}
        self.camera_input_shapes = {}
        self.language = []
        self.selected = {
            stage: name
            for stage, name in POINTS.items()
            if not options.module_names or name in options.module_names
        }
        callbacks = {name: self._callback(stage, name) for stage, name in self.selected.items()}
        if options.extended:
            from vla_visualizer.adapters.smolvla_trace import NativeExecutionTrace

            self.trace = NativeExecutionTrace(policy)
            callbacks[f"{ROOT}.text_model.norm"] = self._prefix_output
        self.hooks = SafeForwardHooks(policy, callbacks)

    def _features(self, tensor, stage, module, context, record_id):
        value = tensor.detach().float()
        rows = value.reshape(-1, value.shape[-1])
        if rows.shape[0] > 1024:
            raise ValueError("Per-token partial-channel capture limit exceeded")
        width = min(rows.shape[-1], self.options.token_channels)
        self.token_features.append(
            {
                "stage": stage,
                "module": module,
                **context,
                "record_id": record_id,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "original_dimension": rows.shape[-1],
                "channel_indices": list(range(width)),
                "token_indices": list(range(rows.shape[0])),
                "values": rows[:, :width].cpu().tolist(),
                "summary_method": "Measured first channels 0:K per token; partial values, not reconstructed or aggregated full vectors",
            }
        )

    def _prefix_output(self, module, args, kwargs, output):
        from vla_visualizer.utils.hooks import summarize

        name = f"{ROOT}.text_model.norm"
        self.observed_order.append({"stage": "vlm_prefix_output", "module": name, "call_index": 0})
        self.trace.timeline.append({"stage": "vlm_prefix_output", "module": name, "call_index": 0})
        self.auxiliary["vlm_prefix_output"] = {
            "source": "forward_hook",
            "module": name,
            "summary": asdict(summarize(output, self.options)),
            "description": "actual final VLM text norm output for prefix; not entire KV cache",
        }
        self._features(output, "vlm_prefix_output", name, {}, "aux_prefix_output")

    def _callback(self, stage, name):
        def callback(module, args, kwargs, output):
            call = self.calls[name]
            self.calls[name] += 1
            context = {"call_index": call}
            if self.options.extended:
                self.observed_order.append({"stage": stage, "module": name, "call_index": call})
                self.trace.timeline.append(
                    {
                        "stage": stage,
                        "module": name,
                        "call_index": call,
                        "iteration": call
                        if stage in {"flow_state", "time_conditioned_actions", "action_velocity"}
                        else None,
                        "camera": self.camera_map[call][0]
                        if stage in {"vision", "connector"}
                        else None,
                    }
                )
            if stage in {"flow_state", "time_conditioned_actions", "action_velocity"}:
                if call not in self.steps:
                    return
                context.update(
                    {
                        "flow_step_index": call,
                        "schedule_time": 1.0 - call / self.policy.config.num_steps,
                        "time_source": "Derived from native euler_integrate schedule; not a hooked time tensor",
                    }
                )
            if stage == "vision":
                value = kwargs["pixel_values"]
                result = output.last_hidden_state
                source_key, native_key = self.camera_map[call]
                patch = module.config.patch_size
                grid = (value.shape[-2] // patch, value.shape[-1] // patch)
                before_h, before_w = self.camera_input_shapes[native_key][-2:]
                height, width = value.shape[-2:]
                ratio = max(before_w / width, before_h / height)
                resized = (int(before_h / ratio), int(before_w / ratio))
                self.vision_layout.setdefault(source_key, {}).update(
                    {
                        "native_key": native_key,
                        "input_shape": list(value.shape),
                        "patch_size": patch,
                        "patch_grid": grid,
                        "token_count": result.shape[1],
                        "embedding_dim": result.shape[2],
                        "ordering": "Row-major patch grid, no CLS token; transformer contextualizes each token",
                        "vision_attentions_returned": getattr(output, "attentions", None)
                        is not None,
                        "resize_geometry": {
                            "source_hw": (before_h, before_w),
                            "resized_hw": resized,
                            "padding_top_left": (height - resized[0], width - resized[1]),
                            "source": "Derived from measured input shapes and native resize_with_pad implementation",
                            "mode": "bilinear, align_corners=False; padding value -1 after image scaling",
                        },
                    }
                )
                context.update({"camera": source_key, "native_key": native_key})
            else:
                value, result = args[0], output
            if stage == "connector":
                source_key, native_key = self.camera_map[call]
                scale = module.scale_factor
                context.update(
                    {"camera": source_key, "native_key": native_key, "pixel_shuffle_scale": scale}
                )
                self.vision_layout.setdefault(source_key, {}).update(
                    {
                        "connector_token_count": result.shape[1],
                        "connector_embedding_dim": result.shape[2],
                        "connector_grid": (int(result.shape[1] ** 0.5),) * 2,
                        "patch_group_size": (scale, scale),
                        "mapping": "Row-major groups on padded/resized image; spatial indices are not attention or attribution",
                    }
                )
            if stage == "language":
                tokenizer = self.policy.model.vlm_with_expert.processor.tokenizer
                ids = value.reshape(-1).detach().cpu().tolist()
                if len(ids) > self.policy.config.tokenizer_max_length:
                    raise ValueError("Unexpected unbounded language token capture.")
                kind = "instruction" if value.ndim == 2 else "image_special_tokens"
                self.language.append(
                    {
                        "kind": kind,
                        "token_ids": ids,
                        "tokens": tokenizer.convert_ids_to_tokens(ids),
                        "call_index": call,
                    }
                )
                context["kind"] = kind
            if self.options.extended:
                record_id = f"record_{len(self.buffer.records):02d}"
                self._features(result, stage, name, context, record_id)
                if stage == "action_velocity":
                    self._features(value, "expert_hidden", name, context, record_id)
                if stage == "vision":
                    from PIL import Image

                    pixels = (
                        ((value[0].detach().float() + 1) / 2)
                        .clamp(0, 1)
                        .mul(255)
                        .round()
                        .byte()
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )
                    stream = BytesIO()
                    Image.fromarray(pixels).save(stream, format="PNG")
                    self.processed_images[context["camera"]] = {
                        "png_base64": base64.b64encode(stream.getvalue()).decode(),
                        "source": name
                        + " actual pixel_values argument; 8-bit display quantization",
                        "shape": list(value.shape),
                        "range_quantization": "RGB display rounded to nearest 1/255",
                    }
            self.buffer.add(
                stage,
                name,
                DESCRIPTIONS[stage],
                "forward_hook",
                {"tensor": value},
                {"tensor": result},
                context,
            )

        return callback

    def capture_inputs(self, inputs, raw_state, noise) -> None:
        self.buffer.add(
            "robot_state_normalization",
            "policy_preprocessor",
            self.state_description,
            "processor_value",
            {"raw_state": torch.tensor(raw_state, dtype=torch.float32)},
            {"normalized_state": inputs["observation.state"]},
        )
        for raw_key, native_key in self.camera_map:
            self.camera_input_shapes[native_key] = tuple(inputs[native_key].shape)
            self.buffer.add(
                "camera_preprocessing",
                "policy_preprocessor",
                "RGB observation converted to BCHW [0,1]",
                "processor_value",
                {},
                {"image": inputs[native_key]},
                {"camera": raw_key, "native_key": native_key},
            )
        self.buffer.add(
            "language_tokenization",
            "policy_preprocessor",
            "Native newline-formatted instruction, right-padded tokens",
            "processor_value",
            {},
            {
                "token_ids": inputs["observation.language.tokens"],
                "attention_mask": inputs["observation.language.attention_mask"],
            },
            {
                "task": inputs["task"],
                "valid_token_count": int(
                    inputs["observation.language.attention_mask"].sum().item()
                ),
            },
        )
        self.buffer.add(
            "initial_noise",
            "SmolVLAPolicy.predict_action_chunk(noise=)",
            "Seeded frozen-inference initial noise x_1",
            "model_input",
            {},
            {"noise": noise},
        )

    def finish(self, normalized, actions) -> dict:
        missing = set(self.selected.values()) - set(self.calls)
        if missing:
            raise RuntimeError(f"Inspection points did not execute: {sorted(missing)}")
        for stage, name in self.selected.items():
            if stage in {"vision", "connector"} and self.calls[name] != len(self.camera_map):
                raise RuntimeError("Unexpected camera execution count; semantic labels are unsafe.")
            if stage in {"flow_state", "time_conditioned_actions", "action_velocity"}:
                if self.calls[name] != self.policy.config.num_steps:
                    raise RuntimeError("Unexpected flow iteration count.")
        self.buffer.add(
            "normalized_action_chunk",
            "SmolVLAPolicy.predict_action_chunk",
            f"Integrated x_0 sliced to {normalized.shape[-1]} action dimensions",
            "model_return",
            {},
            {"normalized_action": normalized},
        )
        self.buffer.add(
            "action_postprocessing",
            "policy_postprocessor",
            self.action_description,
            "processor_value",
            {"normalized_action": normalized},
            {"action": actions},
        )
        result = {
            "schema_version": 1,
            "stages": tuple(self.buffer.records),
            "summary_seconds": self.buffer.summary_seconds,
            "executed_hook_counts": dict(self.calls),
            "selected_flow_steps": self.steps,
            "vision": self.vision_layout,
            "language": self.language,
            "attention_available": False,
            "attention_reason": "Native expert eager attention returns context only; vision returned no attentions. No model patching/request changes.",
            "retention": "Python numeric summaries only; <=64 values for small tensors, bounded previews/token norms; no tensors or KV caches",
        }
        if self.options.extended:
            result["extended_capture"] = {
                "schema_version": 2,
                "token_features": self.token_features,
                "processed_images": self.processed_images,
                "observed_module_order": self.observed_order,
                "full_loop": self.trace.events,
                "function_timeline": self.trace.timeline,
                "prefix_processing": self.trace.prefix,
                "auxiliary": self.auxiliary,
                "solver": "native euler_integrate, dt=-1/N, x<-x+dt*v; profiler observes, never changes solver",
            }
        return result
