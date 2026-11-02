"""Immutable bounded replay snapshots, validation and capture-to-view processing.

No model, adapter, torch or Streamlit dependency. Missing values stay unavailable.
"""

import base64
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from typing import Any
from uuid import uuid4

from PIL import Image

from vla_visualizer.contracts import Observation
from vla_visualizer.inference.observations import observation_fingerprint

MAX_REPLAY_BYTES = 6 * 1024 * 1024


class ReplayError(ValueError):
    """Invalid or unsupported saved measurements; never replace with synthetic data."""


@dataclass(frozen=True)
class ReplayRun:
    """One immutable serialized completed run; no duplicated persistent arrays."""

    snapshot: str

    def read(self) -> dict:
        return json.loads(self.snapshot)

    @property
    def run_id(self) -> str:
        return self.read()["manifest"]["run_id"]


def _finite(value) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _summary(summary: dict) -> None:
    shape = summary.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) > 6
        or any(type(size) is not int or not 1 <= size <= 1_000_000 for size in shape)
    ):
        raise ReplayError("Invalid captured tensor shape.")
    numel = math.prod(shape)
    if summary.get("numel") != numel or numel > 1_000_000_000:
        raise ReplayError("Captured shape and element count disagree.")
    for key in ("minimum", "maximum", "mean", "std", "l2_norm"):
        if not _finite(summary.get(key)):
            raise ReplayError(f"Invalid captured statistic: {key}")
    if summary["minimum"] > summary["maximum"] or summary["std"] < 0 or summary["l2_norm"] < 0:
        raise ReplayError("Invalid captured statistic range.")
    for key, bound in (("preview", 64), ("values", 64), ("token_norms", 1024)):
        values = summary.get(key)
        if values is not None and (
            not isinstance(values, list)
            or len(values) > bound
            or any(not _finite(v) for v in values)
        ):
            raise ReplayError(f"Invalid or unbounded captured {key}.")
    if summary.get("values") is not None and len(summary["values"]) != numel:
        raise ReplayError("Captured full values do not match shape.")
    indices, norms = summary.get("token_indices", []), summary.get("token_norms", [])
    if not isinstance(indices, list) or len(indices) != len(norms):
        raise ReplayError("Token indices and norms disagree.")
    if indices and (
        len(shape) != 3
        or any(type(i) is not int or not 0 <= i < math.prod(shape[:-1]) for i in indices)
        or indices != sorted(set(indices))
        or any(v < 0 for v in norms)
    ):
        raise ReplayError("Invalid token norm indices.")
    if not isinstance(summary.get("dtype"), str) or not isinstance(summary.get("device"), str):
        raise ReplayError("Missing original execution dtype/device.")


def validate_capture(capture: dict) -> None:
    if capture.get("intermediates", {}).get("schema_version") != 1:
        raise ReplayError("Unsupported capture schema; expected schema_version=1.")
    records = capture["intermediates"].get("stages", [])
    if not isinstance(records, list) or len(records) > 64:
        raise ReplayError("Invalid or unbounded stage records.")
    for record in records:
        if not isinstance(record, dict) or not all(
            isinstance(record.get(k), str) for k in ("stage_name", "module_name", "description")
        ):
            raise ReplayError("Invalid stage identity.")
        if record.get("source") not in {
            "forward_hook",
            "processor_value",
            "model_return",
            "model_input",
        }:
            raise ReplayError("Unknown capture source.")
        context = record.get("context", {})
        if not isinstance(context, dict):
            raise ReplayError("Invalid capture context.")
        if "flow_step_index" in context and (
            type(context["flow_step_index"]) is not int
            or not 0 <= context["flow_step_index"] < capture["metadata"].get("denoising_steps", 0)
        ):
            raise ReplayError("Invalid captured flow iteration.")
        for side in ("inputs", "outputs"):
            if not isinstance(record.get(side), dict) or len(record[side]) > 8:
                raise ReplayError("Invalid stage tensor mapping.")
            for summary in record[side].values():
                _summary(summary)
    actions, metadata = capture.get("actions"), capture.get("metadata", {})
    names = metadata.get("action_names", [])
    if not isinstance(actions, list) or not 1 <= len(actions) <= 1000 or not 1 <= len(names) <= 64:
        raise ReplayError("Final action chunk or semantic names are unavailable.")
    if any(
        not isinstance(row, list) or len(row) != len(names) or any(not _finite(v) for v in row)
        for row in actions
    ):
        raise ReplayError("Invalid/nonfinite action chunk.")
    if metadata.get("action_chunk_shape") != [1, len(actions), len(names)]:
        raise ReplayError("Final action shape disagrees with measurements.")
    if not all(
        metadata.get(key) is True
        for key in (
            "normalization_verified",
            "state_normalization_verified",
            "action_postprocessing_verified",
            "strict_loading",
            "frozen",
            "eval_mode",
        )
    ):
        raise ReplayError("Completed run lacks verified inference/normalization safeguards.")
    extended=capture['intermediates'].get('extended_capture')
    if extended is not None:
        validate_extended(extended,capture)


def validate_extended(extended,capture):
    if extended.get('schema_version')!=2:
        raise ReplayError('Unsupported extended capture schema')
    features=extended.get('token_features',[])
    if not isinstance(features,list) or len(features)>64:
        raise ReplayError('Unbounded token feature records')
    for item in features:
        indices=item.get('token_indices',[])
        channels=item.get('channel_indices',[])
        values=item.get('values',[])
        if not 1<=len(indices)<=1024 or not 1<=len(channels)<=32 or indices!=list(range(len(indices))):
            raise ReplayError('Invalid token/channel selection')
        if channels!=list(range(len(channels))) or not len(channels)<=item.get('original_dimension',0)<=4096:
            raise ReplayError('Invalid partial channel mapping')
        if len(values)!=len(indices) or any(len(row)!=len(channels) or any(not _finite(v) for v in row) for row in values):
            raise ReplayError('Invalid measured partial vectors')
    loop=extended.get('full_loop',[])
    if len(loop)!=capture['metadata']['denoising_steps'] or len(loop)>16:
        raise ReplayError('Full loop count does not match native inference')
    for k,event in enumerate(loop):
        if event.get('iteration')!=k or not _finite(event.get('time')) or abs(event['time']-(1-k/len(loop)))>1e-6:
            raise ReplayError('Invalid native flow time/order')
        for key in ('x_before','velocity','x_after'):
            tensor=event.get(key,[])
            if len(tensor)!=1 or not 1<=len(tensor[0])<=1000 or sum(len(row) for row in tensor[0])>2048:
                raise ReplayError('Unbounded loop action state')
            if any(any(not _finite(v) for v in row) for row in tensor[0]):
                raise ReplayError('Invalid loop values')
    for image in extended.get('processed_images',{}).values():
        raw=base64.b64decode(image['png_base64'],validate=True)
        with Image.open(BytesIO(raw)) as pixels:
            if pixels.width*pixels.height>1024**2:
                raise ReplayError('Unbounded measured policy-input image')


def make_run(
    result: Any,
    spec: dict,
    observation: Observation | None = None,
    *,
    origin: str = "live captured inference",
) -> ReplayRun:
    """Serialize once, validate, then publish atomically after a successful run."""
    capture = asdict(result) if is_dataclass(result) else result
    # Roundtrip makes an independent snapshot; discard historical validation arrays.
    capture = json.loads(
        json.dumps(
            {key: capture[key] for key in ("actions", "metadata", "intermediates")}, allow_nan=False
        )
    )
    validate_capture(capture)
    metadata = capture["metadata"]
    if any(metadata.get(k) != spec.get(k) for k in ("checkpoint", "revision")):
        raise ReplayError("Capture and source-verified view specification do not match.")
    provenance = metadata.get("observation_provenance", {})
    if provenance and (
        provenance.get("dataset") != spec.get("dataset")
        or provenance.get("revision") != spec.get("dataset_revision")
    ):
        raise ReplayError("Capture dataset differs from the verified pairing.")
    tokens = select_records(capture, ["language_tokenization"])
    task = tokens[0]["context"].get("task", []) if tokens else []
    instruction = task[0].removesuffix("\n") if task else None
    images = {}
    if observation is not None:
        if instruction is not None and observation.instruction.strip() != instruction:
            raise ReplayError(
                "Saved capture instruction does not match supplied images/observation."
            )
        state = select_records(capture, ["robot_state_normalization"])
        raw = state[0]["inputs"].get("raw_state", {}).get("values") if state else None
        if raw is not None and any(
            abs(a - b) > 1e-5 for a, b in zip(raw, observation.state or (), strict=True)
        ):
            raise ReplayError("Saved capture state does not match supplied observation.")
        for key, image in observation.images.items():
            stream = BytesIO()
            image.convert("RGB").save(stream, format="PNG")
            data = stream.getvalue()
            images[key] = {
                "png_base64": base64.b64encode(data).decode(),
                "sha256": sha256(data).hexdigest(),
            }
    ids = []
    for index, record in enumerate(capture["intermediates"]["stages"]):
        context = record.get("context", {})
        ids.append(
            {
                "id": f"record_{index:02d}",
                "capture_index": index,
                "stage": record["stage_name"],
                "camera": context.get("camera"),
                "flow_step": context.get("flow_step_index"),
                "time_value": context.get("schedule_time"),
                "time_source": context.get("time_source"),
                "representation": record["source"],
            }
        )
    manifest = {
        "schema_version": 1,
        "run_id": str(uuid4()),
        "origin": origin,
        "checkpoint": metadata["checkpoint"],
        "revision": metadata["revision"],
        "dataset": provenance.get("dataset"),
        "dataset_revision": provenance.get("revision"),
        "sample_identity": provenance.get("sample_identity", "unverified custom observation"),
        "instruction": instruction,
        "seed": metadata.get("seed"),
        "device": metadata.get("device"),
        "software_versions": metadata.get("versions", {}),
        "records": ids,
        "observation_fingerprint": observation_fingerprint(observation) if observation else None,
        "timing_scope": "inference_seconds: complete prediction + postprocessing + capture summaries; pipeline_seconds also includes preprocessing; no per-stage durations",
        "summary_method": spec.get("summary_method"),
        "visualization_storage": "CPU serialized Python numbers and PNG bytes",
    }
    payload = {"manifest": manifest, "capture": capture, "view_spec": spec, "images": images}
    return load_run(json.dumps(payload, allow_nan=False, separators=(",", ":")))


def load_run(snapshot: str | bytes) -> ReplayRun:
    if len(snapshot.encode() if isinstance(snapshot, str) else snapshot) > MAX_REPLAY_BYTES:
        raise ReplayError("Replay exceeds 6 MiB payload limit.")
    try:
        data = json.loads(snapshot)
        if data["manifest"]["schema_version"] != 1 or data["view_spec"]["spec_version"] != 1:
            raise ReplayError("Unsupported replay/view schema.")
        validate_capture(data["capture"])
        metadata = data["capture"]["metadata"]
        if any(
            data["manifest"].get(key) != metadata.get(key)
            or data["view_spec"].get(key) != metadata.get(key)
            for key in ("checkpoint", "revision")
        ):
            raise ReplayError("Replay identity does not match capture.")
        descriptors = data["manifest"]["records"]
        records = data["capture"]["intermediates"]["stages"]
        if len(descriptors) != len(records) or len({r["id"] for r in descriptors}) != len(records):
            raise ReplayError("Replay record IDs do not match capture.")
        for index, (descriptor, record) in enumerate(zip(descriptors, records, strict=True)):
            context = record["context"]
            if (
                descriptor["capture_index"] != index
                or descriptor["stage"] != record["stage_name"]
                or any(
                    descriptor.get(left) != context.get(right)
                    for left, right in (
                        ("camera", "camera"),
                        ("flow_step", "flow_step_index"),
                        ("time_value", "schedule_time"),
                        ("time_source", "time_source"),
                    )
                )
                or descriptor.get("representation") != record["source"]
            ):
                raise ReplayError("Replay record mapping differs from capture.")
        spec = data["view_spec"]
        if not 1 <= len(spec["stages"]) <= 32 or len({s["id"] for s in spec["stages"]}) != len(
            spec["stages"]
        ):
            raise ReplayError("Invalid replay stage catalog.")
        provenance = metadata.get("observation_provenance", {})
        if provenance and (
            provenance.get("dataset") != spec.get("dataset")
            or provenance.get("revision") != spec.get("dataset_revision")
        ):
            raise ReplayError("Saved capture dataset differs from source-verified specification.")
        tokenized = select_records(data["capture"], ["language_tokenization"])
        tasks = tokenized[0]["context"].get("task", []) if tokenized else []
        if tasks and data["manifest"].get("instruction") != tasks[0].removesuffix("\n"):
            raise ReplayError("Replay instruction differs from captured task.")
        for key, expected in {
            "seed": metadata.get("seed"),
            "device": metadata.get("device"),
            "dataset": provenance.get("dataset"),
            "dataset_revision": provenance.get("revision"),
        }.items():
            if data["manifest"].get(key) != expected:
                raise ReplayError(f"Replay {key} differs from completed capture.")
        for record in records:
            stage, context = record["stage_name"], record["context"]
            camera = context.get("camera")
            if camera is not None and (
                camera not in spec.get("camera_to_native", {})
                or context.get("native_key") != spec["camera_to_native"][camera]
            ):
                raise ReplayError(
                    "Camera/native input mapping differs from verified specification."
                )
            order = spec.get("camera_invocation_order", {}).get(stage)
            if order is not None:
                call = context.get("call_index")
                if type(call) is not int or not 0 <= call < len(order) or order[call] != camera:
                    raise ReplayError(
                        "Camera identity differs from actual native invocation order."
                    )
            expected_path = spec.get("record_paths", {}).get(stage)
            if expected_path and record["module_name"] != expected_path:
                raise ReplayError(
                    "Stage/module mapping differs from source-verified specification."
                )
            rank = spec.get("output_ranks", {}).get(stage)
            if rank is not None and any(
                len(s["shape"]) != rank for s in record["outputs"].values()
            ):
                raise ReplayError("Captured representation rank differs from verified stage.")
            if "schedule_time" in context and not _finite(context["schedule_time"]):
                raise ReplayError("Invalid captured schedule time.")
        # Validate cross-record semantics when available, without inventing absent data.
        language_rows(data["capture"], reveal_padding=True)
        state_rows(data["capture"], spec)
        prefix_rows(data["capture"], spec)
        for image in data.get("images", {}).values():
            raw = base64.b64decode(image["png_base64"], validate=True)
            if sha256(raw).hexdigest() != image["sha256"]:
                raise ReplayError("Replay image checksum mismatch.")
            with Image.open(BytesIO(raw)) as pixels:
                if pixels.width * pixels.height > 16_000_000:
                    raise ReplayError("Replay image exceeds pixel limit.")
                pixels.verify()
        return ReplayRun(snapshot.decode() if isinstance(snapshot, bytes) else snapshot)
    except ReplayError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, OSError) as exc:
        raise ReplayError(f"Invalid replay data: {exc}") from exc


def select_records(
    capture: dict, stages: list[str], camera: str | None = None, step: int | None = None
) -> list[dict]:
    return [
        record
        for record in capture["intermediates"].get("stages", [])
        if record["stage_name"] in stages
        and (camera is None or record.get("context", {}).get("camera") == camera)
        and (
            step is None
            or "flow_step_index" not in record.get("context", {})
            or record["context"]["flow_step_index"] == step
        )
    ]


def norm_rows(summary: dict) -> list[dict]:
    return [
        {"index": i, "L2 norm": value}
        for i, value in zip(
            summary.get("token_indices", []), summary.get("token_norms", []), strict=True
        )
    ]


def language_rows(capture: dict, *, reveal_padding: bool = False) -> list[dict]:
    records = select_records(capture, ["language_tokenization"])
    language = next(
        (
            item
            for item in capture["intermediates"].get("language", [])
            if item.get("kind") == "instruction"
        ),
        None,
    )
    if not records or not language:
        return []
    masks = records[0]["outputs"].get("attention_mask", {}).get("values")
    ids = records[0]["outputs"].get("token_ids", {}).get("values")
    if masks is None or ids is None:
        return []
    if (
        len(masks) != len(ids)
        or len(ids) != len(language["tokens"])
        or ids != language["token_ids"]
    ):
        raise ReplayError("Language token IDs, strings and mask lengths disagree.")
    if any(mask not in (0, 1) for mask in masks) or records[0]["context"].get(
        "valid_token_count"
    ) != sum(masks):
        raise ReplayError("Language valid-token count differs from actual mask.")
    embeddings = select_records(capture, ["language"])
    norms = (
        dict(
            zip(
                embeddings[0]["outputs"]["tensor"]["token_indices"],
                embeddings[0]["outputs"]["tensor"]["token_norms"],
                strict=True,
            )
        )
        if embeddings
        else {}
    )
    return [
        {
            "position": i,
            "token_id": int(token),
            "token": language["tokens"][i],
            "valid": bool(masks[i]),
            "padding": not bool(masks[i]),
            "special_status": "requires source tokenizer metadata",
            "embedding L2": norms.get(i),
        }
        for i, token in enumerate(ids)
        if reveal_padding or masks[i]
    ]


def state_rows(capture: dict, spec: dict) -> list[dict]:
    records = select_records(capture, ["robot_state_normalization"])
    if not records:
        return []
    raw = records[0]["inputs"].get("raw_state", {}).get("values")
    normalized = records[0]["outputs"].get("normalized_state", {}).get("values")
    if raw is None or normalized is None:
        return []
    names = capture["metadata"]["action_names"]
    if names != spec["joint_names"] or len(raw) != len(normalized) or len(raw) != len(names):
        raise ReplayError("Joint semantics/state dimensions disagree.")
    units = spec.get("joint_units", [None] * len(names))
    return [
        {"joint": name, "raw measured": value, "normalized": normalized[i], "raw unit": units[i]}
        for i, (name, value) in enumerate(zip(names, raw, strict=True))
    ]


def prefix_rows(capture: dict, spec: dict) -> list[dict]:
    result, start = [], 0
    for source in spec.get("prefix_sources", []):
        records = select_records(capture, [source["stage"]], source.get("camera"))
        if not records:
            return []
        shape = records[0]["outputs"]["tensor"]["shape"]
        count = shape[-2] if len(shape) == 3 else 1
        result.append(
            {
                "source": source["label"],
                "start inclusive": start,
                "end exclusive": start + count,
                "positions": count,
            }
        )
        start += count
    prefix = select_records(capture, ["multimodal_prefix"])
    if not prefix or prefix[0]["inputs"]["tensor"]["shape"][-2] != start:
        raise ReplayError("Prefix source lengths do not match captured concatenation.")
    return result


def action_rows(capture: dict) -> list[dict]:
    names = capture["metadata"]["action_names"]
    return [
        {"action index": index, **dict(zip(names, row, strict=True))}
        for index, row in enumerate(capture["actions"])
    ]
