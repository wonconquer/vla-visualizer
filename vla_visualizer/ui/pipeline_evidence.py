"""Connect glyphs to CPU capture records without inference or hook registration."""

from math import sqrt
from vla_visualizer.inference.replay import select_records, language_rows


def enrich(data, visuals, iteration=0):
    if not data:
        return visuals
    capture = data["capture"]
    run = data["manifest"]["run_id"]
    extended = capture["intermediates"].get("extended_capture", {})
    entries = extended.get("token_features", [])
    cameras = list(data["view_spec"]["camera_to_native"])
    sources = {
        "vision_top": ("vision", cameras[0], "outputs", "tensor"),
        "connector_top": ("connector", cameras[0], "outputs", "tensor"),
        "vision_wrist": ("vision", cameras[1], "outputs", "tensor"),
        "connector_wrist": ("connector", cameras[1], "outputs", "tensor"),
        "tokens": ("language_tokenization", None, "outputs", "token_ids"),
        "language": ("language", None, "outputs", "tensor"),
        "state": ("robot_state_normalization", None, "inputs", "raw_state"),
        "normalize": ("state", None, "inputs", "tensor"),
        "projection": ("state", None, "outputs", "tensor"),
        "prefix": ("multimodal_prefix", None, "inputs", "tensor"),
        "conditioning": ("time_conditioned_actions", None, "outputs", "tensor"),
        "expert": ("action_velocity", None, "inputs", "tensor"),
        "velocity": ("action_velocity", None, "outputs", "tensor"),
        "actions": ("action_postprocessing", None, "outputs", "action"),
    }
    stages = capture["intermediates"]["stages"]
    language = language_rows(capture, reveal_padding=True)
    for key, (stage, camera, side, name) in sources.items():
        flow = iteration if key in {"conditioning", "expert", "velocity"} else None
        records = select_records(capture, [stage], camera=camera, step=flow)
        if not records:
            continue
        record = records[0]
        summary = record[side].get(name, {})
        index = next(i for i, r in enumerate(stages) if r is record)
        feature_stage = "expert_hidden" if key == "expert" else stage
        feature = next(
            (
                f
                for f in entries
                if f["stage"] == feature_stage
                and f.get("camera") == camera
                and (flow is None or f.get("flow_step_index") == flow)
            ),
            None,
        )
        item = visuals.setdefault(key, {})
        item.update(
            mapping={
                "run_id": run,
                "stage": stage,
                "module": record["module_name"],
                "record_id": f"record_{index:02d}",
                "camera": camera,
                "modality": key,
                "flow_iteration": flow,
                "summary_method": "Each glyph shows one measured token/vector index and its full-channel L2 norm; preview displays the first 18 indices at most",
                "shape": summary.get("shape"),
                "dtype": summary.get("dtype"),
            },
            token_indices=summary.get("token_indices", []),
            norms=summary.get("token_norms", []),
        )
        if feature:
            item["features"] = feature
        if key.startswith(("vision_", "connector_")):
            layout = capture["intermediates"]["vision"].get(camera, {})
            grid = layout.get("connector_grid" if key.startswith("connector_") else "patch_grid")
            item["grid"] = (
                grid
                if grid
                and "Row-major" in layout.get("ordering", "")
                and layout.get("resize_geometry")
                and summary.get("token_indices") == list(range(grid[0] * grid[1]))
                else None
            )
            item["geometry"] = (
                layout.get("resize_geometry")
                if item["grid"] and "Row-major" in layout.get("ordering", "")
                else None
            )
            item["raw_image"] = (
                "data:image/png;base64," + data["images"][camera]["png_base64"]
                if camera in data.get("images", {})
                else None
            )
            measured = extended.get("processed_images", {}).get(camera)
            item["input_image"] = (
                "data:image/png;base64," + measured["png_base64"] if measured else None
            )
            item["input_image_source"] = (
                measured.get("source") if measured else "unavailable in legacy capture"
            )
        if key in {"language", "tokens"}:
            item["language_tokens"] = language
            if key == "tokens":
                item["features"] = next((f for f in entries if f["stage"] == "language"), None)
        if key == "state":
            item["coordinate_values"] = summary.get("values", [])
            item["coordinate_names"] = capture["metadata"]["action_names"]
            item["mapping"]["summary_method"] = "glyph = measured raw joint coordinate, not a token"
        if key == "normalize":
            item["coordinate_values"] = summary.get("values", [])
            item["mapping"]["summary_method"] = "glyph = padded state coordinate, not a token"
        if key == "projection" and not item["norms"]:
            item.update(norms=[summary["l2_norm"]], token_indices=[0])
        if key == "actions":
            item["full_vectors"] = capture["actions"]
            item["mapping"]["summary_method"] = (
                "glyph = final chunk action index; full six-dimensional joint target"
            )
    prefix = extended.get("auxiliary", {}).get("vlm_prefix_output")
    visuals["vlm_context"] = {
        "shape": str(prefix["summary"]["shape"]) if prefix else "Not recorded",
        "norms": prefix["summary"].get("token_norms", []) if prefix else [],
        "token_indices": prefix["summary"].get("token_indices", []) if prefix else [],
        "features": next((f for f in entries if f["stage"] == "vlm_prefix_output"), None),
        "mapping": {
            "run_id": run,
            "stage": "vlm_prefix_output",
            "record_id": "aux_prefix_output",
            "module": prefix["module"] if prefix else "not recorded",
            "summary_method": "Actual final text norm output, not a KV tensor",
        },
    }
    visuals["cache"] = {
        "text": "KV tensor not retained",
        "shape": str(
            extended.get("prefix_processing", {})
            .get("cache_metadata", {})
            .get("sequence_length", "Source-based description")
        )
        + " prefix slots",
    }
    loop = extended.get("full_loop", [])
    if loop:
        event = loop[iteration]
        for key, field in [
            ("initial_noise", "x_before"),
            ("action_state", "x_before"),
            ("velocity", "velocity"),
            ("solver_update", "x_after"),
        ]:
            actual_event = loop[0] if key == "initial_noise" else event
            actual_iteration = actual_event["iteration"]
            vectors = actual_event[field][0]
            visuals[key] = {
                "shape": str([len(vectors), len(vectors[0])]),
                "norms": [sqrt(sum(v * v for v in row)) for row in vectors],
                "token_indices": list(range(len(vectors))),
                "full_vectors": vectors,
                "mapping": {
                    "run_id": run,
                    "stage": field,
                    "record_id": f"loop_{actual_iteration}:{field}",
                    "module": "native denoise_step / euler_integrate",
                    "flow_iteration": actual_iteration,
                    "dtype": "torch.float32",
                    "shape": [1, len(vectors), len(vectors[0])],
                    "summary_method": "glyph = action chunk position; full recorded internal action vector",
                },
            }
        visuals["time_embedding"] = {
            "text": f"t={event['time']:.4f}; dt={event['dt']:.2f}",
            "mapping": {
                "run_id": run,
                "stage": "time",
                "record_id": f"loop_{iteration}:time",
                "flow_iteration": iteration,
                "summary_method": event["time_source"],
            },
        }
    return visuals
