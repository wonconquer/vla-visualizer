"""Render captured summaries without running the model or reconstructing tensors."""

from vla_visualizer.ui.presentation import present

import base64
from html import escape
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import streamlit as st

from vla_visualizer.inference.replay import language_rows, prefix_rows, select_records, state_rows

STAGES = {
    "top_vision": "Top · vision encoder",
    "top_connector": "Top · visual connector",
    "wrist_vision": "Wrist · vision encoder",
    "wrist_connector": "Wrist · visual connector",
    "language_tokens": "Instruction · tokenizer",
    "language_embedding": "Language embeddings",
    "state_normalize": "Robot state · normalization / padding",
    "state_project": "State projection",
    "prefix": "Prefix composition · first text-layer input boundary",
    "vlm_context": "VLM prefix processing → cached context",
    "initial_noise": "Initial noise",
    "action_state": "Current action sample xₜ",
    "time_embedding": "Flow time / sin-cos embedding",
    "conditioning": "Action / time conditioning",
    "expert": "Action expert hidden",
    "velocity": "Flow velocity",
    "solver_update": "Native Euler state update",
    "loop_next": "Return to the next iteration",
    "actions": "Final action chunk",
}


def shape(summary):
    return (
        "[" + ", ".join(map(str, summary["shape"])) + "]" if "shape" in summary else "unavailable"
    )


def summary_for(capture, stage, side="outputs", name="tensor", camera=None, step=None):
    records = select_records(capture, [stage], camera=camera, step=step)
    return records[0].get(side, {}).get(name, {}) if records else {}


def thumb(data, camera):
    image = data.get("images", {}).get(camera)
    return "data:image/png;base64," + image["png_base64"] if image else None


def pipeline_visuals(data, instruction, flow_step=0):
    """Describe unavailable stages without inventing numerical representations."""
    if data is None:
        return {
            "instruction": {"text": instruction},
            "integration": {"text": "Iterative updates at the selected time"},
        }
    capture = data["capture"]
    cameras = list(data["view_spec"]["camera_to_native"])
    result = {
        "instruction": {"text": data["manifest"]["instruction"]},
        "integration": {"text": "Euler · inference iterations"},
    }
    for index, camera in enumerate(cameras[:2]):
        label = "top" if index == 0 else "wrist"
        result["image_" + label] = {"image": thumb(data, camera)}
        for stage in ("vision", "connector"):
            value = summary_for(capture, stage, camera=camera)
            result[stage + "_" + label] = {
                "shape": shape(value),
                "norms": value.get("token_norms", []),
            }
    for key, stage, side, name in (
        ("tokens", "language_tokenization", "outputs", "token_ids"),
        ("language", "language", "outputs", "tensor"),
        ("state", "robot_state_normalization", "inputs", "raw_state"),
        ("normalize", "state", "inputs", "tensor"),
        ("projection", "state", "outputs", "tensor"),
        ("prefix", "multimodal_prefix", "inputs", "tensor"),
        ("conditioning", "time_conditioned_actions", "outputs", "tensor"),
        ("expert", "action_velocity", "inputs", "tensor"),
        ("velocity", "action_velocity", "outputs", "tensor"),
        ("actions", "action_postprocessing", "outputs", "action"),
    ):
        value = summary_for(
            capture,
            stage,
            side,
            name,
            step=flow_step if key in {"conditioning", "expert", "velocity"} else None,
        )
        result[key] = {"shape": shape(value), "norms": value.get("token_norms", [])}
    from vla_visualizer.ui.pipeline_evidence import enrich

    return present(enrich(data, result, flow_step))


def norm_grid_html(summary, layout, connector=False):
    """Build a grid only when row-major geometry and all token indices are verified."""
    grid = layout.get("connector_grid" if connector else "patch_grid")
    if (
        not grid
        or "Row-major" not in layout.get("ordering", "")
        or not layout.get("resize_geometry")
    ):
        return None
    height, width = grid
    values = summary.get("token_norms", [])
    if height * width != len(values) or summary.get("token_indices") != list(range(height * width)):
        return None
    lo, hi = min(values), max(values)
    cells = []
    for index, value in enumerate(values):
        fraction = (value - lo) / (hi - lo) if hi > lo else 0.5
        color = f"hsl({230 - fraction * 45:.2f},65%,{95 - fraction * 49:.2f}%)"
        cells.append(
            f'<span title="token {index} · L2 {value:.5f}" style="background:{color};aspect-ratio:1;border-radius:1px"></span>'
        )
    return (
        f'<div style="display:grid;grid-template-columns:repeat({width},1fr);gap:2px;max-width:440px">'
        + "".join(cells)
        + f"</div><small>L2 {lo:.3f} → {hi:.3f} · measured {height}×{width} token norm</small>"
    )


def render_norm(summary, label):
    if not summary:
        st.info("Not recorded in this capture.")
        return
    st.caption(f"{label} · Measured shape {shape(summary)}")
    if summary.get("token_norms"):
        st.line_chart(
            {"token index": summary["token_indices"], "L2 norm": summary["token_norms"]},
            x="token index",
            y="L2 norm",
            height=160,
        )
    cols = st.columns(3)
    for col, field, title in zip(
        cols, ("mean", "std", "l2_norm"), ("Mean", "Standard deviation", "Overall L2")
    ):
        col.metric(title, f"{summary[field]:.4f}")


def render_stage(data, stage, flow_step=0, action_index=0):
    st.subheader(STAGES[stage])
    if data is None:
        explain = {
            "top_vision": "The top image is resized and padded, then encoded into contextual patch tokens.",
            "top_connector": "Neighboring patches are grouped and projected to the language-model embedding width.",
            "wrist_vision": "The wrist image passes separately through the same vision encoder.",
            "wrist_connector": "Wrist visual tokens occupy a separate prefix segment from top-camera tokens.",
            "language_tokens": "The native tokenizer receives the selected sentence followed by a newline and produces IDs and a padding mask.",
            "language_embedding": "Token IDs, including padding slots, map to embedding vectors. The mask identifies valid positions.",
            "state_normalize": "Measured joint state is normalized with checkpoint statistics and zero-padded to the model width.",
            "state_project": "Normalized, padded state is projected into one embedding. Its channels are not individual joints.",
            "prefix": "Two camera token sequences, language slots, and the state token are concatenated in the native order.",
            "conditioning": "The current xₜ is combined with time conditioning. Cached prefix context is reused across iterations.",
            "expert": "The action expert processes the action/time suffix conditioned on cached prefix context.",
            "velocity": "Output projection predicts flow velocity. Euler integration updates the action sample.",
            "actions": "After the final update, the chunk is sliced to robot dimensions and inverse-normalized using native statistics.",
            "vlm_context": "The VLM processes the prefix into cached context for the expert. The first-layer input, final norm output, and KV metadata are distinct boundaries.",
            "initial_noise": "Initial x₁ noise from native inference. New full-loop captures record the actual values for the selected seed.",
            "action_state": "The actual input xₜ at this iteration. Internal generation width differs from the robot action width.",
            "time_embedding": "t=1+k·(−1/N). Sin/cos time embedding is concatenated with action projection and passed through the MLP.",
            "solver_update": "Native Euler uses dt=−1/N and x←x+dt·v. Slicing and postprocessing occur only after the last update.",
            "loop_next": "The updated x returns to action/time conditioning. Prefix computation is not repeated in each iteration.",
        }
        st.write(explain[stage])
        st.info(
            "Conceptual structure. Open a recorded capture or explicitly run inference to display measured values."
        )
        return
    capture, spec = data["capture"], data["view_spec"]
    records = []
    extended = capture["intermediates"].get("extended_capture", {})
    if stage in {"initial_noise", "action_state", "time_embedding", "solver_update", "loop_next"}:
        loop = extended.get("full_loop", [])
        if not loop:
            st.info(
                "This legacy capture has detailed summaries only at 0/4/9. Missing x/time/update values are not interpolated."
            )
            st.latex(r"\Delta t=-1/N,\quad t_k=1+k\Delta t,\quad x_{k+1}=x_k+\Delta t\,v_k")
            return
        event = loop[0 if stage == "initial_noise" else flow_step]
        st.write(
            f"Flow iteration {event['iteration']}/{len(loop) - 1} · measured input time {event['time']:.8f} · dt={event['dt']:.3f} (source-derived)"
        )
        if stage == "time_embedding":
            st.write(
                "Measured timestep → sin/cos embedding → concatenate with action projection → MLP(in) → SiLU → MLP(out). Velocity output is not a time-conditioning input."
            )
        else:
            field = "x_after" if stage in {"solver_update", "loop_next"} else "x_before"
            values = event[field][0]
            st.line_chart(
                {f"channel {i}": [row[i] for row in values] for i in range(min(6, len(values[0])))},
                height=180,
            )
            st.caption(
                f"Measured internal shape [{len(values)},{len(values[0])}]; showing channels 0–5. Hover shows recorded values. These are not native robot units."
            )
            st.caption(event.get(field + "_source", "actual trace"))
        st.caption(
            "Flow iteration, action index, simulator control step, optimizer step, and educational playback position are separate indices."
        )
        if stage in {"solver_update", "loop_next"}:
            st.caption(
                f"Next schedule time {1 + (event['iteration'] + 1) * event['dt']:.6f} · derived from the native schedule, not a separate activation measurement"
            )
        return
    if stage == "vlm_context":
        measured = extended.get("auxiliary", {}).get("vlm_prefix_output")
        if measured:
            render_norm(measured["summary"], "Measured final prefix output at VLM text_model.norm")
            st.caption("Measured at the final norm boundary. Full KV tensors are not retained.")
        else:
            st.info(
                "Legacy multimodal_prefix is the first text-layer input_layernorm boundary. Final VLM hidden/KV values were not recorded."
            )
        prefix = extended.get("prefix_processing", {})
        if prefix:
            st.write("Recorded cached-context metadata:", prefix.get("cache_metadata"))
            st.write(
                "Can image/language queries see the state token?",
                prefix.get("image_language_queries_can_see_state"),
            )
            st.caption(
                "The state query can see valid prefix keys. The block mask prevents image/language queries from seeing state/actions. A mask is not an attention probability."
            )
        return
    if stage.startswith(("top_", "wrist_")):
        camera = list(spec["camera_to_native"])[0 if stage.startswith("top_") else 1]
        connector = stage.endswith("connector")
        source = "connector" if connector else "vision"
        records = select_records(capture, [source], camera)
        left, right = st.columns([1, 1.1])
        with left:
            encoded = data.get("images", {}).get(camera)
            if encoded:
                st.image(
                    Image.open(BytesIO(base64.b64decode(encoded["png_base64"]))),
                    caption="Original policy-camera observation from this run",
                    width="stretch",
                )
            st.caption(
                "The grid maps tokens to the resized/padded policy-input canvas, not saliency on the original photo."
            )
        with right:
            value = summary_for(capture, source, camera=camera)
            layout = capture["intermediates"].get("vision", {}).get(camera, {})
            grid = norm_grid_html(value, layout, connector)
            st.caption(f"Measured {source} output {shape(value)}")
            if grid:
                st.html(grid)
            else:
                render_norm(value, "Spatial mapping not recorded · norms by index")
            st.caption("Measured token norms, not attention, saliency, or confidence.")
            geometry = layout.get("resize_geometry", {})
            st.caption(
                f"Verified geometry: source {geometry.get('source_hw')} → resize {geometry.get('resized_hw')}; top/left padding {geometry.get('padding_top_left')}"
            )
    elif stage.startswith("language"):
        st.write("Actual input instruction:", data["manifest"]["instruction"])
        rows = language_rows(capture, reveal_padding=True)
        chips = []
        for row in rows:
            background, color = ("#e5e9ff", "#394b91") if row["valid"] else ("#eff1f4", "#8c96a5")
            chips.append(
                f'<span title="position {row["position"]}, ID {row["token_id"]}, {"valid" if row["valid"] else "padding"}" style="display:inline-block;margin:3px;padding:6px 9px;border-radius:7px;background:{background};color:{color};font:13px monospace">{escape(row["token"])}<small> · {row["token_id"]}</small></span>'
            )
        st.html("".join(chips))
        st.caption(
            f"Valid {sum(r['valid'] for r in rows)} / total {len(rows)} slots · gray denotes padding identified by the recorded mask"
        )
        records = select_records(capture, ["language_tokenization", "language"])
        if stage == "language_embedding":
            render_norm(summary_for(capture, "language"), "Measured language embedding")
        else:
            st.caption(
                "Measured tokenizer IDs/mask shape: "
                + shape(summary_for(capture, "language_tokenization", name="token_ids"))
            )
    elif stage.startswith("state"):
        rows = state_rows(capture, spec)
        for row in rows:
            joint, raw, norm = st.columns([1.3, 1, 2])
            joint.write(row["joint"])
            raw.write(f"{row['raw measured']:.3f}")
            norm.progress(
                min(
                    1.0, abs(row["normalized"]) / max(1.0, max(abs(r["normalized"]) for r in rows))
                ),
                text=f"normalized {row['normalized']:+.4f}",
            )
        st.caption(
            "Raw units: first five joints use legacy calibrated degrees; gripper uses calibrated percent. Normalized values are dimensionless."
        )
        padded = summary_for(capture, "state", side="inputs")
        if padded.get("values"):
            n = len(rows)
            st.html(
                '<div style="display:flex;flex-wrap:wrap;gap:4px">'
                + "".join(
                    f'<span title="state slot {i}" style="padding:5px;background:{"#e5e9ff" if i < n else "#edf0f4"};font:11px monospace">{v:+.2f}</span>'
                    for i, v in enumerate(padded["values"])
                )
                + "</div>"
            )
            st.caption(
                f"Measured projection input {shape(padded)} · first {n} normalized state coordinates, followed by native zero padding"
            )
        records = select_records(capture, ["robot_state_normalization", "state"])
        if stage == "state_project":
            render_norm(summary_for(capture, "state"), "One projected state embedding")
            st.info(
                "Projected embedding channels do not correspond one-to-one to individual joints."
            )
    elif stage == "prefix":
        rows = prefix_rows(capture, spec)
        masks = [row["valid"] for row in language_rows(capture, reveal_padding=True)]
        colors = ["#6d79d4", "#9a88d4", "#54a896", "#dbab58"]
        blocks = []
        for i, row in enumerate(rows):
            valid = masks if row["source"].startswith("language") else [True] * row["positions"]
            slots = "".join(
                f'<span title="prefix slot {row["start inclusive"] + j}; {"valid" if flag else "masked padding"}" style="width:8px;height:16px;display:inline-block;margin:1px;background:{colors[i % 4] if flag else "#e3e6ed"}"></span>'
                for j, flag in enumerate(valid)
            )
            blocks.append(
                f'<div style="padding:10px;border-left:5px solid {colors[i % 4]};margin:5px 0"><b>{escape(row["source"])}</b> · [{row["start inclusive"]}, {row["end exclusive"]}) · {row["positions"]} slots<br>{slots}</div>'
            )
        st.html("".join(blocks))
        value = summary_for(capture, "multimodal_prefix", side="inputs")
        st.caption(
            f"Measured prefix shape {shape(value)} · length is computed from recorded camera/language/state lengths."
        )
        st.write("The expert reuses cached context computed from this prefix across iterations.")
        st.caption(
            "Language masks are measured. Present-camera/state validity is derived from verified source behavior; masks are not attention probabilities."
        )
        records = select_records(capture, ["multimodal_prefix"])
    elif stage in {"conditioning", "expert", "velocity"}:
        names = (
            ["flow_state", "time_conditioned_actions"]
            if stage == "conditioning"
            else ["action_velocity"]
        )
        records = select_records(capture, names, step=flow_step)
        st.caption(f"Recorded flow iteration {flow_step} · separate from final action index.")
        if stage == "conditioning":
            render_norm(
                summary_for(capture, "flow_state", side="inputs", step=flow_step),
                "Current xₜ · padded normalized action space",
            )
            render_norm(
                summary_for(capture, "time_conditioned_actions", step=flow_step),
                "Measured action/time-conditioned suffix",
            )
        else:
            render_norm(
                summary_for(
                    capture,
                    "action_velocity",
                    side="inputs" if stage == "expert" else "outputs",
                    step=flow_step,
                ),
                "Measured expert hidden"
                if stage == "expert"
                else "Measured predicted flow velocity vₜ",
            )
        st.caption(
            "Velocity is a change in normalized flow space, not a physical joint velocity or simulator time."
        )
        if extended.get("full_loop"):
            st.info(
                "See Euler state update for recorded x←x+dt·v. Large hidden summaries exist only at 0/4/9; other hidden values were not recorded."
            )
        else:
            st.info(
                "Updated x and intermediate iterations were not recorded in this legacy capture. No interpolation is used."
            )
    elif stage == "actions":
        names = capture["metadata"]["action_names"]
        rows = [
            {"action index": i, **dict(zip(names, row))} for i, row in enumerate(capture["actions"])
        ]
        st.line_chart(rows, x="action index", y=names, height=230)
        cols = st.columns(3)
        for i, (name, value) in enumerate(zip(names, capture["actions"][action_index])):
            cols[i % 3].metric(name, f"{value:.3f}", help=spec["joint_units"][i])
        st.caption(
            f"Selected action index {action_index} · chunk {capture['metadata']['action_chunk_shape']} · absolute joint targets, not executed motion."
        )
        st.caption(
            "First five joints: legacy calibrated degrees. Gripper: calibrated percent (nominal 0–100)."
        )
        records = select_records(capture, ["normalized_action_chunk", "action_postprocessing"])
    with st.expander("Developer details · record / module / tensor statistics"):
        st.json(present(records), expanded=False)


def component_preview_html(args):
    template = (Path(__file__).parent / "pipeline_component/index.html").read_text()
    payload = json.dumps(args, ensure_ascii=False).replace("</", "<\\/")
    return template.replace(
        "<script>", "<script>window.PIPELINE_PREVIEW=" + payload + ";</script><script>", 1
    )
