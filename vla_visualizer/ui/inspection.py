"""Streamlit viewer of immutable CPU replay snapshots. Never calls inference/hooks."""

import base64
from io import BytesIO
from time import perf_counter, monotonic

from PIL import Image
import streamlit as st

from vla_visualizer.inference.replay import (
    ReplayRun,
    action_rows,
    language_rows,
    norm_rows,
    prefix_rows,
    select_records,
    state_rows,
)
from vla_visualizer.visualization.inspection import inspection_dot


def move_stage(amount: int = 0, *, reset: bool = False) -> None:
    ids = st.session_state["replay_stage_ids"]
    current = st.session_state.get("replay_stage", ids[0])
    st.session_state["replay_stage"] = (
        ids[0] if reset else ids[max(0, min(len(ids) - 1, ids.index(current) + amount))]
    )
    if reset:
        st.session_state["replay_playing"] = False
    st.session_state["replay_tick"] = monotonic()


def autoplay_tick(
    now: float, last: float, index: int, count: int, playing: bool
) -> tuple[int, bool, float]:
    """One explanatory UI step, no blocking loop or measured model timing."""
    if not playing or now - last < 1.5:
        return index, playing, last
    next_index = min(count - 1, index + 1)
    return next_index, next_index < count - 1, now


def _norm_chart(summary: dict, label: str) -> None:
    rows = norm_rows(summary)
    st.write(label)
    if rows:
        st.line_chart(rows, x="index", y="L2 norm", height=180)
    else:
        st.info("Indexed vector norms unavailable in this capture.")


def _record_details(records: list[dict]) -> None:
    if not records:
        st.warning("This stage is unavailable in the completed capture.")
        return
    for record in records:
        st.write(f"**{record['stage_name']}** · {record['source']}")
        st.code(record["module_name"], language=None)
        st.caption(record["description"])
        if record["context"]:
            st.write(record["context"])
        rows = []
        for side in ("inputs", "outputs"):
            for name, summary in record[side].items():
                rows.append(
                    {
                        "side": side,
                        "tensor": name,
                        "shape": str(summary["shape"]),
                        "original dtype": summary["dtype"],
                        "execution device": summary["device"],
                        **{
                            key: summary[key]
                            for key in ("minimum", "maximum", "mean", "std", "l2_norm")
                        },
                    }
                )
        st.dataframe(rows, hide_index=True)
        with st.expander(f"Compact values / diagnostics: {record['stage_name']}"):
            st.json(record, expanded=False)


def _vision(data: dict, stage: dict) -> None:
    capture, camera = data["capture"], stage["camera"]
    image = data.get("images", {}).get(camera)
    if image:
        st.image(
            Image.open(BytesIO(base64.b64decode(image["png_base64"]))),
            caption=f"Completed-run raw RGB · {camera}",
            width="stretch",
        )
    else:
        st.info("Original image pixels unavailable in this saved capture.")
    st.info(
        "Preprocessed image pixels were not retained. Their actual input shape/range/geometry are available; no reconstructed image is shown."
    )
    layout = capture["intermediates"].get("vision", {}).get(camera)
    if layout:
        st.write(layout)
    else:
        st.info("Vision geometry metadata unavailable; use token indices only.")
    for name in ("vision", "connector"):
        records = select_records(capture, [name], camera)
        if records:
            _norm_chart(records[0]["outputs"]["tensor"], f"{name}: indexed token L2 norms")
    st.caption(
        "Token magnitude is not attention, importance, saliency, confidence or localization. Pre/post connector norms belong to different embedding spaces and scales."
    )
    with st.expander("Spatial ordering evidence / limits"):
        st.write(data["view_spec"]["spatial_evidence"])


def _language(data: dict) -> None:
    st.write(data["manifest"]["instruction"] or "Instruction unavailable")
    reveal = st.checkbox("Reveal padding and special tokens", key="replay_padding")
    rows = language_rows(data["capture"], reveal_padding=reveal)
    if rows:
        special = data["view_spec"].get("verified_special_tokens", {})
        ranges = data["view_spec"].get("special_token_id_ranges")
        for row in rows:
            is_special = (
                any(start <= row["token_id"] <= end for start, end in ranges) if ranges else None
            )
            row["special_status"] = (
                "verified special / padding token"
                if special.get(str(row["token_id"])) == row["token"]
                else "verified special token"
                if is_special
                else "ordinary token"
                if is_special is False
                else "special status unavailable"
            )
        st.dataframe(rows, hide_index=True)
    else:
        st.info(
            "Complete token IDs/strings/mask unavailable; embedding summaries may still be inspected."
        )
    st.caption(
        "Default shows mask-valid tokens only. Padding is not meaningful instruction content. Embedding magnitude is not attention or causal action explanation."
    )


def _state(data: dict) -> None:
    capture, spec = data["capture"], data["view_spec"]
    rows = state_rows(capture, spec)
    if rows:
        st.dataframe(rows, hide_index=True)
    else:
        st.info("Full raw/normalized state values unavailable.")
    st.write(spec["state_pipeline"])
    projection = select_records(capture, ["state"])
    if projection:
        padded = projection[0]["inputs"]["tensor"]
        normalized = select_records(capture, ["robot_state_normalization"])
        values = padded.get("values")
        count = len(rows)
        if values is not None and normalized and count:
            expected = normalized[0]["outputs"]["normalized_state"]["values"]
            st.write(
                {
                    "padding_verified_from_captured_values": values[:count] == expected
                    and all(v == 0 for v in values[count:]),
                    "trailing_padding_components": len(values) - count,
                }
            )
        else:
            st.info("Padding values unavailable; no numeric padding verification claimed.")
    st.caption(
        "Batch insertion before normalization is source-verified; the unnormalized batched tensor was not separately retained. The projected embedding is learned; its dimensions have no individual joint labels."
    )


def _prefix(data: dict) -> None:
    rows = prefix_rows(data["capture"], data["view_spec"])
    if rows:
        st.dataframe(rows, hide_index=True)
    else:
        st.info("Prefix composition unavailable because source captures are incomplete.")
    spec = data["view_spec"]
    st.write(
        {
            "image_special_tokens": spec["prefix_special_tokens"],
            "minimum_prefix_padding_length": spec["prefix_min_length"],
        }
    )
    st.write(spec["mask_explanation"])
    st.caption(
        "Composition is derived from source order and measured component lengths. Full prefix masks/KV cache were not retained. State enters the prefix; time/action conditioning enters a separate expert suffix."
    )


def _flow(data: dict, step: int | None) -> None:
    st.write(f"Captured flow iteration: {step}")
    st.caption(data["view_spec"]["time_semantics"])
    records = select_records(
        data["capture"], ["flow_state", "time_conditioned_actions", "action_velocity"], step=step
    )
    for record in records:
        if record["stage_name"] == "flow_state":
            _norm_chart(record["inputs"]["tensor"], "Captured x_t: per-action-index vector norms")
        if record["stage_name"] == "action_velocity":
            _norm_chart(record["inputs"]["tensor"], "Actual expert hidden representation norms")
            _norm_chart(record["outputs"]["tensor"], "Predicted velocity vector norms")
    st.info(
        "Only compact x_t / time-conditioned / expert-hidden / velocity summaries are retained. The separate sinusoidal time input, full intermediate action trajectories and post-update x_t are unavailable. Uncaptured iterations are not reconstructed."
    )
    st.caption(
        "Flow iteration, derived flow time, action index and replay stage are distinct. Norms need not change monotonically; this is frozen generation, not training."
    )


def _actions(data: dict, index: int) -> None:
    capture, spec = data["capture"], data["view_spec"]
    rows = action_rows(capture)
    st.line_chart(rows, x="action index", y=capture["metadata"]["action_names"], height=240)
    names, vector = capture["metadata"]["action_names"], capture["actions"][index]
    units = spec.get("joint_units", [None] * len(names))
    st.dataframe(
        [
            {"component": name, "value": value, "unit": units[i]}
            for i, (name, value) in enumerate(zip(names, vector, strict=True))
        ],
        hide_index=True,
    )
    st.write(
        {
            "selected_action_index": index,
            "native_chunk_shape": capture["metadata"]["action_chunk_shape"],
            "all_values_finite": True,
            "representation": spec["action_semantics"],
        }
    )
    st.caption(
        spec["coordinate_convention"]
        + " No action timestamps are asserted. Predictions do not establish executed motion or robotics task success."
    )


def render_inspection(run: ReplayRun) -> None:
    """Dynamic fragment timer runs only while playing; all controls are replay-only."""
    playing = st.session_state.get("replay_playing", False)
    st.session_state["replay_timer_enabled"] = playing
    st.fragment(run_every=1.5 if playing else None)(_render_body)(run)


def _render_body(run: ReplayRun) -> None:
    start = perf_counter()
    data = run.read()
    manifest, spec, capture = data["manifest"], data["view_spec"], data["capture"]
    ids = [stage["id"] for stage in spec["stages"]]
    if st.session_state.get("replay_run_id") != manifest["run_id"]:
        st.session_state.update(
            replay_run_id=manifest["run_id"],
            replay_stage=ids[0],
            replay_playing=False,
            replay_flow=next(iter(capture["intermediates"].get("selected_flow_steps", [])), None),
            replay_action=0,
            replay_tick=monotonic(),
        )
    st.session_state["replay_stage_ids"] = ids
    stage_index, playing, tick = autoplay_tick(
        monotonic(),
        st.session_state.get("replay_tick", monotonic()),
        ids.index(st.session_state["replay_stage"]),
        len(ids),
        st.session_state.get("replay_playing", False),
    )
    st.session_state.update(replay_stage=ids[stage_index], replay_tick=tick)
    stopped = st.session_state.get("replay_playing", False) and not playing
    st.session_state["replay_playing"] = playing
    st.subheader("Captured inference replay")
    st.write(f"Completed run: `{manifest['run_id']}` · {manifest['origin']}")
    st.caption(
        f"{manifest['checkpoint']} @ {manifest['revision']} · {manifest['sample_identity']} · actual seed {manifest['seed']} · execution device {manifest['device']}"
    )
    st.write(
        f"Measured inference: {capture['metadata']['inference_seconds']:.3f} s. Replay duration: explanatory 1.5 s/stage, not measured GPU execution."
    )
    st.caption(
        "Data dependencies are graph edges. Record order is a limited capture event order. Stage order below is an educational grouping, not an execution timeline."
    )
    controls = st.columns(4)
    controls[0].button(
        "Previous stage",
        key="replay_previous",
        on_click=move_stage,
        args=(-1,),
        disabled=stage_index == 0,
    )
    controls[1].button(
        "Next stage",
        key="replay_next",
        on_click=move_stage,
        args=(1,),
        disabled=stage_index == len(ids) - 1,
    )
    controls[2].button(
        "Reset replay", key="replay_reset", on_click=move_stage, kwargs={"reset": True}
    )
    if controls[3].button("Pause" if playing else "Play", key="replay_play"):
        st.session_state["replay_playing"] = not playing
        st.session_state["replay_tick"] = monotonic()
        st.rerun()
    stage_id = st.selectbox(
        "Inspection stage",
        ids,
        format_func=lambda key: next(s["label"] for s in spec["stages"] if s["id"] == key),
        key="replay_stage",
    )
    steps = sorted(
        {
            record["context"]["flow_step_index"]
            for record in capture["intermediates"]["stages"]
            if "flow_step_index" in record["context"]
        }
    )
    if steps:
        if st.session_state.get("replay_flow") not in steps:
            st.session_state["replay_flow"] = steps[0]
        step = st.selectbox("Captured flow iteration", steps, key="replay_flow")
    else:
        step = None
        st.info("Captured flow iterations unavailable.")
    index = st.slider("Action index", 0, len(capture["actions"]) - 1, key="replay_action")
    stage = next(s for s in spec["stages"] if s["id"] == stage_id)
    graph, detail = st.columns([1, 1.5])
    with graph:
        st.graphviz_chart(inspection_dot(spec, stage_id), width="stretch")
        st.info(
            "Genuine attention probabilities are not returned by the verified inference path. No substitute attention heatmap is shown."
        )
    with detail:
        st.subheader(stage["label"])
        st.write(stage["explanation"])
        st.caption(
            "Statistics summarize actual captures. Original execution device is shown separately; all replay storage is CPU serialized values."
        )
        if stage["kind"] == "vision":
            _vision(data, stage)
        elif stage["kind"] == "language":
            _language(data)
        elif stage["kind"] == "state":
            _state(data)
        elif stage["kind"] == "prefix":
            _prefix(data)
        elif stage["kind"] == "flow":
            _flow(data, step)
        elif stage["kind"] == "actions":
            _actions(data, index)
        records = select_records(
            capture,
            stage["records"],
            stage.get("camera"),
            step if stage["kind"] == "flow" else None,
        )
        _record_details(records)
    with st.expander("Run manifest / capture event order / diagnostics"):
        st.json(manifest, expanded=False)
        st.dataframe(manifest["records"], hide_index=True)
        st.caption(spec["summary_method"])
    st.download_button(
        "Download completed replay (JSON)",
        run.snapshot,
        file_name=f"replay_{manifest['run_id']}.json",
        mime="application/json",
        key="replay_download",
    )
    elapsed = perf_counter() - start
    st.session_state["replay_render_seconds"] = elapsed
    st.caption(
        f"Server-side view construction: {elapsed:.3f} s · immutable payload {len(run.snapshot.encode()):,} bytes. Browser chart/layout time is not included."
    )
    if (
        stopped
        or st.session_state.get("replay_timer_enabled", False) != st.session_state["replay_playing"]
    ):
        st.rerun()
