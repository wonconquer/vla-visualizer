"""Inspect recorded rollouts without starting a simulation."""

from vla_visualizer.ui.presentation import present

from pathlib import Path
import streamlit as st
from vla_visualizer.execution.jobs import read_json


def local_file(directory, name):
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Artifact path escapes its run directory.")
    return path


def render_rollout(directory: Path):
    record = read_json(directory / "rollout.json")
    outcomes = st.columns(3)
    outcomes[0].metric(
        "Policy inference", "Verified" if record["inference_success"] else "Not verified"
    )
    outcomes[1].metric(
        "Simulator stepping",
        "Verified" if record["simulator_integration_success"] else "Not verified",
    )
    outcomes[2].metric(
        "Task outcome",
        "Success"
        if record["task_success"] is True
        else "Not successful"
        if record["task_success"] is False
        else "Unavailable",
    )
    st.caption(
        "Stop reason: "
        + record["termination"]
        + " · reaching a step/time limit is truncation, not task success."
    )
    timing = st.columns(2)
    timing[0].metric(
        "Simulator time excluding settling",
        f"{record['simulator_seconds_excluding_warmup']:.3f} s"
        if "simulator_seconds_excluding_warmup" in record
        else "Not recorded",
    )
    timing[1].metric(
        "Rollout wall time",
        f"{record['wall_seconds']:.3f} s" if "wall_seconds" in record else "Running",
    )
    video = record.get("video")
    if video and local_file(directory, video["file"]).exists():
        st.video(str(local_file(directory, video["file"])))
        st.caption(
            "Saved video after execution · recorded simulator frame intervals · initial playback speed 1×. Browser playback speed is independent of wall time."
        )
    frames = record["frames"]
    if frames:
        position = (
            st.slider(
                "Display frame position", 0, len(frames) - 1, 0, key=f"frame_{directory.name}"
            )
            if len(frames) > 1
            else 0
        )
        frame = frames[position]
        st.image(
            str(local_file(directory, frame["file"])),
            caption=f"Simulator agentview · control step {frame['control_step']} · sim time {frame['simulator_time']:.3f} s",
            width="stretch",
        )
        st.caption(
            "Display position, control/physics step, inference request, and flow iteration are separate indices."
        )
    if not record["requests"]:
        return
    index = st.selectbox(
        "Inference request index", range(len(record["requests"])), key=f"request_{directory.name}"
    )
    request = record["requests"][index]
    st.caption(f"{request['id']} · observation control step {request['observation_control_step']}")
    cameras = st.columns(2)
    for panel, (name, filename) in zip(cameras, request["policy_input_frames"].items()):
        panel.image(
            str(local_file(directory, filename)),
            caption=f"Actual policy input · {name}",
            width="stretch",
        )
    rows = [
        {"action_index": i, **dict(zip(record["action_names"], row))}
        for i, row in enumerate(request["predicted_action_chunk"])
    ]
    st.line_chart(rows, x="action_index", y=list(record["action_names"]), height=200)
    st.write("Actually executed subset", request["executed"])
    st.caption(
        "7D relative OSC_POSE: translation/rotation and gripper. Controller scaling and applied actions are recorded in the details."
    )
    if request["capture_file"]:
        data = read_json(local_file(directory, request["capture_file"]))
        if data["run_id"] != record["run_id"] or data["inference_id"] != request["id"]:
            raise ValueError("Run/capture links do not match.")
        stages = data["capture"]["stages"]
        stage = st.selectbox(
            "Educational capture stage",
            range(len(stages)),
            format_func=lambda i: stages[i]["stage_name"],
            key=f"stage_{directory.name}_{index}",
        )
        event = stages[stage]
        st.write(present(event["description"]))
        st.caption(
            "Recorded detached summaries. Token norms are not attention, saliency, or confidence. Educational stage selection is separate from execution time."
        )
        for summary in event["outputs"].values():
            if summary.get("token_norms"):
                st.line_chart(
                    {"token_index": summary["token_indices"], "L2 norm": summary["token_norms"]},
                    x="token_index",
                    y="L2 norm",
                    height=160,
                )
        with st.expander("Recorded tensor statistics / module path"):
            st.json(present(event))
    else:
        st.info(
            "Not recorded for this request. Internal capture is limited to the first three requests."
        )
    with st.expander("State / normalization / step / time links"):
        st.json(request)
