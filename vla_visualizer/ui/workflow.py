"""Explore SmolVLA information flow; model execution uses explicit jobs."""

from copy import deepcopy

from hashlib import sha256
from io import BytesIO
import base64
from pathlib import Path
from time import monotonic
import uuid

import streamlit as st
import streamlit.components.v1 as components

from vla_visualizer.adapters.education_catalog import INSTRUCTION_EXAMPLES
from vla_visualizer.adapters.registry import list_models
from vla_visualizer.config import ROOT, sample_path
from vla_visualizer.execution.jobs import JobConfig, JobManager, TERMINAL, read_json
from vla_visualizer.inference.observations import read_observation
from vla_visualizer.inference.replay import load_run, make_run
from vla_visualizer.ui.education_views import STAGES, pipeline_visuals, render_stage
from vla_visualizer.ui.education_training import TRAIN_STAGES, render_training
from vla_visualizer.ui.execution_flow import call_events, iterations, step_position
from vla_visualizer.ui.education_state import ensure_selection, widget_value, commit_widget

MODES = {
    "Explore / Inference": "baseline",
    "Training": "training",
    "Simulation": "simulation",
}
_pipeline = components.declare_component(
    "smolvla_pipeline", path=str(Path(__file__).parent / "pipeline_component")
)


@st.cache_resource
def manager():
    return JobManager()


def local_file(directory, name):
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Artifact path escapes its run directory.")
    return path


@st.cache_data
def saved_capture():
    from vla_visualizer.adapters.smolvla import SmolVLAAdapter

    capture = read_json(ROOT / "assets/education/reference_capture.json")
    manifest = read_json(sample_path())
    for camera, filename in manifest["images"].items():
        expected = capture["metadata"]["observation_provenance"]["image_sha256_by_camera"][camera]
        if sha256((sample_path().parent / filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Camera checksum differs from the captured observation.")
    return make_run(
        capture,
        SmolVLAAdapter().get_inspection_spec(),
        read_observation(sample_path()),
        origin="recorded reference capture; no new inference",
    )


def read_training(directory):
    return {
        "config": read_json(directory / "config.json")
        if (directory / "config.json").exists()
        else {},
        "audit": read_json(directory / "training_audit.json"),
        "metrics": read_json(directory / "metrics.json"),
        "result": read_json(directory / "result.json"),
        "visualization": read_json(directory / "training_visualization.json")
        if (directory / "training_visualization.json").exists()
        else {},
    }


@st.cache_data(max_entries=8)
def dataset_observation(episode, frame):
    from vla_visualizer.data.so100 import observation

    return observation(episode, frame)


def training_observation(record):
    """Reopen the recorded training input, never substitute the current UI selection."""
    audit = record.get("audit", {})
    frame = audit.get("observation_frame")
    episode = audit.get("observation_episode", 0)
    if type(frame) is not int or type(episode) is not int:
        raise ValueError("Training input frame identity was not recorded.")
    if (
        episode == 0
        and frame == 0
        and record.get("config", {}).get("observation_mode") != "dataset"
    ):
        observation = read_observation(sample_path())
    else:
        observation = dataset_observation(episode, frame)
        if audit.get("parquet_sha256") != observation.provenance.get("parquet_sha256"):
            raise ValueError("Training input checksum differs from the saved audit.")
    if audit.get("recorded_instruction", observation.instruction) != observation.instruction:
        raise ValueError("Training input instruction differs from the saved audit.")
    return observation


def move_stage(offset=0, reset=False):
    ids = ensure_selection(
        st.session_state,
        st.session_state.get("education_stage_ids", []),
        st.session_state.get("edu_iterations"),
        st.session_state.get("edu_calls"),
        action_count=st.session_state.get("edu_action_count", 1),
        training_steps=st.session_state.get("edu_training_steps", 1),
    )
    if not ids:
        return
    current = st.session_state.get("edu_stage", ids[0])
    if reset:
        st.session_state.update(edu_stage=ids[0], edu_flow=0, edu_call=0)
    elif st.session_state.get("edu_view") == "Recorded invocation order" and st.session_state.get(
        "edu_calls"
    ):
        events = st.session_state["edu_calls"]
        i = max(0, min(len(events) - 1, st.session_state.get("edu_call", 0) + offset))
        event = events[i]
        st.session_state.update(edu_call=i, edu_stage=event["stage"])
        if event["iteration"] is not None:
            st.session_state["edu_flow"] = event["iteration"]
    elif "action_state" in ids:
        stage, iteration = step_position(
            current,
            st.session_state.get("edu_flow", 0),
            ids,
            st.session_state.get("edu_iterations", [0]),
            offset,
        )
        st.session_state.update(edu_stage=stage, edu_flow=iteration)
    else:
        st.session_state["edu_stage"] = ids[max(0, min(len(ids) - 1, ids.index(current) + offset))]
    if reset:
        st.session_state["edu_playing"] = False
        st.session_state["edu_pause_reason"] = "Reset button"
    st.session_state["edu_tick"] = monotonic()


def advance(now, last, index, count, playing, interval):
    if not playing or now - last < interval:
        return index, playing, last
    index = min(index + 1, count - 1)
    return index, index < count - 1, now


def pause_selection():
    st.session_state.update(
        edu_playing=False, edu_tick=monotonic(), edu_pause_reason="Selection changed"
    )


def commit_selection(key):
    return commit_widget(st.session_state, key)


def change_view():
    commit_selection("edu_view")
    pause_selection()
    events = st.session_state.get("edu_calls", [])
    if st.session_state.get("edu_view") == "Recorded invocation order" and events:
        event = events[min(len(events) - 1, max(0, st.session_state.get("edu_call", 0)))]
        st.session_state["edu_stage"] = event["stage"]
        if event["iteration"] is not None:
            st.session_state["edu_flow"] = event["iteration"]


def visualization_identity(data, kind, training=None):
    return (
        ("training:" + training["result"]["run_id"])
        if training
        else data["manifest"]["run_id"]
        if data
        else "concept:" + kind
    )


def explorer(data, instruction, kind, training=None):
    identity = visualization_identity(data, kind, training)
    ids = list(TRAIN_STAGES if kind == "training" else STAGES)
    ensure_selection(
        st.session_state,
        ids,
        iterations(data),
        call_events(data),
        identity,
        len(data["capture"]["actions"]) if data else 1,
        len(training["metrics"]["records"]) if training else 1,
    )
    st.session_state["edu_action_count"] = len(data["capture"]["actions"]) if data else 1
    st.session_state["edu_training_steps"] = len(training["metrics"]["records"]) if training else 1
    components_body = st.fragment(
        run_every=st.session_state.get("edu_speed", 1.5)
        if st.session_state.get("edu_playing")
        else None
    )(_explorer_body)
    components_body(data, instruction, kind, training, st.session_state.get("edu_speed", 1.5))


def _explorer_body(data, instruction, kind, training, scheduled_speed):
    if st.session_state.get("education_identity") != visualization_identity(data, kind, training):
        st.caption("This scheduled playback has been superseded by the selected run or mode.")
        return
    labels = TRAIN_STAGES if kind == "training" else STAGES
    ids = ensure_selection(
        st.session_state,
        list(labels),
        iterations(data),
        call_events(data),
        action_count=len(data["capture"]["actions"]) if data else 1,
        training_steps=len(training["metrics"]["records"]) if training else 1,
    )
    if not ids:
        st.info(
            "No educational stages are available for this record. The current capture is preserved."
        )
        return
    if kind != "training":
        st.radio(
            "Pipeline view",
            ["Information flow", "Recorded invocation order"],
            horizontal=True,
            key=widget_value(st.session_state, "edu_view"),
            on_change=change_view,
        )
    calls = (
        st.session_state.get("edu_calls", [])
        if kind != "training" and st.session_state.get("edu_view") == "Recorded invocation order"
        else []
    )

    def at_end():
        return (
            st.session_state.get("edu_call", 0) == len(calls) - 1
            if calls
            else st.session_state["edu_stage"] == ids[-1]
        )

    was_playing = st.session_state.get("edu_playing", False)
    if was_playing and monotonic() - st.session_state.get(
        "edu_tick", monotonic()
    ) >= st.session_state.get("edu_speed", 1.5):
        move_stage(1)
        if at_end():
            st.session_state["edu_playing"] = False
            st.session_state["edu_pause_reason"] = "Reached the final stage"
    playing = st.session_state.get("edu_playing", False)
    stopped = was_playing and not playing
    index = ids.index(st.session_state["edu_stage"])
    controls = st.columns([1, 1, 1, 1, 1, 1.6])
    controls[0].button(
        "Previous", key="edu_previous", on_click=move_stage, args=(-1,), disabled=index == 0
    )
    controls[1].button("Next", key="edu_next", on_click=move_stage, args=(1,), disabled=at_end())
    if controls[2].button("Play", key="edu_play", disabled=playing or at_end()):
        st.session_state.update(edu_playing=True, edu_tick=monotonic(), edu_pause_reason=None)
        st.rerun()
    if controls[3].button("Pause", key="edu_pause", disabled=not playing):
        st.session_state["edu_playing"] = False
        st.session_state["edu_pause_reason"] = "Pause button"
        st.rerun()
    controls[4].button("Reset", key="edu_reset", on_click=move_stage, kwargs={"reset": True})
    controls[5].selectbox(
        "Educational playback · seconds per stage",
        [0.6, 1.0, 1.5, 2.5, 4.0],
        key=widget_value(st.session_state, "edu_speed"),
        on_change=commit_selection,
        args=("edu_speed",),
    )
    if st.session_state["edu_speed"] != scheduled_speed:
        st.session_state["edu_tick"] = monotonic()
        st.rerun()
    st.caption(
        "Educational playback uses an explanatory order and speed. It does not represent compute time or neuron activity and never reruns inference or training."
    )
    st.caption(
        f"Playback: {'Playing' if playing else 'Paused'} · Current stage: {labels[st.session_state['edu_stage']]}"
        + (
            " · " + st.session_state["edu_pause_reason"]
            if not playing and st.session_state.get("edu_pause_reason")
            else ""
        )
        + " · Pause before changing the stage or flow iteration."
    )
    if kind != "training" and st.session_state.get("edu_view") == "Recorded invocation order":
        if calls:

            def select_call():
                if not commit_selection("edu_call"):
                    return
                event = calls[st.session_state["edu_call"]]
                st.session_state["edu_stage"] = event["stage"]
                if event["iteration"] is not None:
                    st.session_state["edu_flow"] = event["iteration"]
                pause_selection()

            st.select_slider(
                "Recorded module invocation",
                options=list(range(len(calls))),
                key=widget_value(st.session_state, "edu_call"),
                format_func=lambda i: f"{i + 1}. {calls[i]['module']}"
                + (
                    "" if calls[i]["iteration"] is None else f" · iteration {calls[i]['iteration']}"
                ),
                on_change=select_call,
                disabled=playing,
            )
            st.caption(
                "Recorded module hook order. Tokenization and preprocessing happen before the model call. Top encoder and connector run before wrist encoder and connector. Parallel execution and per-stage timings were not measured."
            )
        else:
            st.info(
                "Open a captured run to explore invocation order. This is a source-based conceptual structure."
            )
    stage = st.session_state["edu_stage"]
    pipeline_stage = {
        "forward": "expert",
        "objective": "conditioning",
        "loss": "velocity",
        "backward": "velocity",
        "update": "velocity",
        "next": "expert",
    }.get(stage, stage)
    visuals = pipeline_visuals(data, instruction, st.session_state.get("edu_flow", 0))
    if data is None and kind != "simulation":
        if kind == "training" and training:
            import subprocess

            try:
                observation = training_observation(training)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                observation = None
                st.info(
                    "Training input images not available: "
                    + str(exc)
                    + " Current scenario images are not substituted."
                )
            else:
                st.caption(
                    f"Training input images · recorded episode {training['audit'].get('observation_episode', 0)} / frame {training['audit']['observation_frame']}"
                )
        else:
            observation = st.session_state.get("education_observation") or read_observation(
                sample_path()
            )
        for name, image in observation.images.items() if observation else []:
            stream = BytesIO()
            preview = image.copy()
            preview.thumbnail((180, 135))
            preview.save(stream, format="PNG")
            visuals["image_" + ("top" if name.endswith(".top") else "wrist")] = {
                "image": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode(),
            }
    if kind == "training" and training and training.get("visualization", {}).get("steps"):
        i = min(
            st.session_state.get("training_step_view", 0),
            len(training["visualization"]["steps"]) - 1,
        )
        actual = training["visualization"]["steps"][i]["forward"]
        for key, field in [
            ("expert", "expert_hidden"),
            ("velocity", "predicted_velocity"),
            ("conditioning", "x_t"),
        ]:
            value = actual[field]
            visuals[key] = {"shape": str(value["shape"]), "norms": value.get("token_norms", [])}
    event = _pipeline(
        stage=pipeline_stage,
        playing=playing,
        training=kind == "training",
        backward=kind == "training" and stage in {"backward", "update"},
        visuals=visuals,
        iteration=st.session_state.get("edu_flow", 0),
        total_iterations=data["capture"]["metadata"]["denoising_steps"] if data else 10,
        key="education_pipeline",
        default=None,
    )
    st.caption(
        "Top and wrist call the same encoder weights separately. Branches show data dependencies, not concurrent execution."
    )
    if event and event.get("nonce") != st.session_state.get("pipeline_event_seen"):
        st.session_state["pipeline_event_seen"] = event["nonce"]
        chosen = event.get("stage")
        if kind == "training" and chosen is not None:
            chosen = (
                "objective"
                if chosen == "conditioning"
                else "update"
                if chosen == "velocity"
                else "forward"
            )
        if chosen in ids:
            if kind != "training":
                st.session_state["edu_view"] = "Information flow"
            st.session_state.update(edu_stage=chosen, edu_tick=monotonic(), edu_playing=False)
            st.session_state["edu_pause_reason"] = "Pipeline node selected"
            st.rerun()
    st.selectbox(
        "Current training stage" if kind == "training" else "Current exploration stage",
        ids,
        format_func=lambda key: labels[key],
        key=widget_value(st.session_state, "edu_stage"),
        on_change=commit_selection,
        args=("edu_stage",),
        disabled=bool(calls) or playing,
    )
    if kind == "training":
        render_training(training, stage)
    else:
        if data:
            flow_steps = iterations(data)
            left, right = st.columns(2)
            if flow_steps:
                left.selectbox(
                    "Recorded flow iteration · large hidden summaries at 0/4/9",
                    sorted(flow_steps),
                    key=widget_value(st.session_state, "edu_flow"),
                    on_change=commit_selection,
                    args=("edu_flow",),
                    disabled=bool(calls) or playing,
                )
                if (
                    not data["capture"]["intermediates"]
                    .get("extended_capture", {})
                    .get("full_loop")
                ):
                    st.caption(
                        "This older record contains iterations 0/4/9 only. Missing activations and intermediate states are not interpolated."
                    )
            right.slider(
                "Action index within final chunk",
                0,
                len(data["capture"]["actions"]) - 1,
                key=widget_value(st.session_state, "edu_action"),
                on_change=commit_selection,
                args=("edu_action",),
            )
        render_stage(
            data, stage, st.session_state.get("edu_flow", 0), st.session_state.get("edu_action", 0)
        )
    if stopped:
        st.rerun()


def consume_completed(owner):
    job = st.session_state.get("active_job")
    if not job:
        return None
    status = manager().status(job, owner)
    if status["state"] == "completed" and st.session_state.get("education_loaded_job") != job:
        directory = manager().directory(job)
        if status["kind"] == "baseline":
            st.session_state["education_run"] = load_run((directory / "replay.json").read_bytes())
        elif status["kind"] == "evaluate":
            st.session_state["education_evaluation"] = load_run(
                (directory / "replay.json").read_bytes()
            )
            st.session_state["training_result_view"] = "Checkpoint inference"
        elif status["kind"] == "training":
            st.session_state["education_training"] = read_training(directory)
            st.session_state["training_job"] = job
        st.session_state["education_loaded_job"] = job
    return status


@st.fragment(run_every=1.0)
def job_feedback(job, owner):
    status = manager().status(job, owner)
    names = {
        "baseline": "Real inference",
        "training": "Real training",
        "simulation": "Simulation",
        "evaluate": "Trained-checkpoint inference",
    }
    directory = manager().directory(job)
    st.caption(
        f"{names[status['kind']]} · {status['state']} · {deepcopy(status['progress'].get('phase', 'Waiting to start'))}"
    )
    if status["state"] not in TERMINAL:
        if status["kind"] == "simulation":
            latest = status["progress"].get("latest_scene")
            if latest and local_file(directory, latest).is_file():
                st.image(
                    str(local_file(directory, latest)),
                    caption=f"Actual simulator frame · control step {status['progress'].get('simulator_control_step', 0)}",
                    width="stretch",
                )
                st.caption(
                    "Bounded frame updates during execution; video becomes available after completion."
                )
        if st.button("Cancel execution", key="cancel_job"):
            manager().cancel(job, owner)
    elif st.session_state.get("education_loaded_job") != job and status["state"] == "completed":
        st.rerun()
    elif status["state"] != "completed":
        st.warning("The job did not complete. See the diagnostic panel for the exact failure.")
    with st.expander("Developer details · run / configuration / diagnostics / downloads"):
        st.code(job, language=None)
        st.json(deepcopy(status), expanded=False)
        for name in ("error.json", "preflight.json", "result.json"):
            if (directory / name).exists():
                st.json(deepcopy(read_json(directory / name)), expanded=False)
        if status["state"] in TERMINAL:
            try:
                archive = manager().export(job, owner)
            except ValueError as exc:
                st.warning(
                    str(exc)
                    + " The saved result remains available for playback and operator export."
                )
            else:
                st.download_button(
                    "Download completed artifacts (ZIP)",
                    archive,
                    file_name=f"{job}.zip",
                    key="job_download",
                )


def main():
    st.set_page_config(page_title="VLA Model Explorer and Visualizer", page_icon="◈", layout="wide")
    st.markdown(
        """<style>.block-container{max-width:1500px;padding-top:1.6rem}
    [data-testid="stMetric"]{background:#f5f7fc;padding:12px;border-radius:10px}
    h1{letter-spacing:-.04em}div[data-testid="stVerticalBlock"]{gap:.65rem}</style>""",
        unsafe_allow_html=True,
    )
    st.caption("ECE525 · Observation → representation → context → action")
    st.title("VLA Model Explorer and Visualizer")
    st.caption("Interactive Inspection and Inference Replay with SmolVLA")
    st.write(
        "Follow how images, instructions, and robot state become internal representations and actions."
    )
    cols = st.columns([1, 1.4, 2.3])
    models = list_models(available_only=True)
    cols[0].selectbox(
        "Model",
        [m.model_id for m in models],
        format_func=lambda key: next(m.display_name for m in models if m.model_id == key),
        key="education_model",
    )
    mode = cols[1].selectbox("Mode", list(MODES), key="workflow_mode")
    kind = MODES[mode]
    scenarios = (
        [
            "SO100 · canonical baseline · episode 0 / frame 0",
            "SO100 · dataset episode/frame exploration",
        ]
        if kind != "simulation"
        else ["LIBERO Spatial · task 0 · readiness check"]
    )
    if st.session_state.get("education_scenario") not in scenarios:
        st.session_state["education_scenario"] = scenarios[0]
    scenario = cols[2].selectbox("Scenario / environment", scenarios, key="education_scenario")
    observation_mode = "canonical"
    observation_ready = True
    episode = frame = 0
    if kind != "simulation" and scenario == scenarios[1]:
        from vla_visualizer.data.so100 import catalog

        prepared = catalog()
        episodes = sorted(int(x) for x in prepared["episodes"])
        remote = prepared.get("remote_metadata") or {}
        st.caption(
            f"Remote pinned metadata: {remote.get('total_episodes', 'Not verified')} episodes / {remote.get('total_frames', 'Not verified')} frames · Locally prepared: {len(episodes)} episodes · Only prepared episodes are selectable."
        )
        if episodes:
            picks = st.columns([1, 3])
            if st.session_state.get("dataset_episode") not in episodes:
                st.session_state["dataset_episode"] = episodes[0]
            episode = picks[0].selectbox("Dataset episode", episodes, key="dataset_episode")
            count = prepared["episodes"][str(episode)]["frames"]
            if st.session_state.get("dataset_frame", 0) >= count:
                st.session_state["dataset_frame"] = 0
            frame = picks[1].slider("Recorded frame", 0, count - 1, key="dataset_frame")
            try:
                selected_observation = dataset_observation(episode, frame)
                st.session_state["education_observation"] = selected_observation
                observation_mode = "dataset"
                images = st.columns(2)
                for col, (camera, image) in zip(images, selected_observation.images.items()):
                    col.image(
                        image,
                        width="stretch",
                        caption=f"{camera} · episode {episode} / frame {frame}",
                    )
                p = selected_observation.provenance
                st.caption(
                    f"Recorded timestamp {p['timestamp_seconds']:.6f}s · task_index {p['task_index']} · Recorded instruction: {selected_observation.instruction}"
                )
                st.caption(
                    "Recorded dataset observations. Predicted actions did not cause these frames. Selection does not run inference. Cameras and state share the dataset frame/timestamp; sensor acquisition synchronization was not independently measured."
                )
            except Exception as exc:
                observation_ready = False
                st.error(str(exc))
        else:
            observation_ready = False
            st.warning(
                "No local episodes are prepared. Prepare dataset files or select the canonical scenario."
            )
    else:
        st.session_state.pop("education_observation", None)
    if kind == "simulation":
        from vla_visualizer.simulation.compatibility import INSTRUCTION

        examples = [("Selected simulator task · readiness required", INSTRUCTION)]
    else:
        examples = INSTRUCTION_EXAMPLES if kind == "baseline" else INSTRUCTION_EXAMPLES[:1]
    instructions = [text for _, text in examples]
    if st.session_state.get("instruction_example") not in instructions:
        st.session_state["instruction_example"] = instructions[0]
    prompt, execute = st.columns([3, 1])
    instruction = prompt.selectbox(
        "Instruction example",
        instructions,
        format_func=lambda text: next(label for label, value in examples if text == value),
        key="instruction_example",
    )
    prompt.write(instruction)
    owner = st.session_state.setdefault("job_owner", uuid.uuid4().hex)
    simulation_report = None
    if kind == "simulation":
        from vla_visualizer.simulation.compatibility import worker_readiness

        simulation_report = worker_readiness()
    blocked = not observation_ready or (
        kind == "simulation" and bool(simulation_report["blockers"])
    )
    if execute.button(
        {
            "baseline": "Run inference with selected instruction",
            "training": "Start tiny training",
            "simulation": "Start simulation",
        }[kind],
        type="primary",
        width="stretch",
        disabled=blocked,
        key="start_job",
    ):
        try:
            config = JobConfig(
                kind=kind,
                instruction=instruction if kind != "simulation" else None,
                device=st.session_state.get("worker_device", "cpu"),
                seed=int(st.session_state.get("worker_seed", 0)),
                max_seconds=st.session_state.get("worker_seconds", 300),
                training_steps=st.session_state.get("worker_training_steps", 2),
                observation_mode=observation_mode,
                episode_index=episode,
                frame_index=frame,
                max_steps=st.session_state.get("worker_sim_steps", 100),
                action_steps=st.session_state.get("worker_action_steps", 10),
            )
            st.session_state["active_job"] = manager().submit(config, owner)
        except Exception as exc:
            st.error(str(exc))
    if kind == "baseline":
        if execute.button("Open verified capture", key="saved_evidence", width="stretch"):
            st.session_state["education_run"] = saved_capture()
        full = ROOT / "assets/education/full_loop_replay.json"
        if execute.button(
            "Open recorded 10-iteration loop",
            key="saved_full_loop",
            disabled=not full.exists(),
            width="stretch",
        ):
            st.session_state["education_run"] = load_run(full.read_bytes())
        st.caption(
            "SO100 dataset inference. The selected sentence reaches the tokenizer and model. Task success for sentence variations is not verified."
        )
    elif kind == "training":
        evidence = ROOT / "assets/education/training_capture.json"
        if execute.button(
            "Open recorded training experiment",
            key="saved_training",
            disabled=not evidence.exists(),
            width="stretch",
        ):
            st.session_state["education_training"] = read_json(evidence)
        st.caption(
            "Training uses the original dataset task annotation. Sentence variations are for inference exploration."
        )
    else:
        if blocked:
            st.warning(
                "Simulator prerequisites are missing. You can still explore the model structure below."
            )
        else:
            st.info(
                "Required files and packages are present. Actual reset, rendering, policy execution, and stepping are validated by a real run."
            )
    status = consume_completed(owner)
    if status and (status["kind"] == kind or (kind == "training" and status["kind"] == "evaluate")):
        job_feedback(st.session_state["active_job"], owner)
    elif status:
        with st.expander("Job from another mode · separate from the current pipeline"):
            job_feedback(st.session_state["active_job"], owner)
    data = st.session_state.get("education_run") if kind == "baseline" else None
    data = data.read() if data else None
    training = st.session_state.get("education_training") if kind == "training" else None
    evaluation = st.session_state.get("education_evaluation") if kind == "training" else None
    if kind == "training":
        own = manager().completed(owner, "training")
        choices = [r["run_id"] for r in own]
        if choices:
            selected = st.selectbox(
                "Saved training checkpoint",
                choices,
                key="checkpoint_selection",
                format_func=lambda key: f"Training {key[:8]} · saved checkpoint",
            )
            st.session_state["training_job"] = selected
            if training is None or training["result"]["run_id"] != selected:
                training = read_training(manager().directory(selected))
                st.session_state["education_training"] = training
        if evaluation:
            if st.session_state.get("training_result_view") not in (
                "Training process",
                "Checkpoint inference",
            ):
                st.session_state["training_result_view"] = "Training process"
            view = st.radio(
                "Training result view",
                ["Training process", "Checkpoint inference"],
                key=widget_value(st.session_state, "training_result_view"),
                on_change=commit_selection,
                args=("training_result_view",),
                horizontal=True,
            )
            if view == "Checkpoint inference":
                data = evaluation.read()
                parent = data["capture"]["metadata"]["parent_training_run_id"]
                if (
                    st.session_state.get("training_job")
                    and parent != st.session_state["training_job"]
                ):
                    st.info(
                        f"Displayed evaluation belongs to training {parent[:8]}. The selected checkpoint has not replaced this saved result; request evaluation to produce its own result."
                    )
                st.success(
                    "Trained checkpoint reconstructed in a fresh worker; real inference completed."
                )
                st.caption(
                    f"Checkpoint SHA256: {data['capture']['metadata']['weight_sha256']} · parent training: {data['capture']['metadata']['parent_training_run_id']}"
                )
    if data:
        st.caption("● Saved inference playback · no new model execution")
        if data["manifest"]["instruction"] != instruction:
            st.warning(
                f"This capture used: {data['manifest']['instruction']} The next inference will use the selected sentence above."
            )
        provenance = data["capture"]["metadata"].get("observation_provenance", {})
        if (provenance.get("episode_index", 0), provenance.get("frame_index", 0)) != (
            episode,
            frame,
        ):
            st.info(
                f"Playback below belongs to recorded episode {provenance.get('episode_index')} / frame {provenance.get('frame_index')}. The sample selection above applies to the next execution."
            )
        with st.expander("Developer details · capture identity / immutable export"):
            st.json(data["manifest"], expanded=False)
            st.download_button(
                "Download capture JSON",
                (evaluation or st.session_state.get("education_run")).snapshot,
                file_name="captured_inference.json",
            )
    elif training:
        st.caption("● Recorded training · measured forward / gradients / updates")
    else:
        st.caption("○ Conceptual model structure · no measured activations loaded")
    show_evaluation = (
        kind == "training"
        and evaluation is not None
        and st.session_state.get("training_result_view") == "Checkpoint inference"
    )
    explorer(
        data,
        instruction,
        "baseline" if show_evaluation else kind,
        None if show_evaluation else training,
    )
    if kind == "training" and st.session_state.get("training_job"):
        st.caption(
            "Evaluation uses the saved training observation and seed to verify reconstruction. It is not a task-performance test."
        )
        if st.button("Evaluate saved training checkpoint", key="evaluate_training"):
            try:
                parent_config = read_json(
                    manager().directory(st.session_state["training_job"]) / "config.json"
                )
                st.session_state["active_job"] = manager().submit(
                    JobConfig(
                        kind="evaluate",
                        checkpoint_job=st.session_state["training_job"],
                        seed=parent_config["seed"],
                        device=st.session_state.get("worker_device", "cpu"),
                    ),
                    owner,
                )
                st.session_state["training_result_view"] = "Checkpoint inference"
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    if kind == "simulation":
        st.subheader("Simulator scene")
        st.info(
            "No genuine simulator-rendered frames are available yet. Dataset photos and joint-pose playback are not simulator results."
        )
        report = simulation_report
        st.write("Current blocker: " + "; ".join(report["blockers"][:3]))
        with st.expander("Developer details · simulator compatibility / validation logs"):
            st.json(report, expanded=False)
        if (
            status
            and status["kind"] == "simulation"
            and (manager().directory(status["run_id"]) / "rollout.json").exists()
        ):
            from vla_visualizer.ui.rollout_viewer import render_rollout

            render_rollout(manager().directory(status["run_id"]))
    with st.expander("Developer settings · workers / resources / checkpoint readiness"):
        import json

        st.download_button(
            "Download session receipt (keep private)",
            json.dumps(
                {
                    "schema_version": 1,
                    "owner": owner,
                    "active_job": st.session_state.get("active_job"),
                }
            ),
            file_name="session_receipt.json",
            key="session_receipt_download",
        )
        receipt = st.file_uploader(
            "Restore session receipt after an application restart",
            type=["json"],
            key="session_receipt_upload",
        )
        if receipt and st.button("Restore saved session", key="restore_session"):
            try:
                saved = json.loads(receipt.getvalue())
                if not isinstance(saved.get("owner"), str) or len(saved["owner"]) != 32:
                    raise ValueError("Invalid session receipt.")
                if saved.get("active_job"):
                    manager().status(saved["active_job"], saved["owner"])
                st.session_state["job_owner"] = saved["owner"]
                st.session_state["active_job"] = saved.get("active_job")
                st.session_state.pop("education_loaded_job", None)
                st.rerun()
            except (ValueError, KeyError, PermissionError, FileNotFoundError) as exc:
                st.error(str(exc))
        imported = st.file_uploader(
            "Import a saved capture JSON", type=["json"], key="capture_import"
        )
        if imported and st.button("Open imported capture", key="open_import"):
            try:
                st.session_state["education_run"] = load_run(imported.getvalue())
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        inventory = manager().inventory()
        st.caption(
            f"Recent primary records: {inventory['recent_primary_records']} / {inventory['primary_limit']}"
            f" · Logically archived: {inventory['archived']}"
            f" · All retained records: {inventory['retained_records']} / {inventory['retained_record_limit']}. "
            "Archived results stay accessible; nothing is deleted or moved."
        )
        st.json(inventory, expanded=False)
        st.selectbox("Worker device", ["cpu", "cuda"], key="worker_device")
        st.number_input("Seed", 0, 2147483647, value=0, key="worker_seed")
        st.slider("Job wall-time limit (seconds)", 30, 900, 300, step=30, key="worker_seconds")
        st.slider("Maximum optimizer steps", 1, 3, 2, key="worker_training_steps")
        if kind == "simulation":
            st.slider(
                "Maximum simulator control steps", 20, 280, 100, step=20, key="worker_sim_steps"
            )
            st.selectbox(
                "Actions executed before replanning",
                [1, 5, 10, 50],
                index=2,
                key="worker_action_steps",
            )
        from scripts.readiness import readiness

        st.json(readiness(), expanded=False)
        st.caption(
            "One model job at a time. The immutable SO100 checkpoint and native normalization are preserved."
        )
