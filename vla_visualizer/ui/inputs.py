"""Input widgets; the backend supplies all model input requirements."""

import math
from pathlib import Path

import streamlit as st

from vla_visualizer.contracts import InputSpec, ModelSpec, Observation
from vla_visualizer.inference.observations import read_observation
from vla_visualizer.utils.images import decode_image


def parse_state(text: str, dimension: int) -> tuple[float, ...]:
    """Require a finite, explicitly supplied state vector; never substitute zeros."""
    try:
        values = tuple(float(item.strip()) for item in text.split(","))
    except ValueError as exc:
        raise ValueError("Enter comma-separated numeric robot-state values.") from exc
    if len(values) != dimension or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Provide exactly {dimension} finite robot-state values.")
    return values


def render_inputs(
    models: tuple[ModelSpec, ...], specs: dict[str, InputSpec]
) -> tuple[ModelSpec, Observation | None]:
    st.subheader("Inputs")
    selected_id = st.selectbox(
        "Model",
        options=[model.model_id for model in models],
        format_func=lambda model_id: next(
            f"{model.display_name} · {model.status}"
            for model in models
            if model.model_id == model_id
        ),
        key="model_id",
    )
    selected = next(model for model in models if model.model_id == selected_id)
    spec = specs[selected_id]
    st.caption(f"Checkpoint: {selected.checkpoint_id}")
    if spec.normalization_description:
        st.info(spec.normalization_description)
    if st.checkbox("Use real SO100 smoke observation", key="use_sample"):
        path = Path(__file__).resolve().parents[2] / "assets/sample_inputs/so100_observation.json"
        observation = read_observation(path)
        for camera, image in observation.images.items():
            st.image(image, caption=f"{camera} · SO100 episode 0, frame 0", width="stretch")
        st.write(observation.instruction)
        st.write({"measured_state": observation.state})
        st.caption(
            f"Dataset: {observation.provenance['dataset']} · revision: "
            f"{observation.provenance['revision']} · episode 0 / frame 0 / timestamp 0 s."
        )
        return selected, observation

    images = {}
    for camera in spec.camera_keys:
        uploaded = st.file_uploader(
            f"Robot observation image: {camera}",
            type=["png", "jpg", "jpeg"],
            key=f"observation_image_{camera}",
            help="A synchronized RGB observation, up to 20 MiB and 16 megapixels.",
        )
        if uploaded is not None:
            try:
                images[camera] = decode_image(uploaded.getvalue())
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.image(images[camera], caption=camera, width="stretch")
    instruction = st.text_area("Robot instruction", key="instruction")
    state = None
    if spec.state_dim is not None:
        text = st.text_input(
            f"Measured robot state ({spec.state_dim} values, comma separated)", key="robot_state"
        )
        st.caption(
            "Use the checkpoint's robot-state convention and calibration. Values are not inferred from the image."
        )
        if text:
            try:
                state = parse_state(text, spec.state_dim)
            except ValueError as exc:
                st.error(str(exc))
    st.caption(
        "One synchronized frame per camera per run. Enable capture for completed-run inspection; video history is unavailable."
    )
    observation = None
    if (
        len(images) == len(spec.camera_keys) and instruction.strip()
        and (spec.state_dim is None or state is not None)
    ):
        observation = Observation(
            images=images, instruction=instruction.strip(), state=state
        )
    return selected, observation
