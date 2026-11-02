"""Render real model output and source-derived architecture without synthetic data."""

from dataclasses import asdict
import json

import streamlit as st

from vla_visualizer.contracts import InferenceResult, ModelSpec, ModelStructure
from vla_visualizer.utils.device import DeviceInfo
from vla_visualizer.visualization.architecture import architecture_dot


def render_architecture(structure: ModelStructure, loaded: bool) -> None:
    st.subheader("Architecture / data flow")
    st.caption(
        "Adapter diagram verified against the installed implementation. Module paths are checked on load; this is not a measured execution trace."
    )
    st.graphviz_chart(architecture_dot(structure), width="stretch")
    if loaded:
        with st.expander("Verified module paths"):
            for stage in structure.stages:
                if stage.module_paths:
                    st.write({"stage": stage.label, "modules": stage.module_paths})
    st.caption("Enable internal capture or load a saved capture to open the replay viewer below. Attention maps are unavailable; this diagram is not an execution animation.")


def render_results(
    model: ModelSpec, device: DeviceInfo, loaded: bool, result: InferenceResult | None,
    completed_run_id: str | None = None,
) -> None:
    st.subheader("Results")
    st.write(f"Selected model: {model.display_name}")
    st.write(f"Model status: {'loaded' if loaded else 'not loaded'}")
    if result is None:
        if completed_run_id:
            st.info(f"Completed capture {completed_run_id} is available in the replay viewer below. Its identity is independent of pending inputs and model loading.")
        else:
            st.info("No predicted actions for the current inputs. Load the model and run inference.")
    else:
        st.success(
            f"Real inference completed: {len(result.actions)} action steps × {len(result.actions[0])} dimensions"
        )
        st.caption(
            f"Predicted actions: {result.metadata['action_scale']}. "
            f"{result.metadata['coordinate_convention']} Inference does not establish robotics task success."
        )
        rows = [
            {"step": step, **{
                name: value for name, value in zip(result.metadata["action_names"], row, strict=True)
            }}
            for step, row in enumerate(result.actions)
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
        st.write(
            f"Inference: {result.metadata['inference_seconds']:.2f} s · Device: {result.metadata['device']}"
        )
        st.json(dict(result.metadata), expanded=False)
        st.download_button(
            "Download actual action output (JSON)",
            json.dumps(asdict(result), indent=2, allow_nan=False),
            file_name="vla_actions.json",
            mime="application/json",
        )
    st.subheader("Device availability")
    st.write(f"CUDA available: {device.cuda_available}")
    st.caption(
        f"PyTorch {device.torch_build} · Wheel CUDA runtime: {device.cuda_runtime or 'none'}"
    )
    if device.cuda_device_name:
        st.write(f"GPU: {device.cuda_device_name}")
    if device.detail:
        st.caption(device.detail)
