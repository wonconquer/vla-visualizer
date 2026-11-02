"""Minimal Streamlit frontend for one real pretrained VLA policy."""

import streamlit as st
import json
from pathlib import Path
from hashlib import sha256

from vla_visualizer.contracts import CaptureOptions, LoadOptions
from vla_visualizer.inference.observations import observation_fingerprint, read_observation
from vla_visualizer.inference.runner import InferenceRunner
from vla_visualizer.inference.replay import make_run, load_run, ReplayError
from vla_visualizer.ui.inspection import render_inspection
from vla_visualizer.ui.inputs import render_inputs
from vla_visualizer.ui.results import render_architecture, render_results
from vla_visualizer.utils.device import detect_device


@st.cache_resource
def get_backend() -> InferenceRunner:
    """Share a manager whose loads require the current checkpoint-normalization audit."""
    return InferenceRunner()


def main() -> None:
    st.set_page_config(page_title="VLA Model Explorer and Visualizer", layout="wide")
    st.title("VLA Model Explorer and Visualizer")
    st.caption("ECE525 · Real pretrained inference and completed-capture inspection / replay.")
    backend = get_backend()
    device = detect_device()
    models = backend.list_models()
    inputs, architecture, results = st.columns([1, 1.5, 1])
    with inputs:
        selected, observation = render_inputs(
            models, {m.model_id: backend.get_input_spec(m.model_id) for m in models}
        )
        selected_device = st.selectbox(
            "Inference device", ["cpu", "cuda"] if device.cuda_available else ["cpu"], key="device"
        )
        downloads = st.checkbox(
            f"Allow checkpoint downloads (~{selected.download_size_mb} MB)", value=False
        )
        seed = st.number_input("Inference seed", min_value=0, max_value=2147483647, value=0)
        st.caption(
            "One model is shared by this local server. Load/unload affects all browser sessions."
        )
        if st.button("Load model", width="stretch"):
            st.session_state.pop("last_run", None)
            try:
                with st.spinner("Loading frozen model and checkpoint processors…"):
                    backend.load(
                        selected.model_id,
                        LoadOptions(
                            device=selected_device, local_files_only=not downloads, seed=int(seed)
                        ),
                    )
            except Exception as exc:
                st.error(f"Model load failed: {str(exc)[:700]}")
                st.caption(
                    "If memory is insufficient, choose CPU and reload. No system GPU settings are changed."
                )
        if st.button(
            "Unload model", disabled=not backend.is_loaded(selected.model_id), width="stretch"
        ):
            backend.unload()
            st.session_state.pop("last_run", None)
        capture_enabled = st.checkbox(
            "Capture internal representations for replay", key="capture_enabled"
        )
        fingerprint = observation_fingerprint(observation) if observation is not None else None
        if st.button(
            "Run inference",
            disabled=observation is None or not backend.is_loaded(selected.model_id),
            width="stretch",
        ):
            st.session_state.pop("last_run", None)
            try:
                with st.spinner("Running real pretrained inference…"):
                    result = backend.run_inference(
                        selected.model_id, observation, CaptureOptions(enabled=capture_enabled)
                    )
                if capture_enabled:
                    completed = make_run(
                        result, backend.get_inspection_spec(selected.model_id), observation
                    )
                    st.session_state["replay_run"] = completed
                else:
                    st.session_state["last_run"] = (selected.model_id, fingerprint, result)
            except Exception as exc:
                st.error(f"Inference failed: {str(exc)[:700]}")
        previous = st.session_state.get("last_run")
        result = (
            previous[2]
            if previous
            and previous[:2] == (selected.model_id, fingerprint)
            and previous[2].metadata.get("normalization_verified") is True
            else None
        )
        if result is None:
            st.session_state.pop("last_run", None)
        st.caption(
            "Device and seed selections apply on Load model. CPU uses float32 and eight threads."
        )
        if st.button("Load saved capture", key="load_saved_capture"):
            try:
                root = Path(__file__).resolve().parent
                payload = json.loads((root / "assets/education/reference_capture.json").read_text())
                sample_path = root / "assets/sample_inputs/so100_observation.json"
                sample_manifest = json.loads(sample_path.read_text())
                hashes = payload["metadata"]["observation_provenance"]["image_sha256_by_camera"]
                for camera, filename in sample_manifest["images"].items():
                    if (
                        sha256((sample_path.parent / filename).read_bytes()).hexdigest()
                        != hashes[camera]
                    ):
                        raise ReplayError(
                            "Source sample image checksum differs from saved capture."
                        )
                completed = make_run(
                    payload,
                    backend.get_inspection_spec(selected.model_id),
                    read_observation(sample_path),
                    origin="saved reference capture; no model execution",
                )
                st.session_state["replay_run"] = completed
            except Exception as exc:
                st.error(
                    f"Saved capture rejected; previous completed run preserved: {str(exc)[:700]}"
                )
        if st.checkbox("Import completed replay JSON", key="import_replay"):
            upload = st.file_uploader(
                "Completed replay JSON (up to 6 MiB)", type=["json"], key="replay_upload"
            )
            if st.button("Open uploaded replay", disabled=upload is None):
                try:
                    completed = load_run(upload.getvalue())
                    if completed.read()["view_spec"] != backend.get_inspection_spec(
                        selected.model_id
                    ):
                        raise ReplayError(
                            "Source specification is not the approved adapter specification."
                        )
                    st.session_state["replay_run"] = completed
                except Exception as exc:
                    st.error(f"Replay rejected; previous completed run preserved: {str(exc)[:700]}")
    loaded = backend.is_loaded(selected.model_id)
    completed = st.session_state.get("replay_run")
    with architecture:
        render_architecture(backend.get_structure(selected.model_id), loaded)
    with results:
        render_results(selected, device, loaded, result, completed.run_id if completed else None)
    if completed is not None:
        manifest = completed.read()["manifest"]
        if fingerprint != manifest.get("observation_fingerprint"):
            st.warning(
                "Pending inputs differ from the completed replay. The viewer still belongs to the completed run shown below."
            )
        if manifest.get("seed") != int(seed) or manifest.get("device") != selected_device:
            st.warning(
                "Pending device/seed settings differ from this completed run; selections apply only on model load."
            )
        try:
            render_inspection(completed)
        except ReplayError as exc:
            st.error(f"Capture view unavailable: {exc}")


if __name__ == "__main__":
    main()
