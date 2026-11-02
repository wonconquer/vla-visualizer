"""Opt-in real Streamlit capture → viewer → replay, with execution counters."""

from hashlib import sha256
import json
import os
from pathlib import Path
import resource
from time import perf_counter

import pytest
from streamlit.testing.v1 import AppTest
import torch

from examples.smolvla_inspection import rss_bytes
from vla_visualizer.adapters import registry
from vla_visualizer.inference.runner import InferenceRunner
from vla_visualizer.inference.replay import ReplayRun

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("ECE525_REAL_SMOKE") != "1", reason="Opt-in cached real model test"
)
def test_real_streamlit_capture_and_isolated_replay(monkeypatch):
    reference_files = {
        str(path): sha256(path.read_bytes()).hexdigest()
        for path in [
            ROOT / "docs/ARCHITECTURE.md",
            ROOT / "docs/DATA.md",
            ROOT / "tests/fixtures/reference_actions.json",
            ROOT / "assets/education/reference_capture.json",
        ]
    }
    adapters, invocations, registrations = [], [], []
    create = registry.create_adapter
    run_inference = InferenceRunner.run_inference
    register = torch.nn.Module.register_forward_hook

    def track_create(model_id):
        adapter = create(model_id)
        adapters.append(adapter)
        return adapter

    def track_run(self, *args, **kwargs):
        result = run_inference(self, *args, **kwargs)
        invocations.append(result)
        return result

    def track_register(self, *args, **kwargs):
        registrations.append(type(self).__name__)
        return register(self, *args, **kwargs)

    monkeypatch.setattr(registry, "create_adapter", track_create)
    monkeypatch.setattr(InferenceRunner, "run_inference", track_run)
    monkeypatch.setattr(torch.nn.Module, "register_forward_hook", track_register)
    app = AppTest.from_file(ROOT / "app_baseline.py", default_timeout=600).run()
    measurements = []
    app.checkbox(key="use_sample").check().run()
    try:
        app.button[0].click().run()
        assert not app.exception and not app.error
        adapter = next(a for a in adapters if a._policy is not None)
        policy = adapter._policy

        def hooks():
            return sum(len(module._forward_hooks) for module in policy.modules())

        before_hooks = hooks()
        app.button[2].click().run()  # Actual uncaptured baseline, seed 0.
        assert not app.exception and len(invocations) == 1
        baseline = invocations[-1]
        app.checkbox(key="capture_enabled").check().run()
        app.button[2].click().run()  # Actual captured inference pipeline.
        assert not app.exception and not app.error and len(invocations) == 2
        captured = invocations[-1]
        torch.testing.assert_close(
            torch.tensor(captured.actions), torch.tensor(baseline.actions), rtol=0, atol=1e-6
        )
        old_baseline = json.loads((ROOT / "tests/fixtures/reference_actions.json").read_text())
        torch.testing.assert_close(
            torch.tensor(captured.actions), torch.tensor(old_baseline["actions"]), rtol=0, atol=1e-6
        )
        assert captured.metadata["action_chunk_shape"] == [1, 50, 6]
        assert torch.tensor(captured.actions).isfinite().all()
        assert (
            captured.metadata["normalization_verified"]
            and captured.metadata["action_postprocessing_verified"]
        )
        assert hooks() == before_hooks == 0
        completed = app.session_state["replay_run"]
        snapshot, run_id = completed.snapshot, completed.run_id
        registration_count = len(registrations)
        rss_before_replay = rss_bytes()

        def check(label, action):
            start = perf_counter()
            action()
            elapsed = perf_counter() - start
            assert not app.exception and not app.error, label
            assert len(invocations) == 2 and len(registrations) == registration_count
            assert hooks() == 0
            assert app.session_state["replay_run"].snapshot == snapshot
            assert app.session_state["replay_run"].run_id == run_id
            measurements.append(
                {
                    "control": label,
                    "apptest_rerun_seconds": elapsed,
                    "server_view_construction_seconds": app.session_state["replay_render_seconds"],
                    "rss_bytes": rss_bytes(),
                }
            )

        for stage in (
            "camera_0",
            "camera_1",
            "language",
            "state",
            "prefix",
            "generation",
            "actions",
        ):
            check(
                stage, lambda stage=stage: app.selectbox(key="replay_stage").set_value(stage).run()
            )
        for step in (4, 9, 0):
            check(
                f"flow_{step}",
                lambda step=step: app.selectbox(key="replay_flow").set_value(step).run(),
            )
        check("action_17", lambda: app.slider(key="replay_action").set_value(17).run())
        assert any(frame.value.shape[0] == 6 and "value" in frame.value for frame in app.dataframe)
        check("reset", lambda: app.button(key="replay_reset").click().run())
        assert app.selectbox(key="replay_stage").value == "camera_0"
        assert app.slider(key="replay_action").value == 17
        check("next", lambda: app.button(key="replay_next").click().run())
        check("previous", lambda: app.button(key="replay_previous").click().run())
        check("play", lambda: app.button(key="replay_play").click().run())
        assert app.session_state["replay_playing"]
        # Simulate an elapsed fragment tick without sleeping or another model call.
        app.session_state["replay_tick"] = 0.0
        check("autoplay_tick", lambda: app.run())
        assert app.selectbox(key="replay_stage").value == "camera_1"
        check("pause", lambda: app.button(key="replay_play").click().run())
        assert not app.session_state["replay_timer_enabled"]
        check("play_again", lambda: app.button(key="replay_play").click().run())
        check("reset_stops_play", lambda: app.button(key="replay_reset").click().run())
        assert not app.session_state["replay_timer_enabled"]
        for i in range(5):
            check(f"repeat_{i}", lambda: app.run())

        # Inference errors preserve the prior completed run, never a partial replacement.
        def fail(*args, **kwargs):
            raise RuntimeError("Injected inference/OOM failure for preservation validation")

        with monkeypatch.context() as patch:
            patch.setattr(InferenceRunner, "run_inference", fail)
            app.button[2].click().run()
            assert app.error and not app.exception
            assert app.session_state["replay_run"].snapshot == snapshot
        check("unload", lambda: app.button[1].click().run())
        assert adapter._policy is None
        check(
            "replay_without_model",
            lambda: app.selectbox(key="replay_stage").set_value("state").run(),
        )
        assert sum(isinstance(v, ReplayRun) for v in app.session_state.filtered_state.values()) == 1
        assert "last_run" not in app.session_state
        for filename, digest in reference_files.items():
            assert sha256(Path(filename).read_bytes()).hexdigest() == digest
        data = completed.read()
        report = {
            "check": "captured_viewer",
            "status": "PASS",
            "run_manifest": data["manifest"],
            "actual_action_shape": captured.metadata["action_chunk_shape"],
            "first_actual_action": captured.actions[0],
            "max_absolute_baseline_difference": (
                torch.tensor(captured.actions) - torch.tensor(baseline.actions)
            )
            .abs()
            .max()
            .item(),
            "tolerance": {"rtol": 0, "atol": 1e-6},
            "strict_loading": captured.metadata["loading_keys"],
            "state_normalization_verified": captured.metadata["state_normalization_verified"],
            "action_postprocessing_verified": captured.metadata["action_postprocessing_verified"],
            "load_seconds": captured.metadata["load_seconds"],
            "uncaptured_inference_seconds": baseline.metadata["inference_seconds"],
            "captured_inference_seconds": captured.metadata["inference_seconds"],
            "captured_pipeline_seconds": captured.metadata["pipeline_seconds"],
            "inference_calls_total": len(invocations),
            "inference_calls_during_replay": 0,
            "hook_count_before": before_hooks,
            "hook_count_after": hooks(),
            "hook_registrations_total": registration_count,
            "hook_registrations_during_replay": 0,
            "capture_record_count": len(data["capture"]["intermediates"]["stages"]),
            "capture_json_bytes": len(json.dumps(data["capture"], separators=(",", ":")).encode()),
            "replay_payload_bytes_including_images": len(snapshot.encode()),
            "retained_run_objects": 1,
            "retained_live_tensors": 0,
            "rss_before_replay_bytes": rss_before_replay,
            "rss_after_replay_before_unload_bytes": measurements[-3]["rss_bytes"],
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "memory_scope": "Whole test process including Streamlit/torch runtime and model. RSS changes do not isolate capture retention; immutable bytes/run count and absence of tensors/hooks checked directly.",
            "ui_measurements": measurements,
            "ui_verified": "Streamlit AppTest widget/protobuf/dataframe rendering; real model run, both cameras, every stage, mask/padding, flow/action selectors, navigation, play/pause/tick/reset, unload and failure preservation",
            "browser_limit": "No graphical browser/Playwright available; pixel layout and browser-driven automatic timer delivery are not visually accepted. Fragment tick tested via elapsed monotonic state and rerun.",
            "cuda_available": torch.cuda.is_available(),
            "tested_device": "cpu",
            "gpu_memory_bytes": None,
            "reference_files_evidence_unchanged": reference_files,
        }
        destination = Path(os.environ.get("ECE525_VALIDATION_DIR", ROOT / "outputs/viewer_validation"))
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "viewer_validation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        (destination / "captured_replay.json").write_text(snapshot)
    finally:
        if not app.button[1].disabled:
            app.button[1].click().run()
