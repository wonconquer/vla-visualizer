# SmolVLA Model Explorer

A local Streamlit application for inspecting how camera images, an instruction,
and robot state become an action chunk in SmolVLA. This repository contains the
ECE525 course project, tests, and a small recorded example for review.

The viewer connects a source-checked architecture diagram to actual captured
tensors. It supports both information-flow and recorded-invocation views,
token-level summaries, all ten recorded flow iterations, and final action plots.
Replay reads a completed capture; navigation and playback do not run inference.

## Quick start

Use Linux x86_64 and Python 3.12. The pinned PyTorch wheel can run on CPU.

```bash
git clone https://github.com/wonconquer/ECE525-SmolVLA-Visualizer.git
cd ECE525-SmolVLA-Visualizer
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir pip==26.2.1 setuptools==81.0.0 wheel==0.46.3 packaging==25.0
python -m pip install --no-cache-dir --no-build-isolation --require-hashes -r requirements.lock.txt
python -m pip check
python -m scripts.serve
```

Open **http://127.0.0.1:7860**. For a remote development machine, forward port
7860 over your existing SSH or VS Code connection.

Select **Open recorded 10-iteration loop** to explore the included real capture.
This requires Python packages but no model weights, GPU, or dataset download.
Choose a stage or recorded invocation, use Play/Pause, and inspect token norms,
state transformations, flow updates, and the final 50-action chunk.

The environment lock targets the platform above. Fresh installation on other
platforms has not been validated. `requirements.txt` lists direct runtime
dependencies; the full lock also includes development tools.

## Run pretrained inference

```bash
python -m scripts.prepare_models baseline --allow-download \
  --output outputs/model_preparation.json
python -m scripts.job baseline --device cpu --seed 0 --seconds 300
```

Preparation downloads approximately 907 MB of policy files plus the required
tokenizer/configuration. Workers then run cache-only. In the UI, **Run inference
with selected instruction** starts an explicit worker job. One model worker is
admitted at a time. Completed captures remain available after the worker exits.

| Component | Pinned configuration |
| --- | --- |
| Policy | `Zekai-Chen/smolvla_so100_pickplace_ft_2cam` |
| Policy revision | `77172eaffe82c0897945607bfe818b5b164eb57e` |
| Dataset | `lerobot/svla_so100_pickplace` |
| Dataset revision | `87169e502b742f989b96c952bff7af55e248d467` |
| Runtime | LeRobot 0.6.1, Transformers 5.4.0, PyTorch 2.10.0+cu126 |
| Included observation | Episode 0, frame 0, timestamp 0, task index 0 |
| Instruction | `Pick up the cube and place it in the box.` |
| Native action chunk | `[1, 50, 6]` |

Top and wrist observations map to `camera1` and `camera2`. State and action order
is shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
State contains measured positions; actions are absolute target positions in the
pinned dataset's legacy calibrated coordinates. See [data provenance](docs/DATA.md).

## Optional dataset exploration and training

Frame exploration requires an AV1-capable `ffmpeg` executable.

```bash
# Prepare only the selected episodes, not the full dataset.
python -m scripts.prepare_so100 --episodes 0 1 --allow-download

# Two optimizer steps on the output projection only.
python -m scripts.job training --training-steps 2 --device cpu --seconds 300 \
  --receipt outputs/training_receipt.json
```

In the UI, Training mode exposes the recorded training experiment and evaluation
of the current session's saved checkpoint. A delta references the immutable
pretrained base; it is not a standalone model. The experiment does not establish
improved task performance. Session receipts contain private access capabilities
and should stay outside version control.

## Validation

```bash
python -m pytest -q
python -m ruff check app.py app_baseline.py vla_visualizer scripts tests
python -m compileall -q app.py app_baseline.py vla_visualizer scripts tests

# Requires the prepared model cache; no model downloads occur in this test.
ECE525_REAL_SMOKE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ECE525_VALIDATION_DIR=outputs/real_validation \
  python -m pytest -q tests/test_real_smolvla.py tests/test_real_inspection.py \
  tests/test_real_viewer.py
```

The implementation and measured limits are described in
[validation](docs/VALIDATION.md) and [architecture](docs/ARCHITECTURE.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| `app.py` | Main Streamlit entry point |
| `vla_visualizer/adapters/` | Model-specific loading, inspection, and normalization guards |
| `vla_visualizer/inference/` | Inference runner and immutable replay validation |
| `vla_visualizer/execution/` | Worker jobs, cancellation, resource bounds, and checkpoints |
| `vla_visualizer/ui/` | Stage panels, playback state, and pipeline component |
| `assets/` | Small real observations, captured inference, and recorded training summaries |
| `scripts/` | Launch, preparation, and validation commands |
| `tests/` | Unit, UI, and opt-in real-model tests |
| `app_baseline.py` | Earlier inference UI retained for regression tests |

Model weights, full datasets, environments, private receipts, and generated job
artifacts are excluded. See [supplemental material](docs/SUPPLEMENTAL_MATERIAL.md)
for the publication scope and [third-party sources](THIRD_PARTY.md) for credits.

## Limitations

- Genuine attention probabilities are not returned by the verified path. Norms
  are representation magnitudes, not attention, saliency, or confidence.
- Playback speed is explanatory, not measured stage latency.
- Most hidden tensors are summarized; only bounded token channels are retained.
- Simulation code is experimental. LIBERO reset/render/closed-loop execution
  has not passed integration validation and is disabled when prerequisites fail.
- Browser pixel layout and pointer-event acceptance remain unverified. Streamlit
  AppTest covers server-side behavior, not a graphical browser.
- Predicted actions were not executed on a robot. Inference success is not task success.

The repository is public source for review; it does not publish a hosted app.
