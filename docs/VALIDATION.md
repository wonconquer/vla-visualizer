# Validation and limits

## Automated checks

The suite separates unit fixtures, saved real capture mapping, Streamlit AppTest,
and opt-in pretrained inference. Synthetic unit tensors are not robotics evidence.
The English publication is checked separately from the development workspace.

The development build passed 122 default tests with episode 0 prepared. A fresh
checkout lacks videos, so dataset-dependent tests are skipped until preparation.
Opt-in tests require the pinned model cache and do not download weights.

The English publication checkout passed **120 tests, with 9 skipped** before
dataset preparation. Its saved assets were compared against the development
artifacts: all 138,047 numeric leaves were unchanged. Text annotations and
machine-local paths are handled as described in [DATA.md](DATA.md).
The publication's three separate pretrained-model tests also passed. A compact
machine-readable record is in [validation.json](validation.json).

## Real execution evidence

On AMD EPYC 9354 CPU, Python 3.12.14, float32, eight threads, and seed 0:

- Strict 500-key loading and native normalization/postprocessing passed.
- Canonical action shape was `[1,50,6]`; all values were finite.
- Maximum difference from the established canonical output was 0.0 at
  `rtol=0, atol=1e-6`.
- Ten extended Euler updates matched `x_after = x_before + dt * velocity`
  with maximum error 0.0; hooks were removed after prediction.
- Stage, flow, action-index, and playback interaction submitted no extra jobs.
- A one-step output-head experiment at the final episode frame used one valid
  target and 49 masked positions. Base and frozen weights were preserved.
  A fresh worker reconstructed the delta and matched the same-observation probe
  with maximum action difference 0.0. This is not improved-performance evidence.

Captured inference was approximately 1.8 seconds on that shared CPU host.
Timings vary with hardware, load, and warmup and are not GPU benchmarks.

## Unverified paths

Simulation dependencies, separate LIBERO weights, and task assets were unavailable;
no genuine simulator rollout or task success is claimed. GPU execution was not
validated because the host driver was unavailable. Browser screenshots, mouse
hover, and client timer delivery have not passed graphical acceptance; AppTest
and JavaScript syntax checks cover server state and syntax only.

The application displays these limits and does not substitute recorded dataset
frames for simulator output or invent missing activations.
