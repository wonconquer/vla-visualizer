# Testing

Run the offline checks after installing the pinned environment:

```bash
python -m pytest -q
python -m ruff check app.py app_baseline.py vla_visualizer scripts tests
python -m compileall -q app.py app_baseline.py vla_visualizer scripts tests
```

The default suite covers checkpoint authentication, native normalization,
capture-to-view mapping, replay isolation, callback cleanup, job ownership,
resource limits, and checkpoint reconstruction. It uses fixed captured data and
explicit synthetic unit fixtures. Model and dataset-dependent tests are skipped
until their required resources are prepared.

To run the pretrained-model checks with an existing cache:

```bash
ECE525_REAL_SMOKE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ECE525_VALIDATION_DIR=outputs/real_validation \
  python -m pytest -q tests/test_real_smolvla.py tests/test_real_inspection.py \
  tests/test_real_viewer.py
```

The canonical input must produce a finite `[1,50,6]` chunk matching
`tests/fixtures/reference_actions.json` at `rtol=0, atol=1e-6`. Tests check that
native normalization and inverse normalization are applied, captured inference
preserves the uncaptured result, and replay adds no inference calls or hooks.

Graphical browser acceptance, GPU execution, and simulator integration are not
established by these tests. Predicted actions are not evidence of robot task success.
