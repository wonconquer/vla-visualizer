# Checkpoint, dataset, and processors

The approved pairing uses the SO100 two-camera fine-tuned checkpoint and dataset
revisions in [DATA.md](DATA.md). Canonical camera mapping is top to camera1 and
wrist to camera2. State/action order and legacy units are taken from the pinned
dataset representation, not inferred from tensor dimension alone.

The adapter hashes the native configuration, processor JSONs, processor statistics,
and model weights before loading. Strict state-dict loading checks the complete
parameter schema. Native state normalization and action inverse normalization
are verified numerically on each run.

`smolvla_pairing.json` contains the supporting metadata. The historical
`lerobot/smolvla_base` missing-key audit remains a small regression fixture;
it is not the inference checkpoint used by the application.
