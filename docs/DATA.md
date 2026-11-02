# Data and capture provenance

The canonical observation is episode 0 / frame 0 / timestamp 0 / task index 0 of
`lerobot/svla_so100_pickplace` at revision
`87169e502b742f989b96c952bff7af55e248d467`.

The two included PNGs were extracted from the top and wrist episode-0 videos.
The six measured state values and instruction are stored in
`assets/sample_inputs/so100_observation.json`. Its source hashes identify the
original videos, Parquet, and task metadata. Sensor acquisition synchronization
was not independently measured; views and state share the recorded frame/timestamp.

| Dimension | State | Action | Source coordinate convention |
| --- | --- | --- | --- |
| shoulder_pan | Measured follower position | Absolute target position | Legacy calibrated degrees |
| shoulder_lift | Measured follower position | Absolute target position | Legacy calibrated degrees |
| elbow_flex | Measured follower position | Absolute target position | Legacy calibrated degrees |
| wrist_flex | Measured follower position | Absolute target position | Legacy calibrated degrees |
| wrist_roll | Measured follower position | Absolute target position | Legacy calibrated degrees |
| gripper | Measured follower position | Absolute target position | Calibrated percent, nominal 0-100 |

Native mean/std statistics are authenticated against the pinned checkpoint and
match the dataset aggregates. The app does not substitute missing state/action
statistics or disable normalization. Later dataset revisions changed coordinate
conventions, so the current dataset main branch is not interchangeable with this pin.

The 24,457-byte episode-0 Parquet supplies recorded training targets. It is not
a collection of model predictions. End-of-episode padding is excluded from the
native objective with the action padding mask.

`assets/education/` contains saved inference captures and training summaries.
They supply the viewer's recorded examples; opening one does not run the model.
Tensor statistics, token IDs, module paths, camera identities, and action values
come from actual execution. Missing representations are marked unavailable.

See [THIRD_PARTY.md](../THIRD_PARTY.md) for upstream sources and the dataset license.
