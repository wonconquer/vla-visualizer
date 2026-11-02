# Normalization guard

The historical base-checkpoint audit found dataset-prefixed action statistics
and unresolved canonical state keys. Matching feature sizes did not establish
robot/dataset semantics. The adapter therefore rejected that configuration.

The current MVP uses the dataset-matched fine-tuned checkpoint with native
`observation.state.mean/std` and `action.mean/std` statistics. Active statistics
must be finite, correctly shaped, and have positive standard deviations. Unused
constant metadata such as task-index std does not invalidate active features.

The guard authenticates exact checkpoint artifacts and checks that native
processing actually applies the statistics. Missing data is not replaced with
zeros/ones, normalization is not disabled, and robot embodiments are not mixed.
The small JSON fixtures support the missing-key and silent-skip regression tests.
