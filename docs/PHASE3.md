# Internal inspection validation

Selected native module hooks observe vision, connector, language, state,
multimodal-prefix, action-state, time-conditioned, and velocity representations.
Whole text/expert module names are not treated as executed forward hooks because
the implementation traverses their internal submodules directly.

The basic capture retains 23 semantic records. Three same-seed baseline/capture
comparisons had maximum action difference 0.0, within `rtol=0, atol=1e-6`.
Hook counts were zero before and after each call. The actual record is in
`phase3_integration_validation.json`.

Captured tensors are detached and reduced to bounded CPU data. Large hidden
summaries are retained at iterations 0, 4, and 9. Attention probabilities are
not returned; embedding norms are not used as substitute attention.

See [ARCHITECTURE.md](ARCHITECTURE.md) for source boundaries and extended loop capture.
