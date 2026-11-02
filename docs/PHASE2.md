# Pretrained inference validation

The SO100 fine-tuned policy loads strictly with 500 state-dict keys and no missing
or unexpected parameters. All 450,046,176 parameters are frozen in evaluation mode.
The native preprocessor normalizes measured state and formats/tokenizes the
instruction. The action-chunk API runs under `torch.inference_mode()` and its
actual output is passed through the native inverse-normalization processor.

Canonical CPU inference returns a finite float32 `[1,50,6]` chunk. Repeated
independent calls reset native queues. The seed-0 output matches the preserved
reference at `rtol=0, atol=1e-6`. This validates the inference pipeline, not task success.

The machine-readable capture in `smolvla_matched_cpu_result.json` retains the
actual actions and tensor metadata. [DATA.md](DATA.md) gives the pinned pairing.
