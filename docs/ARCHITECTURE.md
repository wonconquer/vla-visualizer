# Architecture and inspection boundaries

The Streamlit UI submits explicit jobs. A supervisor owns the model lease,
deadline, cancellation, and artifact budget. The worker uses the SmolVLA adapter
for checkpoint authentication, native processors, frozen inference, and capture.
Views read the completed CPU replay snapshot; they never register model hooks.

## Recorded computation

The same vision encoder processes the top camera and then the wrist camera.
Each call produces `[1,1024,768]` contextual patch tokens. The connector groups
4x4 patches into `[1,64,960]` tokens. The native resize maps 480x640 to 384x512,
then pads 128 pixels at the top to form a 512x512 canvas. Encoder tokens are
row-major without a CLS token; connector indices follow the inspected pixel shuffle.
Spatial indices are not attribution or attention measurements.

Instruction IDs and masks have shape `[1,48]`, with 12 valid tokens for the
canonical sentence including its newline. State is normalized using the native
checkpoint statistics, right-padded from six to 32 coordinates, and projected
to width 960. The prefix order is:

| Positions, end exclusive | Representation |
| --- | --- |
| 0-64 | Top connector tokens |
| 64-128 | Wrist connector tokens |
| 128-176 | Language slots, including masked padding |
| 176-177 | Projected state token |

The prefix is processed once. Its cached context conditions the action expert;
the state token and the action/time-conditioned suffix are different inputs.
The cache itself is not retained. Attention masks describe allowed positions,
not returned attention probabilities.

Native generation uses ten Euler steps from noise at t=1 to the integrated
action state at t=0. Expert hidden states have width 720; the velocity projection
returns `[1,50,32]`. The policy slices to six action dimensions, then the native
postprocessor inverse-normalizes the actual `[1,50,6]` result.

## Capture and replay

Basic capture uses eight selected module hooks. Extended capture adds a final
prefix norm hook and a call-scoped read-only Python profiler for native loop
inputs, time, velocity returns, and updated states. Native functions are not patched.
Hooks and profiling are removed after the call, including exception paths.

Large hidden summaries are retained only at iterations 0, 4, and 9. Extended
capture retains all ten small action states and at most the first 16 channels
per selected token. Original shape, dtype/device, module, camera, record ID,
iteration, and summary method remain attached to each representation.

The diagram shows data dependencies. The recorded-call view shows measured
module invocations. Educational replay groups them into stages; these three
orders should not be conflated. Playback durations are not GPU timings.

## Jobs and checkpoints

One model job is active at a time. The recent primary-record window is 20;
terminal overflow is logically archived without moving or deleting artifacts.
All retained records still count toward the separate total bound of 200 and
the 4 GiB artifact budget. Evaluation has its own bounded capacity and reuses
identical pending/completed requests.

Tiny training updates only `model.action_out_proj`, 23,072 parameters. Base
weights, normalization, and the other frozen parameters remain unchanged.
The saved delta includes the base revision and artifact hashes. Evaluation
reconstructs it in a fresh worker and compares with a same-observation probe.
