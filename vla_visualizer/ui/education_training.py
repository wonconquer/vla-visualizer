"""Display measured training events separately from explanatory stages."""

from copy import deepcopy

import streamlit as st

from vla_visualizer.ui.education_views import render_norm
from vla_visualizer.ui.education_state import widget_value, commit_widget

TRAIN_STAGES = {
    "forward": "Forward",
    "objective": "Target / objective",
    "loss": "Loss computation",
    "backward": "Backward / gradient",
    "update": "Optimizer update",
    "next": "Next step",
}


def render_training(record, phase):
    st.caption(
        "Only the output projection is trainable. Vision, connector, language, state, and expert weights are frozen."
    )
    st.write(
        "A training forward predicts velocity once at a sampled time. The iterative generation loop is an inference path."
    )
    st.caption('Native training processes prefix and action/time suffix jointly with use_cache=False. Cached-prefix reuse belongs to inference.')
    if record is None:
        st.info(
            "Conceptual training structure. Open an experiment or run bounded training to display measurements."
        )
        st.latex(
            r"x_t=t\epsilon+(1-t)a,\quad u_t=\epsilon-a,\quad L=\operatorname{mean}_{valid}(v_\theta(x_t,t,c)-u_t)^2"
        )
        st.caption(
            "a is the normalized recorded action and ε is noise. The objective compares flow velocities, not final actions to targets."
        )
        return

    steps = record.get("visualization", {}).get("steps", [])
    result = record["result"]
    if result.get("checkpoint_reload_verified"):
        st.success(
            "Training checkpoint saved and reconstructed; fixed-observation inference matches the post-training model."
        )
        if result.get("checkpoint_kind") == "delta":
            st.caption(
                "Delta checkpoint: immutable base revision plus trained output projection. This is not a standalone full model."
            )
        if result.get("frozen_parameters_preserved"):
            st.caption(
                "All frozen-parameter hashes match before/after training. Original checkpoint and normalization hashes are preserved."
            )
    metrics = record["metrics"]["records"]
    st.caption(
        f"Trainable scope: output projection weight/bias · {record['audit']['trainable_parameters']:,} parameters"
    )
    if not metrics:
        st.info("No optimizer steps were recorded for this experiment.")
        return
    st.session_state["training_step_view"] = min(
        st.session_state.get("training_step_view", 0), len(metrics) - 1
    )

    def select_training_step():
        commit_widget(st.session_state, "training_step_view")

    index = st.selectbox(
        "Recorded optimizer step",
        list(range(len(metrics))),
        format_func=lambda i: f"Step {metrics[i]['optimizer_step']}",
        key=widget_value(st.session_state, "training_step_view"),
        on_change=select_training_step,
    )
    row = metrics[index]
    event = steps[index] if index < len(steps) else None
    st.caption(
        "Recorded experiment · one observation, batch 1, at most 3 optimizer steps · not a generalization evaluation"
    )
    if phase == "forward":
        st.write("Frozen input paths → expert hidden → trainable output projection → vθ")
        if event:
            left, right = st.columns(2)
            with left:
                render_norm(event["forward"]["expert_hidden"], "Measured frozen-expert hidden")
            with right:
                render_norm(
                    event["forward"]["predicted_velocity"], "Measured training velocity output"
                )
        else:
            st.info("Forward representations were not recorded in this legacy experiment.")
    elif phase == "objective":
        st.latex(r"x_t=t\epsilon+(1-t)a,\quad u_t=\epsilon-a")
        st.latex(r"L=\operatorname{mean}_{valid}(v_\theta(x_t,t,c)-u_t)^2")
        st.write(
            "Native conditional flow matching learns velocity between noise and normalized recorded actions."
        )
        if event:
            render_norm(event["forward"]["x_t"], "Measured xₜ at action projection input")
            objective = record["visualization"]["objective"]
            targets = objective["normalized_recorded_targets"][0]
            mask = record["audit"]["padding_mask"][0]
            valid = [i for i, padded in enumerate(mask) if not padded]
            st.caption(
                f"Recorded targets: {len(valid)} valid positions / {len(targets)} slots. The {sum(mask)} masked padding positions are excluded from native loss."
            )
            if len(valid) == 1:
                st.bar_chart(
                    {f"action {i}": [targets[valid[0]][i]] for i in range(len(targets[0]))},
                    height=170,
                )
            else:
                st.line_chart(
                    {
                        f"action {i}": [None if mask[j] else a[i] for j, a in enumerate(targets)]
                        for i in range(len(targets[0]))
                    },
                    height=170,
                )
            native = event["forward"].get("native_objective")
            if native:
                st.write("Measured native time sample:", native["time"])
                render_norm(native["noise"], "Measured native noise summary")
                render_norm(native["target_velocity"], "Measured target velocity ε−a summary")
                st.caption(
                    f"Measured x_t construction maximum error {native['x_t_formula_max_abs_error']} · target construction error {native['target_velocity_formula_max_abs_error']}"
                )
            else:
                st.caption(
                    "Legacy record: individual noise/time values were not captured. The formula describes verified source behavior."
                )
    elif phase == "loss":
        st.metric("Measured native loss", f"{row['loss']:.8f}")
        st.line_chart(metrics, x="optimizer_step", y="loss", height=180)
        st.caption(
            "Measured velocity MSE over valid robot-action dimensions and episode positions. Different noise/time samples make step-to-step loss unsuitable as evidence of improved performance."
        )
        if event:
            width = event["forward"]["predicted_velocity"]["shape"][-1]
            target_width = len(
                record["visualization"]["objective"]["normalized_recorded_targets"][0][0]
            )
            st.caption(
                f"Internal velocity width {width}; loss uses the original {target_width} robot-action dimensions; remaining channels are representation padding."
            )
    elif phase == "backward":
        st.write("Loss → velocity-output gradient → action_out_proj.weight / bias")
        st.metric(
            "Measured parameter gradient norm (before clipping)", f"{row['gradient_norm']:.6f}"
        )
        if event:
            render_norm(event["forward"]["loss_gradient_at_velocity"], "Measured dL/dvθ")
        st.info(
            "The optimizer updates only output projection. The frozen vision encoder does not change during this experiment."
        )
    elif phase == "update":
        st.write("AdamW updates output-projection parameters using measured gradients.")
        if event:
            for name, update in event["parameter_update"].items():
                st.metric(
                    name.rsplit(".", 1)[-1] + " · measured max |Δparameter|",
                    f"{update['max_abs']:.8f}",
                )
            probe = event["fixed_head_probe"]
            st.metric(
                "Measured max |Δvelocity| for fixed hidden input",
                f"{probe['max_abs_output_change']:.8f}",
            )
            left, right = st.columns(2)
            with left:
                render_norm(probe["before"], "Same head probe · before update")
            with right:
                render_norm(probe["after"], "Same head probe · after update")
            render_norm(
                probe["difference"],
                "Measured Δvelocity on the same hidden probe · vector norm per chunk position",
            )
            st.caption(
                "A pure linear-head probe holds the first recorded hidden input fixed. It is not a full-policy/vision comparison and changes neither RNG, training flags, nor optimizer state."
            )
        else:
            st.write(record["result"]["parameter_max_abs_update"])
            st.caption(
                "Legacy whole-experiment maximum update. Per-step comparison was not recorded."
            )
    else:
        st.write(
            "The next optimizer step begins a new forward with newly sampled native noise and time."
        )
        st.write(f"Recorded step: {row['optimizer_step']} / {len(metrics)} step")
        st.caption("Playback explores saved steps; it does not rerun the optimizer.")
    with st.expander("Developer details · training objective / record / modules"):
        st.json(deepcopy({"audit": record["audit"], "event": event, "metric": row}), expanded=False)
