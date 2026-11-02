"""Capture training activations and gradients; additional probes use only a linear head."""

from contextlib import AbstractContextManager
from dataclasses import asdict
import hashlib
import sys

import torch
import torch.nn.functional as F

from vla_visualizer.contracts import CaptureOptions
from vla_visualizer.utils.hooks import SafeForwardHooks, summarize


def linear_probe(head, hidden):
    """Probe a fixed CPU input without changing RNG, training mode, optimizer state or hooks."""
    with torch.no_grad():
        return F.linear(hidden, head.weight.detach().cpu(), head.bias.detach().cpu()).detach()


class TrainingCapture(AbstractContextManager):
    def __init__(self, policy):
        self.policy = policy
        self.options = CaptureOptions(enabled=True, max_token_norms=50)
        self.fixed_hidden = None
        self.gradient_handles = []
        self.event = {}
        self.hooks = None
        self.previous_profile = None

    def _profile(self, frame, event, output):
        if (
            event == "return"
            and frame.f_code is self.policy.model.forward.__func__.__code__
            and output is not None
        ):
            values = frame.f_locals
            self.event["native_objective"] = {
                "source": "actual native SmolVLAFlowMatching.forward return locals",
                "noise": self.small(values["noise"]),
                "time": values["time"].detach().cpu().tolist(),
                "normalized_padded_target": self.small(values["actions"]),
                "target_velocity": self.small(values["u_t"]),
                "unreduced_loss": self.small(output),
                "x_t_formula_max_abs_error": float(
                    (
                        values["x_t"]
                        - (
                            values["time_expanded"] * values["noise"]
                            + (1 - values["time_expanded"]) * values["actions"]
                        )
                    )
                    .abs()
                    .max()
                ),
                "target_velocity_formula_max_abs_error": float(
                    (values["u_t"] - (values["noise"] - values["actions"])).abs().max()
                ),
            }

    def small(self, tensor):
        return asdict(summarize(tensor, self.options))

    def _input(self, module, args, kwargs, output):
        self.event["x_t"] = self.small(args[0])

    def _output(self, module, args, kwargs, output):
        hidden = args[0]
        if hidden.numel() > 100_000:
            raise ValueError("Fixed head-probe input exceeds its size limit")
        if self.fixed_hidden is None:
            self.fixed_hidden = hidden.detach().float().cpu().clone()
        self.event["expert_hidden"] = self.small(hidden)
        self.event["predicted_velocity"] = self.small(output)
        if output.requires_grad:

            def gradient(value):
                self.event["loss_gradient_at_velocity"] = self.small(value)

            self.gradient_handles.append(output.register_hook(gradient))

    def begin_step(self):
        self.event = {}

    def __enter__(self):
        self.previous_profile = sys.getprofile()
        if self.previous_profile is not None:
            raise RuntimeError("Cannot capture native training while another profiler is active.")
        self.hooks = SafeForwardHooks(
            self.policy,
            {
                "model.action_in_proj": self._input,
                "model.action_out_proj": self._output,
            },
        )
        self.hooks.__enter__()
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *args):
        sys.setprofile(self.previous_profile)
        try:
            for handle in self.gradient_handles:
                handle.remove()
            self.gradient_handles.clear()
        finally:
            self.hooks.__exit__(*args)

    def probe_record(self, head, before):
        after = linear_probe(head, self.fixed_hidden)
        return {
            "input_sha256": hashlib.sha256(self.fixed_hidden.numpy().tobytes()).hexdigest(),
            "scope": "Linear head probe using fixed hidden input from the first actual training forward; not a full-policy evaluation.",
            "input": self.small(self.fixed_hidden),
            "before": self.small(before),
            "after": self.small(after),
            "difference": self.small(after - before),
            "max_abs_output_change": (after - before).abs().max().item(),
            "rng_consumed": False,
            "model_training_flags_changed": False,
        }
