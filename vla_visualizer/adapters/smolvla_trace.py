"""Read-only profiling scoped to one native inference call."""

import sys


class NativeExecutionTrace:
    def __init__(self, policy):
        from lerobot.policies.common.flow_matching import euler_integrate

        self.policy = policy
        self.denoise_code = policy.model.denoise_step.__func__.__code__
        self.prefix_code = policy.model.embed_prefix.__func__.__code__
        self.vlm_code = policy.model.vlm_with_expert.forward.__func__.__code__
        self.euler_code = euler_integrate.__code__
        self.events = []
        self.timeline = []
        self.prefix = {}
        self.previous = None

    @staticmethod
    def action(tensor):
        if tensor.numel() > 2048 or tensor.ndim != 3:
            raise ValueError("Full-loop action capture exceeds its bound")
        return tensor.detach().float().cpu().tolist()

    def _profile(self, frame, event, arg):
        code = frame.f_code
        if code is self.denoise_code:
            if event == "call":
                if len(self.events) >= 16:
                    raise ValueError("Loop capture iteration bound exceeded")
                x = self.action(frame.f_locals["x_t"])
                if self.events:
                    self.events[-1]["x_after"] = x
                    self.events[-1]["x_after_source"] = "next native denoise_step input"
                k = len(self.events)
                self.events.append(
                    {
                        "iteration": k,
                        "time": float(frame.f_locals["timestep"][0].item()),
                        "time_source": "actual native denoise_step timestep argument",
                        "dt": -1.0 / self.policy.config.num_steps,
                        "dt_source": "installed euler_integrate source",
                        "x_before": x,
                        "x_before_source": "actual native denoise_step input",
                    }
                )
                self.timeline.append({"stage": "denoise_begin", "iteration": k})
            elif event == "return" and arg is not None:
                self.events[-1]["velocity"] = self.action(arg)
                self.events[-1]["velocity_source"] = (
                    "actual native denoise_step return after action_out_proj"
                )
                self.timeline.append(
                    {"stage": "velocity_return", "iteration": len(self.events) - 1}
                )
        elif code is self.euler_code and event == "return" and arg is not None and self.events:
            self.events[-1]["x_after"] = self.action(arg)
            self.events[-1]["x_after_source"] = "actual native euler_integrate return"
            self.timeline.append(
                {"stage": "solver_final_return", "iteration": len(self.events) - 1}
            )
        elif code is self.prefix_code and event == "return" and arg is not None:
            self.prefix = {
                "embedding_shape": list(arg[0].shape),
                "padding_mask": arg[1].detach().cpu().tolist(),
                "block_boundaries": arg[2].detach().cpu().tolist(),
                "source": "actual embed_prefix return; masks are not attention probabilities",
            }
            self.timeline.append({"stage": "prefix_constructed"})
        elif code is self.vlm_code and event == "call":
            inputs = frame.f_locals.get("inputs_embeds")
            if inputs and inputs[0] is not None and inputs[1] is None:
                mask = frame.f_locals["attention_mask"]
                self.prefix["image_language_queries_can_see_state"] = bool(
                    mask[0, :-1, -1].any().item()
                )
                self.prefix["state_visible_key_count"] = int(mask[0, -1].sum().item())
                self.timeline.append({"stage": "vlm_prefix_processing_begin"})
        elif code is self.vlm_code and event == "return" and arg is not None:
            inputs = frame.f_locals.get("inputs_embeds")
            if inputs and inputs[0] is not None and inputs[1] is None:
                cache = arg[1]
                self.prefix["cache_metadata"] = {
                    "available": cache is not None,
                    "sequence_length": int(cache.get_seq_length()) if cache is not None else None,
                    "layer_count": len(cache.layers) if cache is not None else None,
                    "retained_tensors": False,
                }
                self.timeline.append({"stage": "cached_context_ready"})

    def __enter__(self):
        self.previous = sys.getprofile()
        if self.previous is not None:
            raise RuntimeError(
                "Cannot begin capture while another Python profiler is active."
            )
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *args):
        sys.setprofile(self.previous)
        return False
