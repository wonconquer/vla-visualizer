"""Bounded activation summaries and exception-safe, read-only forward hooks."""

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import Any
from time import perf_counter

import torch
from torch import Tensor, nn

from vla_visualizer.contracts import CaptureOptions, CapturedStage, TensorSummary


def summarize(tensor: Tensor, options: CaptureOptions) -> TensorSummary:
    """Measure the actual tensor; keep only bounded Python numbers on the CPU."""
    value = tensor.detach().to(dtype=torch.float32)
    if not value.numel() or not torch.isfinite(value).all():
        raise ValueError("Captured tensors must be nonempty and finite.")
    flat = value.reshape(-1)
    indices, norms, norm_stats = (), (), {}
    if value.ndim == 3:
        row_norms = torch.linalg.vector_norm(value, dim=-1).reshape(-1)
        selected = torch.linspace(
            0, row_norms.numel() - 1,
            min(options.max_token_norms, row_norms.numel()), device=value.device,
        ).long()
        indices = tuple(selected.cpu().tolist())
        norms = tuple(row_norms[selected].cpu().tolist())
        norm_stats = {
            "min": row_norms.min().item(), "max": row_norms.max().item(),
            "mean": row_norms.mean().item(), "std": row_norms.std(correction=0).item(),
        }
    return TensorSummary(
        shape=tuple(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device),
        numel=tensor.numel(), minimum=value.min().item(), maximum=value.max().item(),
        mean=value.mean().item(), std=value.std(correction=0).item(),
        l2_norm=torch.linalg.vector_norm(flat).item(),
        preview=tuple(flat[:options.max_tensors].cpu().tolist()),
        values=tuple(flat.cpu().tolist()) if flat.numel() <= 64 else None,
        token_indices=indices, token_norms=norms, token_norm_statistics=norm_stats,
    )


class CaptureBuffer:
    """Per-inference store; no model or tensor references survive capture."""

    def __init__(self, options: CaptureOptions) -> None:
        self.options = options
        self.records: list[CapturedStage] = []
        self.summary_seconds = 0.0

    def add(
        self, stage_name: str, module_name: str, description: str, source: str,
        inputs: Mapping[str, Tensor], outputs: Mapping[str, Tensor],
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if len(self.records) >= self.options.max_records:
            raise ValueError("Capture record budget exceeded; no records silently discarded.")
        start = perf_counter()
        self.records.append(CapturedStage(
            stage_name, module_name, description, source,
            {name: summarize(value, self.options) for name, value in inputs.items()},
            {name: summarize(value, self.options) for name, value in outputs.items()},
            dict(context or {}),
        ))
        self.summary_seconds += perf_counter() - start


class SafeForwardHooks(AbstractContextManager):
    """Register only requested modules, undo partial setup, remove on any exit.

    Callbacks receive kwargs and may inspect but must not mutate inputs/outputs.
    This wrapper always returns None to preserve the module result.
    """

    def __init__(
        self, model: nn.Module,
        callbacks: Mapping[str, Callable[[nn.Module, tuple, dict, Any], None]],
    ) -> None:
        modules = dict(model.named_modules())
        unknown = set(callbacks) - set(modules)
        if unknown:
            raise ValueError(f"Capture modules do not exist: {sorted(unknown)}")
        self._modules = {name: modules[name] for name in callbacks}
        self._callbacks = callbacks
        self._handles = []
        self._entered = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Hook context can only be entered once.")
        self._entered = True
        try:
            for name, callback in self._callbacks.items():
                def hook(module, args, kwargs, output, callback=callback):
                    callback(module, args, kwargs, output)
                    return None
                self._handles.append(self._modules[name].register_forward_hook(hook, with_kwargs=True))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        # Drop callback closures too: model-specific callbacks may reference their
        # capture session. Keeping them would retain old summaries through a cycle.
        self._callbacks = {}
        self._modules = {}
        return False
