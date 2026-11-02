"""Safety and numerical contract tests on small tensors; no pretrained downloads."""

from dataclasses import asdict
import json
import weakref

import pytest
import torch
from torch import nn

from vla_visualizer.contracts import CaptureOptions
from vla_visualizer.utils.hooks import CaptureBuffer, SafeForwardHooks, summarize
from vla_visualizer.adapters.smolvla_inspection import SmolVLAInspection


@pytest.mark.parametrize("kwargs", [
    {"max_tensors": 0}, {"max_tensors": 65}, {"max_token_norms": 1025},
    {"max_token_norms": 0}, {"max_records": 65}, {"max_records": 0},
])
def test_capture_limits_are_enforced(kwargs) -> None:
    with pytest.raises(ValueError):
        CaptureOptions(**kwargs)


def test_measurements_are_exact_bounded_detached_and_serializable() -> None:
    tensor = torch.arange(120.0).reshape(1, 20, 6).requires_grad_()
    summary = summarize(tensor, CaptureOptions(max_tensors=3, max_token_norms=5))
    assert summary.shape == (1, 20, 6)
    assert summary.dtype == "torch.float32" and summary.device == "cpu"
    assert summary.minimum == 0 and summary.maximum == 119
    assert summary.mean == 59.5
    assert summary.std == pytest.approx(tensor.detach().std(correction=0).item())
    assert summary.l2_norm == pytest.approx(tensor.detach().norm().item())
    assert summary.preview == (0, 1, 2) and summary.values is None
    assert len(summary.token_norms) == len(summary.token_indices) == 5
    expected = tensor.detach().norm(dim=-1).flatten()[list(summary.token_indices)]
    torch.testing.assert_close(torch.tensor(summary.token_norms), expected)
    encoded = json.dumps(asdict(summary), allow_nan=False)
    with torch.no_grad():
        tensor.fill_(999)
    assert json.dumps(asdict(summary), allow_nan=False) == encoded
    assert summarize(torch.arange(6), CaptureOptions()).values == (0, 1, 2, 3, 4, 5)


@pytest.mark.parametrize("value", [torch.empty(0), torch.tensor([float("nan")]), torch.tensor([float("inf")])])
def test_invalid_tensor_rejected(value) -> None:
    with pytest.raises(ValueError, match="nonempty and finite"):
        summarize(value, CaptureOptions())


def test_budget_exhaustion_is_explicit() -> None:
    buffer = CaptureBuffer(CaptureOptions(max_records=1))
    buffer.add("small", "", "", "model_input", {}, {"value": torch.ones(1)})
    with pytest.raises(ValueError, match="budget exceeded"):
        buffer.add("small", "", "", "model_input", {}, {"value": torch.ones(1)})


def test_readonly_hooks_preserve_output_and_existing_hooks_across_repeats() -> None:
    model = nn.Sequential(nn.Linear(6, 8), nn.Linear(8, 3)).eval()
    inputs = torch.arange(6.0).unsqueeze(0)
    baseline = model(inputs).detach()
    existing = model[0].register_forward_hook(lambda *args: None)
    try:
        for _ in range(3):
            buffer = CaptureBuffer(CaptureOptions())
            def callback(module, args, kwargs, output):
                buffer.add("linear", "0", "actual linear", "forward_hook",
                           {"x": args[0]}, {"y": output})
                return torch.zeros_like(output)  # Wrapper must ignore callback return.
            with SafeForwardHooks(model, {"0": callback}):
                torch.testing.assert_close(model(inputs), baseline, rtol=0, atol=0)
                assert len(model[0]._forward_hooks) == 2
            assert len(model[0]._forward_hooks) == 1
            assert len(buffer.records) == 1
    finally:
        existing.remove()


@pytest.mark.parametrize("failure", ["callback", "forward"])
def test_hooks_removed_on_execution_exception(failure) -> None:
    class Probe(nn.Module):
        def forward(self, *, value):
            if failure == "forward":
                raise RuntimeError("forward failed")
            return value + 1
    model = Probe()
    def callback(module, args, kwargs, output):
        assert not args and kwargs["value"].shape == (2,)
        raise RuntimeError("callback failed")
    context = SafeForwardHooks(model, {"": callback})
    with pytest.raises(RuntimeError, match="failed"):
        with context:
            model(value=torch.ones(2))
    assert not model._forward_hooks and not context._callbacks and not context._modules
    reference = weakref.ref(model)
    del model
    assert reference() is None


def test_partial_hook_registration_rolls_back(monkeypatch) -> None:
    model = nn.Sequential(nn.Identity(), nn.Identity())
    def fail(*args, **kwargs):
        raise RuntimeError("registration failed")
    monkeypatch.setattr(model[1], "register_forward_hook", fail)
    with pytest.raises(RuntimeError, match="registration failed"):
        with SafeForwardHooks(model, {"0": lambda *args: None, "1": lambda *args: None}):
            pass
    assert not model[0]._forward_hooks


def test_unknown_module_rejected_before_hook_registration() -> None:
    model = nn.Sequential(nn.Identity())
    with pytest.raises(ValueError, match="do not exist"):
        SafeForwardHooks(model, {"0": lambda *args: None, "missing": lambda *args: None})
    assert not model[0]._forward_hooks


def test_smolvla_rejects_unverified_or_bypassed_module_points() -> None:
    with pytest.raises(ValueError, match="approved inspection modules"):
        SmolVLAInspection(None, CaptureOptions(module_names=("model.vlm_with_expert",)), ())
