"""Store deltas against an immutable base and load supported standalone checkpoints."""

import json
from pathlib import Path

from vla_visualizer.adapters.smolvla import CHECKPOINT_FILES, file_sha256
from vla_visualizer.adapters.smolvla_config import MVP_CHECKPOINT

DELTA_NAMES = {"model.action_out_proj.weight", "model.action_out_proj.bias"}


def save_delta(directory, policy, base_hashes, configuration):
    from safetensors.torch import save_file
    from vla_visualizer.execution.jobs import atomic_json

    directory.mkdir()
    tensors = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in policy.named_parameters()
        if name in DELTA_NAMES
    }
    if set(tensors) != DELTA_NAMES:
        raise ValueError("Delta parameter-name mismatch")
    save_file(tensors, str(directory / "delta.safetensors"))
    manifest = {
        "schema_version": 1,
        "kind": "delta",
        "standalone": False,
        "base_checkpoint": MVP_CHECKPOINT.repo_id,
        "base_revision": MVP_CHECKPOINT.revision,
        "base_artifact_sha256": base_hashes,
        "delta_file": "delta.safetensors",
        "delta_sha256": file_sha256(directory / "delta.safetensors"),
        "parameter_names": sorted(DELTA_NAMES),
        "normalization_source": "immutable base checkpoint native processors",
    }
    atomic_json(directory / "checkpoint_manifest.json", manifest)
    atomic_json(directory / "training_config.json", configuration)
    return manifest


def load_trained_checkpoint(adapter, directory):
    import torch
    from safetensors.torch import load_file

    directory = Path(directory)
    manifest_path = directory / "checkpoint_manifest.json"
    if not manifest_path.exists():
        # Existing Phase 5 standalone checkpoint contract remains supported.
        tensors = load_file(str(directory / "model.safetensors"))
        adapter._policy.load_state_dict(tensors, strict=True)
        return {
            "kind": "standalone",
            "payload_sha256": file_sha256(directory / "model.safetensors"),
        }
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "delta"
        or manifest.get("standalone") is not False
    ):
        raise ValueError("Unsupported checkpoint contract")
    if (
        manifest["base_checkpoint"] != MVP_CHECKPOINT.repo_id
        or manifest["base_revision"] != MVP_CHECKPOINT.revision
    ):
        raise ValueError("Delta/base revision mismatch")
    if (
        manifest["parameter_names"] != sorted(DELTA_NAMES)
        or manifest["delta_file"] != "delta.safetensors"
    ):
        raise ValueError("Delta path/parameter validation failed")
    if file_sha256(directory / "delta.safetensors") != manifest["delta_sha256"]:
        raise ValueError("Delta checksum mismatch")
    if set(manifest["base_artifact_sha256"]) != set(CHECKPOINT_FILES):
        raise ValueError("All base-artifact hashes are required.")
    for name, digest in manifest["base_artifact_sha256"].items():
        if (
            "/" in name
            or name.startswith(".")
            or file_sha256(Path(adapter._snapshot_path) / name) != digest
        ):
            raise ValueError("Base artifact checksum mismatch")
    delta = load_file(str(directory / "delta.safetensors"))
    reference = adapter._policy.state_dict()
    if set(delta) != DELTA_NAMES or any(
        delta[k].shape != reference[k].shape
        or delta[k].dtype != reference[k].dtype
        or not torch.isfinite(delta[k]).all()
        for k in delta
    ):
        raise ValueError("Delta tensor schema/value mismatch")
    incompatible = adapter._policy.load_state_dict(delta, strict=False)
    if (
        incompatible.unexpected_keys
        or set(incompatible.missing_keys) != set(reference) - DELTA_NAMES
    ):
        raise ValueError("Delta reconstruction mismatch")
    return {
        "kind": "delta",
        "payload_sha256": manifest["delta_sha256"],
        "standalone": False,
        "base_revision": manifest["base_revision"],
    }
