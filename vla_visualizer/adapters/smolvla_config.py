"""Explicit checkpoint/domain approval; adding a pin requires a new pairing audit."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SmolVLACheckpoint:
    """A SmolVLA pin with authenticated processors and a documented input domain."""

    repo_id: str
    revision: str
    dataset: str
    dataset_revision: str
    camera_map: tuple[tuple[str, str], ...]
    joint_names: tuple[str, ...]
    artifact_sha256: tuple[tuple[str, str], ...]
    weights_sha256: str
    coordinate_convention: str


MVP_CHECKPOINT = SmolVLACheckpoint(
    repo_id="Zekai-Chen/smolvla_so100_pickplace_ft_2cam",
    revision="77172eaffe82c0897945607bfe818b5b164eb57e",
    dataset="lerobot/svla_so100_pickplace",
    dataset_revision="87169e502b742f989b96c952bff7af55e248d467",
    camera_map=(
        ("observation.images.top", "observation.images.camera1"),
        ("observation.images.wrist", "observation.images.camera2"),
    ),
    joint_names=(
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
    ),
    artifact_sha256=(
        ("config.json", "ee725c2a4b4cce0ec41af6a8f8c94d221473879442ac35aa8495f3ab304264eb"),
        (
            "policy_preprocessor.json",
            "4c58f664eefd33771cca3a5c093e07dce8a8e8d5546d275ac6681d7af9566253",
        ),
        (
            "policy_postprocessor.json",
            "2b78bb742065288df2ec63b0ee35f97a1f6950171cf2272f7e34b1fe3873b17b",
        ),
        (
            "policy_preprocessor_step_5_normalizer_processor.safetensors",
            "2ee26fd4a765912f77b60c21f7c8ae9fd68d906966e2f3fe8d08009baa415e84",
        ),
        (
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            "2ee26fd4a765912f77b60c21f7c8ae9fd68d906966e2f3fe8d08009baa415e84",
        ),
    ),
    weights_sha256="e738f888458e9bbd0fef986f9ff33460ba89dc4f0ec8dcfc7f25513ba1ced4a4",
    coordinate_convention=(
        "Legacy pre-PR-777 SO100: first five joints in calibrated degrees, "
        "gripper in calibrated percent (nominal 0–100). State is measured follower "
        "position; actions are absolute target positions."
    ),
)
