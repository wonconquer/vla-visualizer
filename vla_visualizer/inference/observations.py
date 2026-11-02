"""Read a local observation manifest; no model-specific conversion or downloads."""

import hashlib
import json
from pathlib import Path

from vla_visualizer.contracts import Observation
from vla_visualizer.utils.images import decode_image


def read_observation(path: Path) -> Observation:
    """Load camera images, instruction, and measured state from JSON."""
    data = json.loads(path.read_text())
    return Observation(
        images={
            key: decode_image((path.parent / name).read_bytes())
            for key, name in data["images"].items()
        },
        instruction=data["instruction"],
        state=tuple(data["state"]),
        provenance=data.get("provenance", {}),
    )


def observation_fingerprint(observation: Observation) -> str:
    """Bind displayed actions to the exact current observation."""
    digest = hashlib.sha256()
    digest.update(json.dumps(
        [observation.instruction, observation.state, dict(observation.provenance)], sort_keys=True
    ).encode())
    for name, image in sorted(observation.images.items()):
        digest.update(name.encode())
        digest.update(str(image.size).encode())
        digest.update(image.convert("RGB").tobytes())
    return digest.hexdigest()
