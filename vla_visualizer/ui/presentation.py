"""English presentation of immutable historical records; source files remain untouched."""

import json
from pathlib import Path
import re

LEGACY = json.loads(Path(__file__).with_name("legacy_translations.json").read_text())
LEGACY.update(
    {
        "\uc800\uc7a5 \uc791\uc5c5 \uc218 \uc81c\ud55c\uc785\ub2c8\ub2e4. \uad00\ub9ac\uc790\uac00 \uc644\ub8cc artifact\ub97c \ubcf4\uad00\xb7\uc815\ub9ac\ud574\uc57c \ud569\ub2c8\ub2e4.": "Historical job-count limit reached under the previous retained-record policy. Existing artifacts are preserved.",
        "\uc804\uccb4 artifact \uc800\uc7a5 \ud55c\ub3c4\ub97c \ucd08\uacfc\ud588\uc2b5\ub2c8\ub2e4.": "Historical total artifact byte-budget limit exceeded.",
    }
)
MEASURED_FIELDS = {
    "instruction",
    "recorded_instruction",
    "task",
    "token",
    "tokens",
    "token_ids",
    "module",
    "module_name",
    "camera",
    "checkpoint",
    "revision",
    "record_id",
    "run_id",
    "stage",
    "stage_name",
    "action_names",
    "state_names",
}


def present(value):
    if isinstance(value, str) and re.search("[\uac00-\ud7a3]", value):
        if value in LEGACY:
            return LEGACY[value]
        rendered = value
        for original, english in sorted(
            LEGACY.items(), key=lambda pair: len(pair[0]), reverse=True
        ):
            rendered = rendered.replace(original, english)
        return (
            rendered
            if not re.search("[\uac00-\ud7a3]", rendered)
            else "Historical description: original text is preserved in the downloaded source artifact."
        )
    if isinstance(value, dict):
        return {
            key: item if key in MEASURED_FIELDS else present(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [present(item) for item in value]
    return value
