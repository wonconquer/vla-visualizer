"""Supported instructions and their relationship to the recorded regression task."""

CANONICAL_INSTRUCTION = "Pick up the cube and place it in the box."
INSTRUCTION_EXAMPLES = (
    ("Canonical regression instruction", CANONICAL_INSTRUCTION),
    ("Sentence variation 1 · task success not verified", "Place the cube into the box."),
    ("Sentence variation 2 · task success not verified", "Pick the cube up, then put it in the box."),
    ("Sentence variation 3 · task success not verified", "Move the cube into the box."),
    ("Polite sentence variation · task success not verified", "Please pick up the cube and put it in the box."),
)


def validate_instruction(instruction: str | None, kind: str) -> None:
    if instruction is None:
        return
    permitted = {text for _, text in INSTRUCTION_EXAMPLES} if kind == "baseline" else {CANONICAL_INSTRUCTION}
    if instruction not in permitted:
        raise ValueError("The instruction is not supported for the selected task.")
