"""Persistent replay selection, independent of Streamlit's disposable widget keys."""

from time import monotonic

POSITION_KEYS = (
    "edu_stage",
    "edu_flow",
    "edu_call",
    "edu_action",
    "training_step_view",
    "edu_view",
)

# These controls follow the timer while playing. Late browser widget echoes
# must not be interpreted as a new user request to jump backwards and pause.
PLAYBACK_FOLLOWERS = {"edu_stage", "edu_flow", "edu_call"}


def ensure_selection(
    state, ids, available=None, calls=None, identity=None, action_count=1, training_steps=1
):
    if state.get("education_selection_schema") != 1:
        # Streamlit 1.50 supports self-assignment to detach an old widget-owned key
        # from cleanup. Preserve legacy values once; new widgets use underscore keys.
        for key in (*POSITION_KEYS, "edu_speed"):
            if key in state:
                state[key] = state[key]
        state["education_selection_schema"] = 1
    ids = list(dict.fromkeys(ids))
    available = list(available or [0])
    calls = [event for event in (calls or []) if event.get("stage") in ids]
    old = state.get("education_identity")
    if identity is not None and identity != old:
        memory = dict(state.get("education_positions", {}))
        if old is not None:
            memory[old] = {key: state[key] for key in POSITION_KEYS if key in state}
        saved = memory.get(identity, {})
        for key in POSITION_KEYS:
            state.pop(key, None)
        state.update(saved)
        state.update(education_identity=identity, edu_playing=False, edu_tick=monotonic())
        state["edu_pause_reason"] = "Selected run or mode changed"
        state["education_positions"] = dict(list(memory.items())[-8:])
    state["education_stage_ids"] = ids
    state["edu_iterations"] = available
    state["edu_calls"] = calls
    if state.get("edu_stage") not in ids:
        state["edu_stage"] = ids[0] if ids else None
    if state.get("edu_flow") not in available:
        state["edu_flow"] = available[0]
    for key, count in [
        ("edu_call", len(calls)),
        ("edu_action", action_count),
        ("training_step_view", training_steps),
    ]:
        value = state.get(key, 0)
        state[key] = max(0, min(value if type(value) is int else 0, max(0, count - 1)))
    if state.get("edu_view") not in ("Information flow", "Recorded invocation order"):
        state["edu_view"] = "Information flow"
    if state.get("edu_speed") not in (0.6, 1.0, 1.5, 2.5, 4.0):
        state["edu_speed"] = 1.5
    if not ids:
        state["edu_playing"] = False
        state["edu_pause_reason"] = "No stages available"
    state.setdefault("edu_playing", False)
    state.setdefault("edu_tick", monotonic())
    return ids


def widget_value(state, key):
    """Call before widget creation, including on every fragment run."""
    widget = "_" + key
    state[widget] = state[key]
    return widget


def commit_widget(state, key):
    if state.get("edu_playing") and key in PLAYBACK_FOLLOWERS:
        return False
    if "_" + key in state:
        state[key] = state["_" + key]
    state.update(
        edu_playing=False,
        edu_tick=monotonic(),
        edu_pause_reason="Selection changed: " + key.removeprefix("edu_"),
    )
    return True
