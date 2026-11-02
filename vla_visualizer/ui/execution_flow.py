"""Track replay position separately from recorded module invocation order."""

LOOP_PHASES = (
    "action_state",
    "time_embedding",
    "conditioning",
    "expert",
    "velocity",
    "solver_update",
    "loop_next",
)


def iterations(data):
    if data:
        extended = data["capture"]["intermediates"].get("extended_capture", {})
        if extended.get("full_loop"):
            return [x["iteration"] for x in extended["full_loop"]]
        return list(data["capture"]["intermediates"].get("selected_flow_steps", []))
    return [0]


def step_position(stage, iteration, stages, available, direction):
    if not stages:
        return None, 0
    if stage not in stages:
        stage = stages[0]
    available = available or [0]
    if iteration not in available:
        iteration = available[0]
    if stage == "loop_next" and direction > 0 and "action_state" in stages and "actions" in stages:
        index = available.index(iteration)
        return (
            ("action_state", available[index + 1])
            if index < len(available) - 1
            else ("actions", iteration)
        )
    if (
        stage == "action_state"
        and direction < 0
        and iteration != available[0]
        and "loop_next" in stages
    ):
        return "loop_next", available[available.index(iteration) - 1]
    if stage == "actions" and direction < 0 and "loop_next" in stages:
        return "loop_next", available[-1]
    index = max(0, min(len(stages) - 1, stages.index(stage) + direction))
    return stages[index], available[-1] if stages[index] == "actions" else iteration


def call_events(data):
    if not data:
        return []
    capture = data["capture"]
    extended = capture["intermediates"].get("extended_capture", {})
    observed = extended.get("observed_module_order")
    if observed is None:
        observed = [
            {"stage": r["stage_name"], "module": r["module_name"], **r.get("context", {})}
            for r in capture["intermediates"]["stages"]
            if r["source"] == "forward_hook"
        ]
    mapping = {
        "language": "language_embedding",
        "state": "state_project",
        "multimodal_prefix": "prefix",
        "vlm_prefix_output": "vlm_context",
        "flow_state": "action_state",
        "time_conditioned_actions": "conditioning",
        "action_velocity": "velocity",
    }
    result = []
    cameras = list(data["view_spec"]["camera_to_native"])
    counters = {}
    for event in observed:
        source = event["stage"]
        call = event.get("call_index", 0)
        if source in ("vision", "connector"):
            camera = event.get("camera", cameras[call])
            stage = ("top_" if camera == cameras[0] else "wrist_") + (
                "vision" if source == "vision" else "connector"
            )
        else:
            camera = event.get("camera")
            stage = mapping.get(source)
        if not stage:
            continue
        flow = (
            call
            if source in ("flow_state", "time_conditioned_actions", "action_velocity")
            else None
        )
        result.append(
            {
                "stage": stage,
                "module": event["module"],
                "camera": camera,
                "iteration": flow,
                "source": "actual module hook invocation; no per-stage timing",
            }
        )
        counters[source] = counters.get(source, 0) + 1
    return result
