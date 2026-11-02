"""Build an explicitly illustrative VLA data-flow graph."""

from graphviz import Digraph

from vla_visualizer.contracts import ModelStructure, Stage


def placeholder_structure() -> ModelStructure:
    """Describe conceptual stages, not a loaded model or an execution trace."""
    return ModelStructure(
        stages=(
            Stage("image", "Image observation"),
            Stage("preprocess", "Vision preprocessing"),
            Stage("vision", "Vision encoder"),
            Stage("features", "Visual tokens / features"),
            Stage("instruction", "Language instruction"),
            Stage("language", "Tokenization / language processing"),
            Stage("state", "Robot state (model-dependent)"),
            Stage("fusion", "Multimodal processing"),
            Stage("decoder", "Action head / decoder"),
            Stage("action", "Predicted actions (unavailable)"),
        ),
        edges=(
            ("image", "preprocess"),
            ("preprocess", "vision"),
            ("vision", "features"),
            ("features", "fusion"),
            ("instruction", "language"),
            ("language", "fusion"),
            ("state", "fusion"),
            ("fusion", "decoder"),
            ("decoder", "action"),
        ),
        provenance="placeholder",
    )


def architecture_dot(structure: ModelStructure) -> str:
    """Produce DOT text for browser rendering; no system Graphviz binary is used."""
    graph = Digraph("vla_flow")
    graph.attr(rankdir="TB")
    if structure.provenance == "placeholder":
        graph.attr(label="PLACEHOLDER: conceptual flow only", labelloc="t")
        graph.attr("node", shape="box", style="dashed")
    else:
        graph.attr("node", shape="box")
    for stage in structure.stages:
        graph.node(stage.stage_id, stage.label)
    for source, target in structure.edges:
        graph.edge(source, target)
    return graph.source
