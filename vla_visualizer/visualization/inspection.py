"""Render-independent diagrams of source-verified capture dependencies."""

from graphviz import Digraph


def inspection_dot(spec: dict, selected: str) -> str:
    graph = Digraph("captured_inference")
    graph.attr(
        rankdir="TB", label="Educational replay of a completed run · dependency graph", labelloc="t"
    )
    for stage in spec["stages"]:
        graph.node(
            stage["id"],
            stage["label"],
            shape="box",
            style="filled",
            fillcolor="#b8ddff" if stage["id"] == selected else "#f4f4f4",
            penwidth="3" if stage["id"] == selected else "1",
        )
    for source, target, relationship in spec["edges"]:
        graph.edge(
            source,
            target,
            label=relationship,
            style="dashed" if "cached" in relationship else "solid",
        )
    return graph.source
