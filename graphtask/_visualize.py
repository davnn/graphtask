# type: ignore[all]
"""Visualization of a `Task` using `pygraphviz`."""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, overload

import networkx as nx

if TYPE_CHECKING:
    from graphtask._task import Task

from graphtask._globals import STEP_ATTRIBUTE
from graphtask._step import Step, StepKind

try:
    import pygraphviz as pg
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "Graph visualisation requires 'pygraphviz' to be installed, but no installation found. See "
        "http://pygraphviz.github.io/ for guidance. If you install 'graphtask' with pip, use "
        "'pip install graphtask[visualize]' to install the optional dependencies.",
    ) from err

__all__ = ["to_pygraphviz", "Orientation"]

_GRAPH_LAYOUT = "dot"
_EDGE_ATTRIBUTES = {"color": "black", "arrowsize": 2 / 3, "style": "solid"}


class Orientation(Enum):
    VERTICAL = "TB"
    HORIZONTAL = "LR"


@overload
def to_pygraphviz(graph: Task) -> pg.AGraph:
    ...


@overload
def to_pygraphviz(graph: nx.DiGraph) -> pg.AGraph:
    ...


def to_pygraphviz(graph: Task | nx.DiGraph, *, orientation: Orientation = Orientation.VERTICAL) -> pg.AGraph:
    from graphtask._task import Task

    if isinstance(graph, Task):
        graph = graph._graph

    agraph = nx.nx_agraph.to_agraph(graph)

    # set graph attributes
    agraph.graph_attr.update(_graph_attributes(orientation))

    # set node attributes
    for v in graph.nodes:
        node = agraph.get_node(v)
        step = graph.nodes[v].get(STEP_ATTRIBUTE, None)
        node.attr.update(_node_attributes(step))

    # set edge attributes
    for u, v in graph.edges:
        edge = agraph.get_edge(u, v)
        edge.attr.update(_EDGE_ATTRIBUTES)

    agraph.layout(_GRAPH_LAYOUT)
    return agraph


def _graph_attributes(orientation: str) -> dict[str, str]:
    return {"rankdir": orientation, "bgcolor": "white"}


def _node_attributes(step: Step | None) -> dict[str, any]:
    return {"color": "black", "fillcolor": "#f0f0f0", "fontcolor": "#111111", "fontsize": 10, **_get_node_style(step)}


def _get_node_style(step: Step | None) -> dict[str, str]:
    if step is not None:
        if step.kind == StepKind.FUNCTION:
            return {"style": "filled", "shape": "box"}
        # must be mapping type
        return {"style": "filled", "shape": "box3d"}

    # otherwise
    return {"style": "rounded,dashed", "shape": "box"}
