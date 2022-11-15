# type: ignore
"""
Visualization of a `Task` using `pygraphviz`.
"""
from typing import Union, overload

from enum import Enum

import networkx as nx

from graphtask import Task
from graphtask._task import MAP_ATTRIBUTE, TYPE_ATTRIBUTE, NodeType

try:
    import pygraphviz as pg
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "Graph visualisation requires 'pygraphviz' to be installed, but no installation found. See "
        + "http://pygraphviz.github.io/ for guidance. If you install 'graphtask' with pip, use "
        + "'pip install graphtask[visualize]' to install the optional dependencies."
    ) from err

__all__ = ["to_pygraphviz", "Orientation"]

_GRAPH_LAYOUT = "dot"
_GRAPH_ATTRIBUTES = lambda orientation: {"rankdir": orientation, "bgcolor": "white"}
_NODE_ATTRIBUTES = lambda node_type: {
    "color": "black",
    "fillcolor": "#f0f0f0",
    "fontcolor": "#111111",
    "fontsize": 10,
    **get_node_style(node_type),
}
_EDGE_ATTRIBUTES = lambda is_map: {"color": "black", "arrowsize": 2 / 3, "style": "dotted" if is_map else "solid"}


def get_node_style(node_type: NodeType) -> dict[str, str]:
    if node_type == NodeType.ATTRIBUTE:
        return {"style": "rounded", "shape": "box"}
    elif node_type == NodeType.FUNCTION:
        return {"style": "filled", "shape": "box"}
    elif node_type in [NodeType.MAP_KEYS, NodeType.MAP_VALUES, NodeType.MAP_ITEMS]:
        return {"style": "filled", "shape": "box3d"}
    else:
        return {"style": "rounded,dashed", "shape": "box"}


class Orientation(Enum):
    VERTICAL = "TB"
    HORIZONTAL = "LR"


@overload
def to_pygraphviz(graph: Task) -> pg.AGraph:
    ...


@overload
def to_pygraphviz(graph: nx.DiGraph) -> pg.AGraph:
    ...


def to_pygraphviz(graph: Union[Task, nx.DiGraph], orientation: Orientation = Orientation.VERTICAL) -> pg.AGraph:
    if isinstance(graph, Task):
        graph = graph._graph

    agraph = nx.nx_agraph.to_agraph(graph)

    # set graph attributes
    agraph.graph_attr.update(_GRAPH_ATTRIBUTES(orientation))

    # set node attributes
    for v in graph.nodes:
        node = agraph.get_node(v)
        node_type = graph.nodes[v].get(TYPE_ATTRIBUTE, None)
        node.attr.update(_NODE_ATTRIBUTES(node_type))

    # set edge attributes
    for u, v in graph.edges:
        is_map = graph.edges[u, v][MAP_ATTRIBUTE]
        edge = agraph.get_edge(u, v)
        edge.attr.update(_EDGE_ATTRIBUTES(is_map))

    agraph.layout(_GRAPH_LAYOUT)
    return agraph
