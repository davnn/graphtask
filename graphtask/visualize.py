# type: ignore
"""
Visualization of a `Task` using `pygraphviz`.
"""
from typing import Union, overload

import networkx as nx

from graphtask import Task

try:
    import pygraphviz as pg
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "Graph visualisation requires 'pygraphviz' to be installed, but no installation found "
        + "See http://pygraphviz.github.io/ for guidance."
    ) from err

__all__ = ["to_pygraphviz"]

_GRAPH_LAYOUT = "dot"
_GRAPH_ATTRIBUTES = {"rankdir": "TB", "bgcolor": "white"}
_NODE_ATTRIBUTES = {
    "color": "#f0f0f0",
    "style": "filled",
    "fontcolor": "#111111",
    "shape": "box",
    "fontsize": 10,
}
_EDGE_ATTRIBUTES = {"color": "black", "arrowsize": 2 / 3}


@overload
def to_pygraphviz(graph: Task) -> pg.AGraph:
    ...


@overload
def to_pygraphviz(graph: nx.DiGraph) -> pg.AGraph:
    ...


def to_pygraphviz(graph: Union[Task, nx.DiGraph]) -> pg.AGraph:
    if isinstance(graph, Task):
        graph = graph._graph

    agraph = nx.nx_agraph.to_agraph(graph)

    # set graph attributes
    agraph.graph_attr.update(_GRAPH_ATTRIBUTES)

    # set node attributes
    for v in graph.nodes:
        node = agraph.get_node(v)
        node.attr.update(_NODE_ATTRIBUTES)

    # set edge attributes
    for u, v in graph.edges:
        edge = agraph.get_edge(u, v)
        edge.attr.update(_EDGE_ATTRIBUTES)

    agraph.layout(_GRAPH_LAYOUT)
    return agraph
