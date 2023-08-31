from typing import Any

class AGraph:
    graph_attr: dict[str, Any]
    def get_node(self, v: str) -> Node: ...
    def get_edge(self, u: str, v: str) -> Edge: ...

class Node:
    attr: dict[str, Any]

class Edge:
    attr: dict[str, Any]
