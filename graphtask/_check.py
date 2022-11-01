"""
Generic checks and assertions.
"""
from typing import Any, Callable

from collections.abc import Mapping

import networkx as nx


def is_iterable(iterable: Any) -> bool:
    """
    Check if `iterable` is iterable, by invoking `inter`. This is more reliable than `isinstance(iterable, Iterable)`,
    see: https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable

    Args:
        iterable: A value that may be iterable.

    Returns:
        `True` if `iterable` is iterable, otherwise `False`.
    """
    try:
        iter(iterable)
        return True
    except Exception:
        return False


def is_mapping(mapping: Any) -> bool:
    """
    Check if `mapping` is a mapping type based on `Mapping`

    Args:
        mapping: A value that may be a mapping.

    Returns:
        `True` if `iterable` is an instance of `Mapping`, otherwise `False`.
    """
    return isinstance(mapping, Mapping)


def is_dag(graph: nx.DiGraph) -> bool:
    """Determine if `graph` is directed and acyclic."""
    return nx.is_directed_acyclic_graph(graph)


def verify(predicate: Callable[..., bool], *args: Any, **kwargs: Any):
    """Assert that a `predicate` function (bool return) holds given `*args` and `**kwargs`.

    Args:
        predicate: Function returning `True` or `False`
        *args: Positional arguments to `predicate`.
        **kwargs: Keyword arguments to `predicate`.

    Raises:
        AssertionError: If the `predicate` function does not hold (returns False).
    """
    proposition = predicate(*args, **kwargs)
    assert proposition, f"Cannot verify that predicate '{predicate.__name__}' holds for arguments: {args} {kwargs}."
