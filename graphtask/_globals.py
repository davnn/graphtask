"""
Those globals are used throughout the codebase
"""
from collections.abc import Callable
from typing import Literal, Any, TypeVar, Hashable

STEP_ATTRIBUTE = "__step__"  # attribute for nodes in the graph to store the underlying ``Step``
ArgsT = dict[str, Any]
DecorableT = TypeVar("DecorableT", bound=Callable[..., Any])  # type of ``@step``-decorable values
MapArgsT = list[tuple[str, Hashable, Any]]
MapTypeT = Literal["keys", "values", "items"]
