"""
Those globals are used throughout the codebase
"""
from typing import Any, Literal

from collections.abc import Callable, Hashable

STEP_ATTRIBUTE = "__step__"  # attribute for nodes in the graph to store the underlying ``Step``
DecorableT = Callable[..., Any]  # type of ``@step``-decorable values
ArgsT = dict[str, Any]
MapArgsT = list[tuple[Hashable, Any]]
MapTypeT = Literal["keys", "values", "items"]
