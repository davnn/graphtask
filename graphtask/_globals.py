"""Those globals are used throughout the codebase."""
from collections.abc import Callable, Hashable
from typing import Any, Literal

STEP_ATTRIBUTE = "__step__"  # attribute for nodes in the graph to store the underlying ``Step``
DecorableT = Callable[..., Any]  # type of ``@step``-decorable values
ArgsT = dict[str, Any]
MapArgsT = list[tuple[Hashable, Any]]
MapTypeT = Literal["keys", "values", "items"]
BackendT = Literal["threading", "multiprocessing", "loky"]
