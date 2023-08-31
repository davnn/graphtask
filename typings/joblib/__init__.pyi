from typing import Any, Literal, TypeVar

from collections.abc import Callable, Iterable
from logging import Logger

_FunctionT = TypeVar("_FunctionT", bound=Callable[..., Any])

class Parallel(Logger):
    def __init__(
        self,
        n_jobs: int,
        backend: Literal[None, "threading", "loky", "multiprocessing"] = None,
        prefer: Literal[None, "processes", "threads"] = None,
    ) -> None: ...
    def __call__(self, iterable: Iterable[Any]) -> list[Any]: ...

def delayed(function: _FunctionT) -> _FunctionT: ...
