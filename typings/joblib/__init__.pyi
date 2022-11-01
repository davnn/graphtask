from typing import Any, Literal, TypeVar

from collections.abc import Callable, Iterable
from logging import Logger

_FunctionT = TypeVar("_FunctionT", bound=Callable[..., Any])

class Parallel(Logger):
    def __init__(self, n_jobs: int, backend: Literal["threading", "loky", "multiprocessing"]) -> None: ...
    def __call__(self, iterable: Iterable[Any]) -> list[Any]: ...

def delayed(function: _FunctionT) -> _FunctionT: ...
