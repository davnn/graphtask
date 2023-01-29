"""
Define the ``Step`` data type, which wraps and enhances the function given to ``Task.step``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast, get_args

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from inspect import Signature
from logging import getLogger
from warnings import warn

if TYPE_CHECKING:
    from graphtask._task import Task

from joblib import Parallel, delayed

from graphtask._check import is_iterable, is_mapping, is_mutable_mapping
from graphtask._globals import MapArgsT, MapTypeT

logger = getLogger(__name__)


class StepFnT(Protocol):
    """A function of keyword-only arguments (the dependencies in the graph) to the original function return value"""

    def __call__(self, **kw: Any) -> Any:  # pragma: no cover
        ...


class MapFnT(Protocol):
    """A map function converts an argument name (key), key and value to a (key, value) result."""

    def __call__(self, name: str, key: Hashable, value: Any, **kw: Any) -> tuple[Hashable, Any]:  # pragma: no cover
        ...


@dataclass
class StepParams:
    """Single source of truth for parameters of @step decorators defined in ``Task``"""

    map: str | None = None
    map_type: MapTypeT = "values"
    rename: str | None = None
    args: str | Iterable[str] | None = None
    kwargs: str | Iterable[str] | None = None
    alias: Mapping[str, str] | None = None


@dataclass
class StepArgs:
    """Classification of __call__ arguments, such that the original function signature can be reconstructed"""

    positional: list[str] = field(default_factory=list)
    keyword: list[str] = field(default_factory=list)
    positional_only: list[str] = field(default_factory=list)


class StepKind(Enum):
    """The internal type of node, which is stored in `TYPE_ATTRIBUTE` of a node."""

    FUNCTION = "function"
    MAP_KEYS = "map_keys"
    MAP_VALUES = "map_values"
    MAP_ITEMS = "map_items"


class InvalidStepArgumentException(Exception):
    ...


class Step:
    def __init__(
        self,
        name: str,
        fn: StepFnT,
        signature: Signature,
        task: Task,
        args: StepArgs = StepArgs(),
        params: StepParams = StepParams(),
        n_jobs: int = 1,
    ) -> None:
        super().__init__()
        self.name = name
        self.kind = determine_step_kind(params.map, params.map_type)
        self.signature = signature
        self.function = fn
        self.args = args
        self._task = task
        self._parameters = params
        self._parallel = lambda: Parallel(n_jobs=n_jobs, prefer="threads")

    def run(self, **dependencies: Any) -> Any:
        """
        Run the Step by specifying all direct dependencies in the graph as keyword arguments. The direct dependencies
        are the original arguments, but might be aliased.

        Parameters
        ----------
        dependencies: Any
            All necessary arguments to invoke the function transformed into keyword arguments and possible aliased.

        Returns
        -------
        Any
            The result of calling the original function with the given dependencies as arguments.
        """
        direct_dependencies = list(self._task._graph.predecessors(self.name))  # type: ignore[reportPrivateUsage]
        assert set(dependencies) == set(
            direct_dependencies
        ), f"Resolved direct dependencies to be '{direct_dependencies}', but the passed dependencies were '{dependencies}'."

        kind, arg = self.kind, self._parameters.map
        if kind == StepKind.FUNCTION:
            logger.debug(f"Determined function kind '{kind}'.")
            data = self.function(**dependencies)
        else:
            assert arg is not None, f"No 'map' argument specified for Step with type '{kind}'."
            assert arg in dependencies, f"'map' argument '{arg}' not found in the dependencies: '{dependencies}'."
            logger.debug(f"Determined kind {kind}, mappping over argument {arg}")
            data = dependencies[arg]
            iterable, mappable, mappable_and_mutable = is_iterable(data), is_mapping(data), is_mutable_mapping(data)
            assert iterable, f"Data for 'map' argument '{arg}' must be iterable, but found '{data}'"
            if kind == StepKind.MAP_KEYS or kind == StepKind.MAP_ITEMS:
                assert mappable, f"Data for 'map' argument '{arg}' must be mappable, but found '{data}'"

            mapping = map_data(data, mappable, iterable)  # convert the raw mapping data into a suitable format
            logger.debug(f"Converted data '{data}' to mapping '{mapping}'.")
            del dependencies[arg]  # remove the raw mapping data from the dependencies
            mappable_fn = map_fn(self.function, kind)
            result = self._parallel()(delayed(mappable_fn)(arg, mk, mv, **dependencies) for mk, mv in mapping)

            if mappable_and_mutable:
                # if the mappable is also mutable, we mutate the original mapping, which might be a custom mappable
                for k, v in result:
                    data[k] = v
            elif mappable:
                # if the mappable is not mutable, we return a dictionary by default
                warn(f"Mapping over non-mutable mapping type '{type(data)}', returning dictionary result.")
                data = dict(result)
            else:
                # if the input is not mappable, we return a generator of the results
                data = (value for _, value in result)

        return data

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pos, kw, pos_only = (set(x) for x in [self.args.positional, self.args.keyword, self.args.positional_only])
        logger.debug(f"Determined '{pos}' as pos, '{kw}' as kw-only and '{pos_only}' as pos-only arguments.")

        # note: a keyword argument can never appear before a positional argument, thus, checking if there are more
        # positional arguments than keyword arguments ensures that no keyword arguments are invoked positionally
        if len(args) > len(pos):
            raise InvalidStepArgumentException(
                f"Step takes {len(pos)} positional arguments, but {len(args)} positional arguments were given."
            )

        if len(nonexistant_kw := set(kwargs).difference(pos.union(kw))) > 0:
            raise InvalidStepArgumentException(f"Step got unexpected keyword arguments: '{nonexistant_kw}'")

        if len(pos_only_kw := pos_only.intersection(kwargs)) > 0:
            raise InvalidStepArgumentException(
                f"Step got positional-only arguments passed as keyword arguments: '{pos_only_kw}'."
            )

        def maybe_alias(name: str) -> str:
            """Return the name if not aliased, otherwise the aliased name."""
            if self._parameters.alias is not None and name in self._parameters.alias:
                return self._parameters.alias[name]
            else:
                return name

        # prepare args and kwargs as replacements
        args_aliased = dict(zip(map(maybe_alias, pos), args))
        kwargs_aliased = {maybe_alias(k): v for k, v in kwargs.items()}

        if len(pos_and_kw := set(args_aliased).intersection(kwargs_aliased)) > 0:
            raise InvalidStepArgumentException(
                f"Step got duplicate arguments '{pos_and_kw}' as positional and keyword arguments."
            )

        dependencies = {k: v for k, v in self._task.get(drop_last=True, **args_aliased, **kwargs_aliased).items()}
        logger.debug(f"Determined dependencies as '{dependencies}'.")

        return self.run(**dependencies)

    def __repr__(self):
        return f"Step{self.signature}\n - Identifier: '{self.name}'\n - Parameters: {self._parameters}"


def determine_step_kind(map_arg: str | None, map_type: MapTypeT) -> StepKind:
    """Based on ``map`` and ``map_type`` determine the type of the node."""
    if map_arg is None:
        return StepKind.FUNCTION
    elif map_type == "keys":
        return StepKind.MAP_KEYS
    elif map_type == "values":
        return StepKind.MAP_VALUES
    elif map_type == "items":
        return StepKind.MAP_ITEMS
    else:
        raise AssertionError(f"The parameter 'map_type' must be one of '{get_args(MapTypeT)}' but found '{map_type}'")


def map_fn(fn: StepFnT, kind: StepKind) -> MapFnT:
    """
    Turn a function of variable keyword argument dependencies into a mapped function additionally accepting a
    mapping key ``key``, mapping value ``value`` and the original keyword parameter ``name``.

    Parameters
    ----------
    fn: StepFnT
        Function of only keyword arguments, each specifying one of the Step's dependencies.
    kind: StepKind
        Kind of step to determine the correct mapping function (keys, values, items).

    Returns
    -------
    MapFnT
        Enhanced ``fn`` that takes an additional key and value to map over.
    """

    def check_tuple(result: tuple[Hashable, Any]) -> tuple[Hashable, Any]:
        """Ensure that the return value of a function mapping over items is correct."""
        # this check does not add a significant overhead to the function and stays in the low hundreds of
        # nanoseconds even for very large inputs
        assert (
            isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], Hashable)
        ), f"Must return a 'key, value' tuple from function mapping over `items`, but found '{result}'."
        return result

    map_fn: dict[StepKind, MapFnT] = {
        StepKind.MAP_VALUES: lambda name, key, value, **kw: (key, fn(**{name: value}, **kw)),
        StepKind.MAP_KEYS: lambda name, key, value, **kw: (fn(**{name: key}, **kw), value),
        StepKind.MAP_ITEMS: lambda name, key, value, **kw: check_tuple(fn(**{name: (key, value)}, **kw)),
    }
    return map_fn[kind]


def map_data(data: Mapping[Hashable, Any] | Iterable[Any], mappable: bool, iterable: bool) -> MapArgsT:
    """
    Ensure that the arg is mappable and convert to mapping of map keys to argument dictionaries.

    The mapping data is transformed from ``{"key": {"map_key1": 1, "map_key2": 2, ...}}`` to
    ``[(key, map_key1, 1), (key, map_key_2, 2), ...]``. If the map edge is not a mappable, but an iterable input,
    it is transformed from ``{"key": [1, 2, ...]}`` to a mapping of indices ``[(key, 0, 1), (key, 1, 2), ...]``.

    Parameters
    ----------
    data: Mapping | Iterable
        Data to be transformed into suitable format for mapping function.
    mappable: bool
        If the given ``data`` is mappable.
    iterable: bool
        If the given ``data`` is iterable.

    Returns
    -------
    MapArgsT
        A list of (key, value) tuples containing the keys and values to be mapped over.

    Raises
    ------
    AssertionError
        If the given ``data`` is not mappable or iterable.
    """
    if mappable:
        data = cast(Mapping[Hashable, Any], data)
        result = [(map_key, map_value) for map_key, map_value in data.items()]
    elif iterable:
        data = cast(Iterable[Any], data)
        result = [(cast(Hashable, map_key), map_value) for map_key, map_value in enumerate(data)]
    else:  # this should not happen, because ``is_iterable`` is already checked before ``map_data`` is called
        raise AssertionError(f"Parameter 'map' requires an iterable input argument, but found {data}")
    return result
