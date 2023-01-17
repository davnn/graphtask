from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable, Mapping, Hashable
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, Any, get_args, cast, Protocol

import networkx as nx

from graphtask._check import is_mapping, is_iterable
from graphtask._globals import DecorableT, STEP_ATTRIBUTE, MapTypeT, ArgsT, MapArgsT
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


class MapFnT(Protocol):
    """A map function converts an argument name (key), map_key and map_value to a (key, value) result."""

    def __call__(
        self, key: str, map_key: Hashable, map_value: Any, **kw: Any
    ) -> tuple[Hashable, Any]:  # pragma: no cover
        ...


@dataclass
class StepParams:
    """Single source of truth for parameters of @step decorators defined in ``Task``"""
    map_arg: Optional[str] = None
    map_type: MapTypeT = "values"
    rename: Optional[str] = None
    args: Optional[Union[str, Iterable[str]]] = None
    kwargs: Optional[Union[str, Iterable[str]]] = None
    alias: Optional[Mapping[str, str]] = None


@dataclass
class StepArgs:
    """Classification of __call__ arguments, such that the original function signature can be reconstructed"""
    positional: list[str] = field(default_factory=list)
    keyword: list[str] = field(default_factory=list)
    positional_only: list[str] = field(default_factory=list)


class StepKind(Enum):
    """The internal type of node, which is stored in `TYPE_ATTRIBUTE` of a node."""

    ATTRIBUTE = "attribute"
    FUNCTION = "function"
    MAP_KEYS = "map_keys"
    MAP_VALUES = "map_values"
    MAP_ITEMS = "map_items"


class InvalidStepArgumentException(Exception):
    pass


class Step:
    def __init__(
        self,
        name: str,
        fn: DecorableT,
        signature: inspect.Signature,
        task: Any,
        args: StepArgs = StepArgs(),
        dependencies: nx.DiGraph = nx.DiGraph(),
        params: StepParams = StepParams(),
        cache: Any = None,
        n_jobs: int = 1
    ) -> None:
        self.name = name
        self.kind = determine_step_kind(params.map_arg, params.map_type)
        self.signature = signature
        self.function = fn
        self.args = args
        self._dependencies = dependencies
        self._parameters = params
        self._cache = cache
        self._parallel = lambda: Parallel(n_jobs=n_jobs, prefer="threads")

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        self._cache = value

    @cache.deleter
    def cache(self):
        self._cache = None

    def _mapped_call(self, mappable, map_arg, kwargs):
        map_fn: Optional[MapFnT] = None
        if self.kind == StepKind.MAP_VALUES:
            # it is already asserted that map_arg is iterable in ``prepare_data_from_predecessors``
            map_fn = lambda key, map_key, map_value, **kw: (map_key, self.function(**{key: map_value}, **kw))
        elif self.kind == StepKind.MAP_KEYS:
            assert mappable, "Cannot use 'map_type=keys' with no mappable argument."
            map_fn = lambda key, map_key, map_value, **kw: (self.function(**{key: map_key}, **kw), map_value)
        elif self.kind == StepKind.MAP_ITEMS:
            assert mappable, "Cannot use 'map_type=items' with no mappable argument."
            map_fn = lambda key, map_key, map_value, **kw: self.function(**{key: (map_key, map_value)}, **kw)

        if map_fn is not None:
            assert map_arg is not None, f"No mappable argument found for node specified as type '{self.kind}'."
            result = self._parallel()(delayed(map_fn)(key, mk, mv, **kwargs) for key, mk, mv in map_arg)
            result = dict(result) if mappable else [value for _, value in result]
        else:
            result = self.function(**kwargs)

        return result

    def materialize(self) -> Any:
        kwargs, map_arg, mappable = prepare_data_from_predecessors(self._dependencies.nodes, self._parameters.map_arg)
        result = self._mapped_call(mappable, map_arg, kwargs)
        self.cache = result
        return result

    def __call__(self, *args, **kwargs) -> Any:
        kwargs, map_arg, mappable = prepare_data_from_predecessors(self._dependencies.nodes, self._parameters.map_arg)

        dependency_caches = {k: v[STEP_ATTRIBUTE].cache for k, v in self._dependencies.nodes.items()}
        pos, kw, positional_only = map(set, [self.args.positional, self.args.keyword, self.args.positional_only])

        # note: a keyword argument can never appear before a positional argument, thus, checking if there are more
        # positional arguments than keyword arguments ensures that no keyword arguments are invoked positionally
        if len(args) > len(pos):
            raise InvalidStepArgumentException(
                f"Step takes {len(pos)} positional arguments, but {len(args)} positional arguments were given."
            )

        if len(nonexistant_kw := set(kwargs).difference(set(pos).union(kw))) > 0:
            raise InvalidStepArgumentException(
                f"Step got unexpected keyword argument(s): {nonexistant_kw}"
            )

        if len(pos_only_kw := set(positional_only).intersection(kwargs)) > 0:
            raise InvalidStepArgumentException(
                f"Step got positional-only argument(s) passed as keyword argument(s): {pos_only_kw}."
            )

        maybe_alias = lambda name: self._parameters.alias[
            name] if self._parameters.alias is not None and name in self._parameters.alias else name

        # override the positional arguments
        set_posargs = set()
        for name, value in zip(pos, args):
            dependency_caches[maybe_alias(name)] = value
            set_posargs.add(name)

        if len(pos_and_kw := set_posargs.intersection(kwargs)) > 0:
            raise InvalidStepArgumentException(
                f"Step got duplicate arguments(s) {pos_and_kw} as positional and keyword arguments."
            )

        # override the keyword arguments
        for name, value in kwargs.items():
            dependency_caches[maybe_alias(name)] = value

        # return f(existing_kwargs["a"], existing_kwargs["b"], c=existing_kwargs["c"], d=existing_kwargs["d"])
        return self._mapped_call(**dependency_caches)


def determine_step_kind(map_arg: Optional[str], map_type: MapTypeT) -> StepKind:
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


def prepare_data_from_predecessors(
    dependencies: nx.NodeView,
    map_arg: Optional[str]
) -> tuple[ArgsT, Optional[MapArgsT], bool]:
    """Prepare the input data for `node` based on normal edges and an optional `map` edge.

    The map edge (there can only be one) is transformed from ``{"key": {"map_key1": 1, "map_key2": 2, ...}}`` to
    ``[(key, map_key1, 1), (key, map_key_2, 2), ...]``. If the map edge is not a mappable, but an iterable input,
    it is transformed from ``{"key": [1, 2, ...]}`` to a mapping of indices ``[(key, 0, 1), (key, 1, 2), ...]``.

    Parameters
    ----------
    dependencies: NodeView
        A view to all predecessor nodes.
    name: Hashable
        Identifier of the node to predecessors.

    Returns
    -------
    dict[str, Any]
        Keyword arguments directly from edges that have not been processed.
    list[tuple[Hashable, Hashable, Any]]
        Optional processed arguments for ``map`` edge.
    """
    mappable = False
    kwargs: ArgsT = {}
    map_args: Optional[MapArgsT] = None
    logger.debug(f"Predecessor nodes: {dependencies}")

    for node_name, node in dependencies.items():
        assert STEP_ATTRIBUTE in node, f"Node '{node_name}' not defined, but set as a dependency."
        step = node[STEP_ATTRIBUTE]
        data = step.cache if step.cache is not None else step.materialize()
        if node_name == map_arg:
            iterable, mappable = is_iterable(data), is_mapping(data)
            map_args = preprocess_map_arg(node_name, iterable, mappable, data)
        else:
            kwargs[node_name] = data

    logger.debug(f"Determined kwargs:    {kwargs}")
    logger.debug(f"Determined map kwarg: {map_arg}\n")
    return kwargs, map_args, mappable


def preprocess_map_arg(
    key: str,
    iterable: bool,
    mappable: bool,
    data: Union[dict[Hashable, Any], Iterable[Any]]
) -> MapArgsT:
    """Ensure that the arg is mappable and convert to mapping of map keys to argument dictionaries."""
    if mappable:
        data = cast(Mapping[Hashable, Any], data)
        result = [(key, map_key, map_value) for map_key, map_value in data.items()]
    elif iterable:
        data = cast(Iterable[Any], data)
        result = [(key, cast(Hashable, map_key), map_value) for map_key, map_value in enumerate(data)]
    else:
        raise AssertionError(f"Parameter 'map' requires an iterable input argument, but found {data}")
    return result
