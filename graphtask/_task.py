"""
Definition of a `Task`.
"""
from typing import Any, Optional, TypeVar, Union, cast, overload

import inspect
import logging
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping
from sys import maxsize

import networkx as nx
from joblib import Parallel, delayed
from stackeddag.core import edgesToText, mkEdges, mkLabels  # type: ignore[reportUnknownVariableType]

from graphtask._check import is_dag, is_iterable, is_mapping, verify

logger = logging.getLogger(__name__)

__all__ = ["Task", "step"]

DecorableT = TypeVar("DecorableT", bound=Callable[..., Any])
ArgsT = dict[str, Any]
SplitArgsT = list[ArgsT]
MapArgsT = dict[Hashable, ArgsT]

FN_ATTRIBUTE = "__function__"
DATA_ATTRIBUTE = "__data__"
STEP_ATTRIBUTE = "__step__"


@overload
def step(
    fn: DecorableT,
    *,
    split: Optional[str] = None,
    map: Optional[str] = None,
    rename: Optional[str] = None,
    args: Optional[Union[str, Iterable[str]]] = None,
    kwargs: Optional[Union[str, Iterable[str]]] = None,
    alias: Optional[Mapping[str, str]] = None,
) -> DecorableT:
    """Step invoked with a `fn`, returns the `fn`"""
    ...


@overload
def step(
    *,
    split: Optional[str] = None,
    map: Optional[str] = None,
    rename: Optional[str] = None,
    args: Optional[Union[str, Iterable[str]]] = None,
    kwargs: Optional[Union[str, Iterable[str]]] = None,
    alias: Optional[Mapping[str, str]] = None,
) -> Callable[[DecorableT], DecorableT]:
    """Step invoked without a `fn`, return a decorator"""
    ...


def step(
    fn: Optional[DecorableT] = None,
    *,
    split: Optional[str] = None,
    map: Optional[str] = None,
    rename: Optional[str] = None,
    args: Optional[Union[str, Iterable[str]]] = None,
    kwargs: Optional[Union[str, Iterable[str]]] = None,
    alias: Optional[Mapping[str, str]] = None,
) -> Union[DecorableT, Callable[[DecorableT], DecorableT]]:
    """A method decorator (or decorator factory) to add steps to the graph of a class inheriting from `Task`.

    Args:
        fn: The function to be decorated.
        split: Iterable argument to invoke `fn` on each iterated value.
        map: Mappable argument to invoke `fn` on each iterated value.
        rename: Rename the node created with `fn`, default is `fn.__name__`.
        args: Identifiers for variable positional arguments for `fn`.
        kwargs: Identifiers for variable keyword arguments for `fn`.
        alias: Rename arguments according to {"argument_name": "renamed_value"}.

    Returns:
        A decorator if no `fn` is given, otherwise `fn`."""

    def decorator(fn: DecorableT) -> DecorableT:
        setattr(fn, STEP_ATTRIBUTE, dict(split=split, map=map, rename=rename, args=args, kwargs=kwargs, alias=alias))
        return fn

    if callable(fn):
        # use `step` directly as a decorator (return the decorated fn)
        return decorator(fn)
    else:
        # use `step` as a decorator factory (return a decorator)
        return decorator


class TaskMeta(type):
    """A metaclass to enable classes inheriting from `Task` to use the `@step` decorator, which sets a `STEP_ATTRIBUTE`
    attribute on each decorated method containing `kwargs` to `@step` for the decorated method.

    On an `__init__` call of a class inheriting from `Task`, the metaclass iterates over all methods containing the
    `STEP_ATTRIBUTE` and adds them as 'real' steps of the `Task`. For an explanation of why this works, see:
    https://stackoverflow.com/questions/16017397/injecting-function-call-after-init-with-decorator
    """

    def __call__(cls, *args: Any, **kwargs: Any):
        """Called when you call MyNewClass()"""
        obj = type.__call__(cls, *args, **kwargs)

        # iterate over all the attribute names of the newly created object
        for attr_name in dir(obj):
            attr = getattr(obj, attr_name)

            # only add steps for attributes that are callable and contain the `STEP_ATTRIBUTE`
            if callable(attr) and hasattr(attr, STEP_ATTRIBUTE):
                kwargs = getattr(attr, STEP_ATTRIBUTE)
                obj.step(attr, **kwargs)

        return obj


class Task(metaclass=TaskMeta):
    """A Task consists of `step`s that are implicitly modeled as a directed, acyclic graph (DAG)."""

    def __init__(self, n_jobs: int = 1) -> None:
        super().__init__()
        # public attributes
        self.n_jobs = n_jobs

        # private attributes
        self._graph = nx.DiGraph()
        self._parallel = lambda: Parallel(n_jobs=n_jobs, backend="threading")

    @overload
    def step(
        self,
        fn: DecorableT,
        *,
        split: Optional[str] = None,
        map: Optional[str] = None,
        rename: Optional[str] = None,
        args: Optional[Union[str, Iterable[str]]] = None,
        kwargs: Optional[Union[str, Iterable[str]]] = None,
        alias: Optional[Mapping[str, str]] = None,
    ) -> DecorableT:
        """Step invoked with a `fn`, returns the `fn`"""
        ...

    @overload
    def step(
        self,
        *,
        split: Optional[str] = None,
        map: Optional[str] = None,
        rename: Optional[str] = None,
        args: Optional[Union[str, Iterable[str]]] = None,
        kwargs: Optional[Union[str, Iterable[str]]] = None,
        alias: Optional[Mapping[str, str]] = None,
    ) -> Callable[[DecorableT], DecorableT]:
        """Step invoked without a `fn`, return a decorator"""
        ...

    def step(
        self,
        fn: Optional[DecorableT] = None,
        *,
        split: Optional[str] = None,
        map: Optional[str] = None,
        rename: Optional[str] = None,
        args: Optional[Union[str, Iterable[str]]] = None,
        kwargs: Optional[Union[str, Iterable[str]]] = None,
        alias: Optional[Mapping[str, str]] = None,
    ) -> Union[DecorableT, Callable[[DecorableT], DecorableT]]:
        """A function decorator (or decorator factory) to add steps to the graph.

        Args:
            fn: The function to be decorated.
            split: Iterable argument to invoke `fn` on each iterated value.
            map: Mappable argument to invoke `fn` on each iterated value.
            rename: Rename the node created with `fn`, default is `fn.__name__`.
            args: Identifiers for variable positional arguments for `fn`.
            kwargs: Identifiers for variable keyword arguments for `fn`.
            alias: Rename arguments according to {"argument_name": "renamed_value"}.

        Returns:
            A decorator if no `fn` is given, otherwise `fn`.
        """

        def decorator(fn: DecorableT) -> DecorableT:
            params = get_function_params(fn)
            original_posargs, original_kwargs = split_step_params(params, args, kwargs)
            # combine all parameters to a single `set` of parameters (the dependencies in the graph)
            # we only care about the parameter names from now on (as a set of string)
            params = set(original_posargs).union(set(original_kwargs))
            alias_step_parameters(params, alias)
            verify_step_parameters(params, split=split, map=map)
            logger.debug(f"Extracted function parameters: '{params}'.")

            # rename the node if `rename` is given
            fn_name = fn.__name__ if rename is None else rename

            # 1. A node can be `defined`, meaning that the name of the node is already in the graph, but there is
            # no corresponding function available for the node. This is the case if a node is referred to before
            # the function is declared.
            # 2. A node can be `initialized`, meaning that the node in the graph has an existing `FN_ATTRIBUTE`.
            if fn_name in self._graph.nodes:
                assert FN_ATTRIBUTE not in self._graph.nodes[fn_name], (
                    f"Cannot add already existing node '{fn_name}' to graph, use 'rename' or provide a named function "
                    f"different from the existing nodes."
                )

            def fn_processed(**passed: Any) -> Any:
                """A closure function, that re-arranges the passed keyword arguments into positional-only, variable
                positional and keyword arguments such that the signature of the original `fn` is respected.

                Args:
                    **passed: Keyword arguments from predecessor nodes.

                Returns:
                    Return value of the original (unprocessed) function.
                """
                invert_alias_step_parameters(passed, alias)
                positional = process_positional_args(passed, original_posargs)
                return fn(*positional, **passed)

            # add the processed function to the graph
            logger.debug(f"Adding node '{fn_name}' to graph")
            self._graph.add_node(fn_name, **{FN_ATTRIBUTE: fn_processed})

            # make sure the fn's parameters are nodes in the graph
            for param in params:
                logger.debug(f"Adding dependency '{param}' to graph")
                self._graph.add_edge(param, fn_name, split=split == param, map=map == param)

            # make sure that the resulting graph is a DAG
            verify(is_dag, self._graph)
            return fn

        if callable(fn):
            # use `step` directly as a decorator (return the decorated fn)
            return decorator(fn)
        else:
            # use `step` as a decorator factory (return a decorator)
            return decorator

    def register(self, **kwargs: Any):
        """Register all keyword arguments with `key` and `value` as a node with identifier `key` on the graph.

        Args:
            **kwargs: Key/Values, where each key identifies a node on the graph.
        """
        for key, value in kwargs.items():
            logger.debug(f"Registering node {key}")
            lazy_value: Callable[[Any], Any] = lambda v=value: v
            self._graph.add_node(key, **{FN_ATTRIBUTE: lazy_value})

    def run(self, node: Optional[str] = None) -> Any:
        """Run the full task if no `node` is given, otherwise run up until `node`.

        Args:
            node: Optional identifier of the `node` to run.

        Returns:
            Empty tuple if graph is empty, value of the last node if there is a single last node otherwise
            a tuple of values of all last nodes.

        Raises:
            AssertionError: If one of the assertions does not hold.
        """
        verify(is_dag, self._graph)  # this assertion should always hold, except the user messes with `_graph`
        assert node is None or node in self._graph.nodes, f"The 'node' must be in Task, but did not find '{node}'."
        gens = topological_generations(self._graph) if node is None else topological_predecessors(self._graph, node)

        result = ()  # only relevant if no generations
        for generation in gens:  # we are only interested in the last result here (that's the one to return)
            result = self._parallel()(delayed(self._materialize)(node) for node in generation)

        # return the result of the last generation
        return result[0] if len(result) == 1 else tuple(result)

    def _materialize(self, node: Hashable) -> Any:
        """Materialize the result of calling a node as an edge weight stored in `DATA_ATTRIBUTE`.

        Args:
            node: Identifier of the node to materialize.

        Returns:
            The materialized value of the node.
        """
        current_node = self._graph.nodes[node]
        logger.debug(f"Current node:        {repr(node)}")
        kwargs, kwarg_split, kwarg_map = predecessor_edges(self._graph, node)
        logger.debug(f"Determined kwargs: {kwargs}")
        logger.debug(f"Determined split kwarg: {kwarg_split}")
        logger.debug(f"Determined map kwarg: {kwarg_map}\n")

        assert FN_ATTRIBUTE in current_node, f"Node '{node}' not defined, but set as a dependency."
        fn = current_node[FN_ATTRIBUTE]

        if kwarg_split is not None:
            result = self._parallel()(delayed(fn)(**kwargs, **arg) for arg in kwarg_split)
        elif kwarg_map is not None:
            map_fn: Callable[..., tuple[Hashable, Any]] = lambda key, **kw: (key, fn(**kw))
            result = dict(self._parallel()(delayed(map_fn)(key, **kwargs, **kw) for key, kw in kwarg_map.items()))
        else:
            result = fn(**kwargs)

        # add the materialized result to the node
        current_node[DATA_ATTRIBUTE] = result

        # return the materialized result
        return result

    def __str__(self):
        return f"Task(n_jobs={self.n_jobs})"

    def __repr__(self) -> str:
        graph = self._graph
        header = f"{self.__str__()}"

        nodes: list[str] = list(graph.nodes)
        edges: list[tuple[str, str]] = list(graph.edges)

        # empty graph
        if len(nodes) == 0:
            return header

        # graph with no edges
        if len(edges) == 0:
            return header + f"\no    {','.join(nodes)}"

        nodes_reshaped = [(node, node) for node in nodes]
        edges_reshaped = [(src, [dst]) for src, dst in edges]
        text: str = edgesToText(mkLabels(nodes_reshaped), mkEdges(edges_reshaped))
        return header + f"\n{text.strip()}"


def predecessor_edges(graph: nx.DiGraph, node: Hashable) -> tuple[ArgsT, Optional[SplitArgsT], Optional[MapArgsT]]:
    """Prepare the input data for `node` based on normal, `split` and `map` edges.
    The split edge (there can only be one) is transformed from {"key": [1, 2, ...]} to [{"key": 1}, {"key": 2}, ...].
    The map edge: {"key": {"map_key_1": 1, "map_key_2": 2}} -> {"map_key_1": {"key": 1}, "map_key_2": {"key": 2}, ...}.

    Args:
        graph: Directed acyclic graph.
        node: Identifier of the node to predecessors.

    Returns:
        kwargs: Keyword arguments directly from edges that have not been processed.
        split: Optional `split` argument.
        map: Optional `map` arguments.
    """
    direct_predecessors = list(graph.predecessors(node))
    logger.debug(f"Direct predecessors: {direct_predecessors}")

    kwargs: ArgsT = {}
    kwarg_split = None
    kwarg_map = None

    edges: dict[Hashable, dict[Hashable, Any]] = {dep: graph.edges[dep, node] for dep in direct_predecessors}
    logger.debug(f"Predecessor edges: {edges}")

    for key, edge in edges.items():
        key = cast(str, key)  # we only use str keys
        data = graph.nodes[key][DATA_ATTRIBUTE]
        if edge["split"]:
            kwarg_split = split_arg(key, data)
        elif edge["map"]:
            kwarg_map = map_arg(key, data)
        else:
            kwargs[key] = data

    return kwargs, kwarg_split, kwarg_map


def map_arg(key: str, data: Mapping[Hashable, Any]) -> MapArgsT:
    """Ensure that the arg is mappable and convert to mapping of map keys to argument dictionaries."""
    verify(is_mapping, data)
    result = {map_key: {key: map_value} for map_key, map_value in data.items()}
    return result


def split_arg(key: str, data: Iterable[Any]) -> SplitArgsT:
    """Ensure that the arg is splittable (iterable) and split into a list of args containing the values in the list."""
    verify(is_iterable, data)
    result = [{key: value} for value in data]
    return result


def process_positional_args(passed: dict[str, Any], pos_args: list[str]) -> list[str]:
    """Process `passed` args to extract positional arguments and remove them from `passed` (the remaining kw-args).

    Args:
        passed: Original arguments passed as keyword arguments.
        pos_args: Ordered identifiers of the positional only arguments.

    Returns:
        pos_values: Ordered values of positional-only arguments.
    """
    pos_values: list[str] = []
    for arg in pos_args:
        pos_values.append(passed[arg])
        del passed[arg]

    return pos_values


def split_step_params(
    params: list[inspect.Parameter],
    args: Optional[Union[str, Iterable[str]]],
    kwargs: Optional[Union[str, Iterable[str]]],
) -> tuple[list[str], list[str]]:
    """Split positional and keyword arguments from `params`.

    Note that an argument has to be supplied using a keyword if it follows a var-positional argument (following *args),
    if it is declared as a keyword only argument (following *) or if it declared as a variable keyword argument.

    Args:
        params: Initial inspected parameters.
        args: Given `args` from `step`.
        kwargs: Given `kwargs` from `step`.

    Returns:
        positional: Positional arguments (can be `POSITIONAL_ONLY`, `POSITIONAL_OR_KEYWORD`, or `VAR_POSITIONAL`).
        keyword: Keyword arguments
    """
    param_names: list[Union[str, list[str]]] = []
    param_kinds: list[inspect._ParameterKind] = []  # type:ignore[reportPrivateUsage]
    has_var_arg = False
    has_var_kwarg = False
    for param in params:
        name = param.name
        kind = param.kind
        param_kinds.append(kind)

        # replace the *args and **kwargs param with a list of replacements
        if kind == inspect.Parameter.VAR_POSITIONAL:
            assert args is not None, f"Variable argument '*{name}' requires 'args' parameter to be set in 'step'."
            args = [args] if isinstance(args, str) else list(args)
            param_names.append(args)
            has_var_arg = True
        elif kind == inspect.Parameter.VAR_KEYWORD:
            assert kwargs is not None, f"Variable argument '**{name}' requires 'kwargs' parameter to be set in 'step'."
            kwargs = [kwargs] if isinstance(kwargs, str) else list(kwargs)
            param_names.append(kwargs)
            has_var_kwarg = True
        else:
            param_names.append(name)

    # validate the input to `args` and `kwargs`
    if has_var_arg and args is not None:
        duplicates = [arg for arg in args if arg in param_names]
        assert not any(duplicates), (
            f"The names provided to 'args' cannot be duplicates of the "
            f"function parameters, but found duplicates: '{duplicates}'."
        )

    if has_var_kwarg and kwargs is not None:
        duplicates = [arg for arg in kwargs if arg in param_names]
        assert not any(duplicates), (
            f"The names provided to 'kwargs' cannot be duplicates of the "
            f"function parameters, but found duplicates: '{duplicates}'."
        )

    if args is not None and kwargs is not None:
        duplicates = [arg for arg in args if arg in kwargs]
        assert not any(
            duplicates
        ), f"There cannot be duplicate names provided to 'args' and 'kwargs', but found duplicates: '{duplicates}."

    if args is not None and not has_var_arg:
        warnings.warn("Provided 'args' argument for 'step', but no '*args' parameter found.")

    if kwargs is not None and not has_var_kwarg:
        warnings.warn("Provided 'kwargs' argument for 'step', but no '**kwargs' parameter found.")

    # split into positional and keyword arguments according to idx and flatten the nested `args` and `kwargs`
    kw_idx = first_keyword_idx(param_kinds)
    pos_names = flatten_names(param_names[:kw_idx])
    kw_names = flatten_names(param_names[kw_idx:])
    return pos_names, kw_names


def flatten_names(ls: list[Union[str, list[str]]]) -> list[str]:
    """Flatten a list of possible nested strings, such that ['ab', ['c', 'd']] becomes ['ab', 'c', 'd']"""
    result: list[str] = []
    for item in ls:
        if isinstance(item, list):
            for sub in item:
                result.append(sub)
        else:
            result.append(item)
    return result


def first_keyword_idx(param_kinds: list[inspect._ParameterKind]) -> int:  # type: ignore[reportPrivateUsage]
    """Return the first idx of an argument that can only be specified as a keyword or otherwise a very large integer"""
    var_pos = inspect.Parameter.VAR_POSITIONAL
    kw_only = inspect.Parameter.KEYWORD_ONLY
    var_kw = inspect.Parameter.VAR_KEYWORD

    # if we do not find a var-pos or kw-only argument, we return a very large int (sys.maxsize)
    # treating every argument as positional
    var_pos_idx = kw_only_idx = var_kw_idx = maxsize

    if var_pos in param_kinds:
        # + 1 because we have to include the var pos argument in the positional arguments
        var_pos_idx = param_kinds.index(var_pos) + 1

    if kw_only in param_kinds:
        kw_only_idx = param_kinds.index(kw_only)

    if var_kw in param_kinds:
        var_kw_idx = param_kinds.index(var_kw)

    return min(var_pos_idx, kw_only_idx, var_kw_idx)


def alias_step_parameters(params: set[str], alias: Optional[Mapping[str, str]]) -> None:
    """Rename function parameters to use a given alias."""
    if alias is not None:
        params_to_replace = (key for key in alias.keys() if key in params)
        for param in params_to_replace:
            params.remove(param)
            params.add(alias[param])


def invert_alias_step_parameters(params: dict[str, Any], alias: Optional[Mapping[str, str]]) -> None:
    """Undo renaming of function parameters, i.e. for the passed `params` change the aliased keys to original keys."""
    if alias is not None:
        inverse_alias = {v: k for k, v in alias.items()}
        keys_to_rename = (key for key in inverse_alias.keys() if key in params)
        for key in keys_to_rename:
            params[inverse_alias[key]] = params.pop(key)


def verify_step_parameters(params: set[str], split: Optional[str], map: Optional[str]) -> None:
    """Ensure that given `split` and `map` are in the (final) parameter set."""
    if split is not None and map is not None:
        raise AssertionError("Cannot combine 'split' and 'map' in a single step.")

    if split is not None:
        assert split in params, f"Step argument 'split' must refer to one of the parameters, but found '{split}'."

    if map is not None:
        assert map in params, f"Step argument 'map' must refer to one of the parameters, but found '{map}'."


def get_function_params(fn: Callable[..., Any]) -> list[inspect.Parameter]:
    """From a given `fn`, return a list of inspected function parameters."""
    return list(inspect.signature(fn).parameters.values())


def bfs_successors(graph: nx.DiGraph, node: Hashable) -> list[Hashable]:  # pragma: no cover (currently not used)
    """The names of all successors to `node`"""
    result: list[Hashable] = list(nx.bfs_tree(graph, node))[1:]
    return result


def bfs_predecessors(graph: nx.DiGraph, node: Hashable) -> list[Hashable]:  # pragma: no cover (currently not used)
    """The names of all predecessors to `node`"""
    result: list[Hashable] = list(nx.bfs_tree(graph.reverse(copy=False), node))[1:]
    return result


def topological_successors(graph: nx.DiGraph, node: Hashable) -> list[list[Hashable]]:
    """The names of all invalidated nodes (grouped in generations) if `node` changed"""
    bfs_tree = nx.bfs_tree(graph, node)
    subgraph = nx.induced_subgraph(graph, bfs_tree.nodes)
    generations = nx.topological_generations(subgraph)
    return list(generations)


def topological_predecessors(graph: nx.DiGraph, node: Hashable) -> list[list[Hashable]]:
    """The names of all dependency nodes (grouped in generations) for `node`"""
    generations = topological_successors(graph.reverse(copy=False), node)
    return list(reversed(generations))


def topological_generations(graph: nx.DiGraph) -> list[list[Hashable]]:
    """The names of all nodes in the graph grouped into generations"""
    return list(nx.topological_generations(graph))
