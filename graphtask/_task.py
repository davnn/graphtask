"""
Definition of a `Task` and `step`.
"""
from typing import Any, Optional, TypeVar, Union, cast, overload

import inspect
import logging
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping
from enum import Enum
from sys import maxsize

import networkx as nx
from joblib import Parallel, delayed

from graphtask._check import is_dag, is_iterable, is_mapping, verify
from vendor.stackeddag import edgesToText, mkEdges, mkLabels

logger = logging.getLogger(__name__)

__all__ = ["Task", "step"]

DecorableT = TypeVar("DecorableT", bound=Callable[..., Any])
ArgsT = dict[str, Any]
SplitArgsT = list[ArgsT]
MapArgsT = dict[Hashable, ArgsT]

FUNC_ATTRIBUTE = "__func__"
DATA_ATTRIBUTE = "__data__"
STEP_ATTRIBUTE = "__step__"
TYPE_ATTRIBUTE = "__type__"


class NodeType(Enum):
    """The kind of node stored in `TYPE_ATTRIBUTE` of a node."""

    ATTRIBUTE = 1
    FUNCTION = 2


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

    Parameters
    ----------
    fn: optional callable
        The method to be decorated, which gets added to the underlying ``Task``.
    split: optional str
        Iterable dependency to split and invoke ``fn`` on each iterated value. An iterable dependency must return an
        ``Iterable``, for example a list of ``[value1, value2]``. If ``split`` is specified for the dependency, the
        function is invoked as ``[fn(value1), fn(value2)]`` always returning a list.
    map: optional str
        Mappable dependency to invoke ``fn`` on each iterated value of the map. A mappable dependency must return a
        ``Mapping``, for example a dictionary of ``{key1: value1, key2: value2}``. If ``map`` is specified for the
        dependency, the function would only operate on the values and the return value of the node would look like
        ``{key1: fn(value1), key2: fn(value2)}``.
    rename: optional str
        Rename the node in the graph created with ``fn``, the default node name is ``fn.__name__``. For lambda
        functions, the ``fn.__name__`` is always ``<lambda>``, for example. Because node names have to be unique
        in the graph, renaming is useful in certain scenarios.
    args: optional iterable of str
        Iterable of identifiers for variable positional arguments. This argument is required if ``*args`` is specified
        for the given ``fn``. If ``args`` is ``["a", "b", "c"]``, it would inject the dependencies ``"a"``, ``"b"``
        and ``"c"`` as positional arguments.
    kwargs: optional iterable of str
        Iterable of identifiers for variable keyword arguments. This argument is required if `*args` is specified
        for the given ``fn``. If ``kwargs`` is ``["a", "b", "c"]``, it would inject the dependencies ``"a"``, ``"b"``
        and ``"c"`` as keyword arguments.
    alias: optional mapping of str to str
        Rename arguments according to an ``{original: renamed}`` mapping. Aliasing happens after ``*args`` and
        ``*kwargs`` have been injected, because ``*args`` and ``**kwargs`` are unique and the name of both arguments
        is not processed further.

    Returns
    -------
    decorator or function
        A decorator if no ``fn`` is given, otherwise ``fn``.
    """

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
    """A metaclass to enable classes inheriting from `Task` to decorate methods using `@step`

    Decorating a method using ``@step`` sets a ``STEP_ATTRIBUTE`` on the decorated method containing ``kwargs`` to
    ``@step`` for the method. On ``__init__`` of a class inheriting from ``Task``, the metaclass iterates over all
    methods containing the ``STEP_ATTRIBUTE`` and adds them as steps of the ``Task``. For an explanation, see:
    https://stackoverflow.com/questions/16017397/injecting-function-call-after-init-with-decorator
    """

    def __call__(cls, *args: Any, **kwargs: Any):
        """Called when you call Task()"""
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
    """A Task consists of steps that are implicitly modeled as a directed, acyclic graph (DAG)."""

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
        """A decorator (or decorator factory) to add steps to the graph (documented at :meth:`graphtask.step`)"""

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

            def fn_processed(**passed: Any) -> Any:
                """A closure function, that re-arranges the passed keyword arguments into positional-only, variable
                positional and keyword arguments such that the signature of the original `fn` is respected.

                Parameters
                ----------
                passed: Any
                    Keyword arguments from predecessor nodes.

                Returns
                -------
                Any
                    Return value of the original (unprocessed) function.
                """
                invert_alias_step_parameters(passed, alias)
                positional = process_positional_args(passed, original_posargs)
                return fn(*positional, **passed)

            # add the processed function to the graph
            logger.debug(f"Adding node '{fn_name}' to graph")
            self._graph.add_node(fn_name, **{FUNC_ATTRIBUTE: fn_processed, TYPE_ATTRIBUTE: NodeType.FUNCTION})

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

    def register(self, **kwargs: Any) -> None:
        """Register all keyword arguments with `key` and `value` as a node with identifier `key` on the graph.

        Parameters
        ----------
        kwargs: keys to any values
            Each provided key identifies a node on the graph. For example ``task.register(a=1)`` would register the
            node ``"a"`` on the graph with a value of ``1``. The value is registered lazily, such that there is no
            difference between nodes registered using ``register`` or ``step``.
        """
        for key, value in kwargs.items():
            logger.debug(f"Registering node {key}")
            lazy_value: Callable[[Any], Any] = lambda v=value: v
            self._graph.add_node(key, **{FUNC_ATTRIBUTE: lazy_value, TYPE_ATTRIBUTE: NodeType.ATTRIBUTE})

    def run(self, node: Optional[str] = None) -> Any:
        """Run the full task if no `node` is given, otherwise run up until `node`.

        Parameters
        ----------
        node: optional str
            Optional identifier of the ``node`` to run. This runs all dependencies of ``node`` and returns the
            value returned by ``node``. If node is a function, this is equal to calling the function, though without
            having to resolve all dependencies manually.

        Returns
        -------
        Any
            The value of the last node if there is a single last node in the graph, otherwise a tuple of values of all
            last nodes. If the graph is empty, an empty tuple is returned. If ``node`` is specified, the value of
            the specified node is returned instead of the last node.

        Raises
        ------
        AssertionError
            If the specified ``node`` is not found in the graph.
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

        Parameters
        ----------
        node: Hashable
            Identifier of the node to materialize.

        Returns
        -------
        Any
            The materialized value of the node.
        """
        current_node = self._graph.nodes[node]
        logger.debug(f"Current node:        {repr(node)}")
        kwargs, kwarg_split, kwarg_map = predecessor_edges(self._graph, node)
        logger.debug(f"Determined kwargs: {kwargs}")
        logger.debug(f"Determined split kwarg: {kwarg_split}")
        logger.debug(f"Determined map kwarg: {kwarg_map}\n")

        assert FUNC_ATTRIBUTE in current_node, f"Node '{node}' not defined, but set as a dependency."
        fn = current_node[FUNC_ATTRIBUTE]

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

    def __str__(self) -> str:
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

    Parameters
    ----------
    graph: DiGraph
        A directed acyclic graph (DAG).
    node: Hashable
        Identifier of the node to predecessors.

    Returns
    -------
    dict[str, Any]
        Keyword arguments directly from edges that have not been processed.
    list[dict[str, Any]]
        Optional `split` argument.
    dict[Hashable, dict[str, Any]]
        Optional `map` arguments.
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

    Parameters
    ----------
    passed: dict[str, Any]
        Original arguments passed as keyword arguments.
    pos_args: list[str]
        Ordered identifiers of the positional only arguments.

    Returns
    -------
    list[str]
        Ordered values of positional-only arguments.
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

    Parameters
    ----------
    params: list of Parameter
        Initial inspected parameters.
    args: optional str iterable
        Given `args` from `step`.
    kwargs: optional str iterable
        Given `kwargs` from `step`.

    Returns
    -------
    list[str]
        List of positional arguments.
    list[str]
        List of keyword arguments.
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
