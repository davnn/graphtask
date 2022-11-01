"""
Definition of a `Task`.
"""
from typing import Any, Optional, TypeVar, Union, cast, overload

import inspect
import logging
from collections.abc import Callable, Hashable, Iterable, Mapping

import networkx as nx
from joblib import Parallel, delayed
from networkx import DiGraph
from stackeddag.core import edgesToText, mkEdges, mkLabels  # type: ignore[reportUnknownVariableType]

from graphtask._check import is_dag, is_iterable, is_mapping, verify

logger = logging.getLogger(__name__)

__all__ = ["Task"]

DecorableT = TypeVar("DecorableT", bound=Callable[..., Any])
ArgsT = dict[str, Any]
SplitArgsT = list[ArgsT]
MapArgsT = dict[Hashable, ArgsT]


class Task:
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
        args: Optional[Iterable[str]] = None,
        kwargs: Optional[Iterable[str]] = None,
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
        args: Optional[Iterable[str]] = None,
        kwargs: Optional[Iterable[str]] = None,
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
        args: Optional[Iterable[str]] = None,
        kwargs: Optional[Iterable[str]] = None,
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
            """The decorator function is returned if no `fn` is given."""
            params = get_function_params(fn)
            original_kwargs, original_posargs, _, _ = extract_step_parameters(params, args, kwargs)
            # we only care about the parameter names from now on (as a set of string)
            params = combine_step_parameters(original_kwargs, original_posargs, args, kwargs)
            alias_step_parameters(params, alias)
            verify_step_parameters(params, split=split, map=map)
            logger.debug(f"Extracted function parameters: '{params}'.")

            # rename the node if `rename` is given
            fn_name = fn.__name__ if rename is None else rename
            assert fn_name != "<lambda>", "Cannot name node '<lambda>', use 'rename' or provide a named function."

            def fn_processed(**passed: Any) -> Any:
                """A closure function, that re-arranges the passed keyword arguments into positional-only, variable
                positional and keyword arguments such that the signature of the original `fn` is respected.

                Args:
                    **passed: Keyword arguments from predecessor nodes.

                Returns:
                    Return value of the original (unprocessed) function.
                """
                pos, var = process_passed_args(passed, original_posargs, args)
                return fn(*pos, *var, **passed)

            # add the processed function to the graph
            logger.debug(f"Adding node '{fn_name}' to graph")
            self._graph.add_node(fn_name, __function__=fn_processed, __shape__="box")

            # make sure the fn's parameters are nodes in the graph
            for param in params:
                logger.debug(f"Adding dependency '{param}' to graph")
                assert param in self._graph.nodes, f"Cannot find '{param}' in the graph, but set as a dependency."
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
            self._graph.add_node(key, __function__=lazy_value, __shape__="point")

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
        """Materialize the result of calling a node as an edge weight stored in `__data__`.

        Args:
            node: Identifier of the node to materialize.

        Returns:
            The materialized value of the node.
        """
        current_node = self._graph.nodes[node]
        logger.debug(f"Current node:        {repr(node)}")

        # short circuit if there is already valid data
        # if "__data__" in current_node:
        #     return current_node["__data__"]
        kwargs, kwarg_split, kwarg_map = predecessor_edges(self._graph, node)
        logger.debug(f"Determined kwargs: {kwargs}")
        logger.debug(f"Determined split kwarg: {kwarg_split}")
        logger.debug(f"Determined map kwarg: {kwarg_map}\n")

        fn = current_node["__function__"]
        if kwarg_split is not None:
            result = self._parallel()(delayed(fn)(**kwargs, **arg) for arg in kwarg_split)
        elif kwarg_map is not None:
            map_fn: Callable[..., tuple[Hashable, Any]] = lambda key, **kw: (key, fn(**kw))
            result = dict(self._parallel()(delayed(map_fn)(key, **kwargs, **kw) for key, kw in kwarg_map.items()))
        else:
            result = fn(**kwargs)

        # add the materialized result to the node
        current_node["__data__"] = result

        # return the materialized result
        return result

    def __str__(self):
        return f"Task(n_jobs={self.n_jobs})"

    def __repr__(self) -> str:
        graph = self._graph
        header = f"{self.__str__()}\n"

        nodes: list[str] = list(graph.nodes)
        edges: list[tuple[str, str]] = list(graph.edges)

        # empty graph
        if len(nodes) == 0:
            return header

        # graph with no edges
        if len(edges) == 0:
            return header + f"o    {','.join(nodes)}"

        nodes_reshaped = [(node, node) for node in nodes]
        edges_reshaped = [(src, [dst]) for src, dst in edges]
        text: str = edgesToText(mkLabels(nodes_reshaped), mkEdges(edges_reshaped))
        return header + text


def predecessor_edges(graph: DiGraph, node: Hashable) -> tuple[ArgsT, Optional[SplitArgsT], Optional[MapArgsT]]:
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
        data = graph.nodes[key]["__data__"]
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


def process_passed_args(
    passed: dict[str, Any], pos_args: list[str], var_args: Optional[Iterable[str]]
) -> tuple[list[str], list[str]]:
    """Process `passed` args to extract positional only and variable positional arguments and remove them from `passed`.

    Args:
        passed: Original arguments passed as keyword arguments.
        pos_args: Ordered identifiers of the positional only arguments.
        var_args: Ordered identifiers of the variable positional arguments.

    Returns:
        pos_values: Ordered values of positional-only arguments.
        var_values: Ordered values of variable positional arguments.
    """
    pos_values: list[str] = []
    var_values: list[str] = []
    for arg in pos_args:
        pos_values.append(passed[arg])
        del passed[arg]

    if var_args is not None:
        for arg in var_args:
            var_values.append(passed[arg])
            del passed[arg]

    return pos_values, var_values


def extract_step_parameters(
    params: list[inspect.Parameter], args: Optional[Iterable[str]], kwargs: Optional[Iterable[str]]
) -> tuple[set[str], list[str], Optional[str], Optional[str]]:
    """Extract parameters by kind (kw, pos-only, var-pos, var-kw) from initially inspected `params`.

    Args:
        params: Initial inspected parameters.
        args: Given `args` from `step`.
        kwargs: Given `kwargs` from `step`.

    Returns:
        kw: Keyword arguments (includes positional or keyword arguments and keyword only)
        pos: Positional-only arguments.
        var_pos: Optional variable positional argument.
        var_kw: Optional variable position argument.
    """
    kwargs_names: set[str] = set()
    pos_only_names: list[str] = []
    var_arg_name: Optional[str] = None  # there can only be one
    var_kwarg_name: Optional[str] = None  # there can only be one
    for p in params:
        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            pos_only_names.append(p.name)
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            var_arg_name = p.name
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            var_kwarg_name = p.name
        else:  # POSITIONAL_OR_KEYWORD or KEYWORD_ONLY
            kwargs_names.add(p.name)

    assert_var_arg = f"Variable arguments `*args` require step `args` parameter and vice versa."
    assert_var_kwarg = f"Variable keyword arguments `**kwargs` require step `kwargs` parameter and vice versa."
    assert not (args is not None) ^ (var_arg_name is not None), assert_var_arg
    assert not (kwargs is not None) ^ (var_kwarg_name is not None), assert_var_kwarg

    return kwargs_names, pos_only_names, var_arg_name, var_kwarg_name


def combine_step_parameters(
    original_kwargs: set[str],
    original_posargs: list[str],
    step_args: Optional[Iterable[str]],
    step_kwargs: Optional[Iterable[str]],
):
    """Combine all possible parameters to a single `set` of parameters (the dependencies in the graph)"""
    return (
        original_kwargs.union(set(original_posargs))
        .union(set(step_args) if step_args is not None else set())
        .union(set(step_kwargs) if step_kwargs is not None else set())
    )


def alias_step_parameters(params: set[str], alias: Optional[Mapping[str, str]]) -> None:
    """Rename function parameters to use a given alias."""
    if alias is not None:
        params_to_replace = (key for key in alias.keys() if key in params)
        for param in params_to_replace:
            params.remove(param)
            params.add(alias[param])


def verify_step_parameters(params: set[str], split: Optional[str], map: Optional[str]) -> None:
    """Ensure that given `split` and `map` are in the (final) parameter set."""
    if split is not None and map is not None:
        raise AssertionError("Cannot combine `split` and `map` in a single step.")

    if split is not None:
        assert split in params, f"Argument `split` must refer to one of the parameters, but found {split}."

    if map is not None:
        assert map in params, f"Argument `map` must refer to one of the parameters, but found {map}."


def get_function_params(fn: Callable[..., Any]) -> list[inspect.Parameter]:
    """From a given `fn`, return a list of inspected function parameters."""
    return list(inspect.signature(fn).parameters.values())


def bfs_successors(graph: DiGraph, node: Hashable) -> list[Hashable]:
    """The names of all successors to `node`"""
    result: list[Hashable] = list(nx.bfs_tree(graph, node))[1:]
    return result


def bfs_predecessors(graph: DiGraph, node: Hashable) -> list[Hashable]:
    """The names of all predecessors to `node`"""
    result: list[Hashable] = list(nx.bfs_tree(graph.reverse(copy=False), node))[1:]
    return result


def topological_successors(graph: DiGraph, node: Hashable) -> list[list[Hashable]]:
    """The names of all invalidated nodes (grouped in generations) if `node` changed"""
    bfs_tree = nx.bfs_tree(graph, node)
    subgraph = nx.induced_subgraph(graph, bfs_tree.nodes)
    generations = nx.topological_generations(subgraph)
    return list(generations)


def topological_predecessors(graph: DiGraph, node: Hashable) -> list[list[Hashable]]:
    """The names of all dependency nodes (grouped in generations) for `node`"""
    generations = topological_successors(graph.reverse(copy=False), node)
    return list(reversed(generations))


def topological_generations(graph: DiGraph) -> list[list[Hashable]]:
    """The names of all nodes in the graph grouped into generations"""
    return list(nx.topological_generations(graph))
