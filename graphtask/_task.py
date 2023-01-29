"""
Definition of a `Task` and `step`.
"""
from typing import Any, Optional, Union, cast, overload

import inspect
from collections.abc import Callable, Iterable, Mapping
from copy import copy
from dataclasses import asdict
from logging import getLogger
from sys import maxsize
from warnings import warn

import networkx as nx
from joblib import Parallel, delayed
from stackeddag.core import edgesToText, mkEdges, mkLabels  # type: ignore[reportUnknownVariableType]

from graphtask._check import is_dag, verify
from graphtask._globals import STEP_ATTRIBUTE, DecorableT, MapTypeT
from graphtask._step import Step, StepArgs, StepFnT, StepParams

logger = getLogger(__name__)

__all__ = ["Task", "step"]


@overload
def step(
    fn: DecorableT,
    *,
    map: Optional[str] = None,
    map_type: MapTypeT = "values",
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
    map: Optional[str] = None,
    map_type: MapTypeT = "values",
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
    map: Optional[str] = None,
    map_type: MapTypeT = "values",
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
    map: optional str
        Name of the iterable argument to invoke ``fn`` on each key, value or item of the argument value. A non-mappable
        iterable argument value, for example a list of values, is treated as a mapping of indices to values, i.e. the
        list ``["a", "b", "c"]`` would be treated as ``{0: "a", 1: "b", 2: "c"}``. If ``map`` is specified, the node
        automatically returns a mapping (dictionary) of keys to values.
    map_type: MapTypeT
        The type of mapping to perform using ``map``. Can be ``"keys"`` resulting in
        ``{fn(key1): value1, fn(key2): value2, ...}``, ``"values"`` resulting in
        ``{key1: fn(value1), key2: fn(value2), ...}`` or ``"items"`` resulting in
        ``{fn((key1, value1), fn(key2, value2), ...}``.
    rename: optional str
        Rename the node in the graph created with ``fn``, the default node name is ``fn.__name__``. For lambda
        functions, the ``fn.__name__`` is always ``<lambda>``, for example. Because node names have to be unique
        in the graph, renaming is useful in certain scenarios.
    args: optional iterable of str
        Iterable of identifiers for variable positional arguments. This argument is required if ``*args`` is specified
        for the given ``fn``. If ``args`` is ``["a", "b", "c"]``, it would inject the dependencies ``"a"``, ``"b"``
        and ``"c"`` as positional arguments.
    kwargs: optional iterable of str
        Iterable of identifiers for variable keyword arguments. This argument is required if ``**kwargs`` is specified
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
        setattr(
            fn,
            STEP_ATTRIBUTE,
            StepParams(map=map, map_type=map_type, rename=rename, args=args, kwargs=kwargs, alias=alias),
        )
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
                step_params = getattr(attr, STEP_ATTRIBUTE)
                setattr(obj, attr_name, obj.step(attr, **asdict(step_params)))

        return obj


class Task(metaclass=TaskMeta):
    """A Task consists of steps that are implicitly modeled as a directed, acyclic graph (DAG)."""

    def __init__(self, n_jobs: int = 1) -> None:
        super().__init__()
        # public attributes
        self.n_jobs = n_jobs

        # private attributes
        self._graph = nx.DiGraph()
        self._parallel = lambda: Parallel(n_jobs=n_jobs, prefer="threads")

    @overload
    def step(
        self,
        fn: DecorableT,
        *,
        map: Optional[str] = None,
        map_type: MapTypeT = "values",
        rename: Optional[str] = None,
        args: Optional[Union[str, Iterable[str]]] = None,
        kwargs: Optional[Union[str, Iterable[str]]] = None,
        alias: Optional[Mapping[str, str]] = None,
    ) -> Step:
        """Step invoked with a `fn`, returns the `fn`"""
        ...

    @overload
    def step(
        self,
        *,
        map: Optional[str] = None,
        map_type: MapTypeT = "values",
        rename: Optional[str] = None,
        args: Optional[Union[str, Iterable[str]]] = None,
        kwargs: Optional[Union[str, Iterable[str]]] = None,
        alias: Optional[Mapping[str, str]] = None,
    ) -> Callable[[DecorableT], Step]:
        """Step invoked without a `fn`, return a decorator"""
        ...

    def step(
        self,
        fn: Optional[DecorableT] = None,
        *,
        map: Optional[str] = None,
        map_type: MapTypeT = "values",
        rename: Optional[str] = None,
        args: Optional[Union[str, Iterable[str]]] = None,
        kwargs: Optional[Union[str, Iterable[str]]] = None,
        alias: Optional[Mapping[str, str]] = None,
    ) -> Union[Step, Callable[[DecorableT], Step]]:
        """A decorator (or decorator factory) to add steps to the graph (documented at :meth:`graphtask.step`)"""

        def decorator(fn: DecorableT) -> Step:
            params = get_function_params(fn)
            step_args = determine_step_arguments(params, args, kwargs)
            # combine all parameters to a single `set` of parameters (the dependencies in the graph)
            # we only care about the parameter names from now on (as a set of string)
            params = set(step_args.positional).union(set(step_args.keyword))
            alias_step_parameters(params, alias)
            verify_map_parameter(params, map=map)
            logger.debug(f"Extracted function parameters: '{params}'.")

            # rename the node if `rename` is given
            step_name = fn.__name__ if rename is None else rename

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
                logger.debug(f"Passed arguments '{passed}' to processed function.")
                invert_alias_step_parameters(passed, alias)
                positional = process_positional_args(passed, step_args.positional)
                return fn(*positional, **passed)

            # add the processed function to the graph
            logger.debug(f"Adding node '{step_name}' to graph")

            self._graph.add_node(step_name)
            predecessors = set(self._graph.predecessors(step_name))
            assert all(p in params for p in predecessors), (
                f"Cannot update node '{step_name}' with missing dependencies '{predecessors.difference(params)}', "
                f"you should rebuild the entire task if you intend to replace steps with remove dependencies."
            )

            # make sure the fn's parameters are nodes in the graph
            for param in params:
                logger.debug(f"Adding dependency '{param}' to graph")
                self._graph.add_edge(param, step_name)

            step = Step(
                name=step_name,
                fn=fn_processed,
                args=step_args,
                signature=inspect.signature(fn),
                task=self[step_name],
                params=StepParams(map=map, map_type=map_type, rename=rename, args=args, kwargs=kwargs, alias=alias),
            )

            # add the step to the node
            self._graph.nodes[step_name][STEP_ATTRIBUTE] = step

            # make sure that the resulting graph is a DAG
            verify(is_dag, self._graph)
            return step

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
            assert (
                key not in self._graph.nodes or len(p := list(self._graph.predecessors(key))) == 0
            ), f"Cannot register existing node with predecessors '{p}' without predecessors."
            logger.debug(f"Registering node {key}")
            fn: StepFnT = (lambda v=value: lambda: v)()  # type: ignore[reportUnknownLambdaType]
            self._graph.add_node(
                key, **{STEP_ATTRIBUTE: Step(name=key, fn=fn, signature=inspect.signature(fn), task=self)}
            )

    def get(self, drop_last: bool = False, include_all: bool = False, **replacements: Any) -> dict[str, Any]:
        """Run the full task if no `node` is given, otherwise run up until `node`.

        Parameters
        ----------
        drop_last: bool
            Remove the last topological generation from further processing.
        include_all: bool
            Return all graph elements as a dictionary of node names to retrieved values in the graph.
        **replacements: Any
            Replace specific ``nodes`` in the graph without invoking the replaced nodes.

        Returns
        -------
        dict[str, Any]
            A dictionary of node names to their resolved or replaced values.

        Raises
        ------
        AssertionError
            If the specified ``node`` is not found in the graph.
        """
        assert all(r in self._graph.nodes for r in replacements), (
            f"Replacement **kwargs must refer to existing nodes, but found replacements"
            f" '{replacements}' for nodes '{self._graph.nodes}'."
        )
        verify(is_dag, self._graph)  # this assertion should always hold, except the user messes with `_graph`

        # identify the topological generations, which can be parallelized
        gens = topological_generations(self._graph)[:-1] if drop_last else topological_generations(self._graph)

        dependencies: dict[str, Any] = {}
        last_dependencies: dict[str, Any] = {}
        for generation in gens:
            result = self._parallel()(delayed(self._run_step)(node, replacements, dependencies) for node in generation)
            last_dependencies = dict(zip(generation, result))
            dependencies = dependencies | last_dependencies

        return dependencies if include_all else last_dependencies

    def show(self):
        from graphtask._visualize import to_pygraphviz

        return to_pygraphviz(self)

    def _run_step(self, node: str, replacements: dict[str, Any], dependencies: dict[str, Any]) -> Any:
        """Safely invoke a step of the graph with existence checks.

        Parameters
        ----------
        node: str
            Identifier of the node to run.
        replacements: dict[str, Any]
            Collection of replacements to use as dependencies instead of resolving them using the graph.
        dependencies: dict[str, Any]
            Collection of all necessary dependencies to invoke the function.

        Returns
        -------
        Any
            The output of running a ``Step``.
        """
        if node in replacements:
            logger.debug(f"Replacement used for node '{node}'.")
            return replacements[node]

        current_node = self._graph.nodes[node]
        logger.debug(f"Current node:         {repr(node)}")
        assert STEP_ATTRIBUTE in current_node, f"Node '{node}' not defined, but set as a dependency."
        step = current_node[STEP_ATTRIBUTE]
        logger.debug(f"Current step:         {repr(step)}")
        direct_predecessors = list(self._graph.predecessors(node))
        direct_dependencies = {k: v for k, v in dependencies.items() if k in direct_predecessors}
        assert len(direct_predecessors) == len(
            direct_dependencies
        ), f"Not all dependencies defined for direct predecessors '{direct_predecessors}' of node '{node}'"
        return step.run(**direct_dependencies)

    def __getitem__(self, item: str):
        assert item in self._graph.nodes, f"Attempted to retrieve node '{item}', but it was not found in the graph."
        view = copy(self)
        view._graph = nx.induced_subgraph(self._graph, [item] + bfs_predecessors(view._graph, item))
        return view

    def __call__(self, **replacements: Any) -> Any:
        """
        Run the task and return the topologically last value (if there is one last node), or a tuple of last values.

        Parameters
        ----------
        **replacements: Any
            Replace specific ``nodes`` in the graph without invoking the replaced nodes.

        Returns
        -------
        Any
            The value of the last node if there is a single last node in the graph, otherwise a tuple of values of all
            last nodes. If the graph is empty, an empty tuple is returned. If ``node`` is specified, the value of
            the specified node is returned instead of the last node.
        """
        result = tuple(self.get(**replacements).values())
        return result[0] if len(result) == 1 else result

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


def determine_step_arguments(
    params: list[inspect.Parameter],
    args: Optional[Union[str, Iterable[str]]],
    kwargs: Optional[Union[str, Iterable[str]]],
) -> StepArgs:
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
    StepArgs:
        Classification of positional, keyword and positional-only arguments.
    """
    param_names: list[Union[str, list[str]]] = []
    param_kinds: list[inspect._ParameterKind] = []  # type:ignore[reportPrivateUsage]
    pos_only_names: list[str] = []
    has_var_arg = False
    has_var_kwarg = False
    for param in params:
        name = param.name
        kind = param.kind
        param_kinds.append(kind)

        # replace the *args and **kwargs param with a list of replacements
        # we do not append the variable arguments directly to a list[str], because that would make it more difficult to
        # correctly check for duplicates in the arguments; we want to know where the duplicates stem from.
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
        elif kind == inspect.Parameter.POSITIONAL_ONLY:
            param_names.append(name)
            pos_only_names.append(name)
        else:
            param_names.append(name)

    if has_var_arg:
        duplicates = [arg for arg in cast(list[str], args) if arg in param_names]
        assert not any(duplicates), (
            f"The names provided to 'args' cannot be duplicates of the "
            f"function parameters, but found duplicates: '{duplicates}'."
        )

    if has_var_kwarg:
        duplicates = [arg for arg in cast(list[str], kwargs) if arg in param_names]
        assert not any(duplicates), (
            f"The names provided to 'kwargs' cannot be duplicates of the "
            f"function parameters, but found duplicates: '{duplicates}'."
        )

    if has_var_arg and has_var_kwarg:
        duplicates = [arg for arg in cast(list[str], args) if arg in cast(list[str], kwargs)]
        assert not any(
            duplicates
        ), f"There cannot be duplicate names provided to 'args' and 'kwargs', but found duplicates: '{duplicates}."

    if args is not None and not has_var_arg:
        warn("Provided 'args' argument for 'step', but no '*args' parameter found.")

    if kwargs is not None and not has_var_kwarg:
        warn("Provided 'kwargs' argument for 'step', but no '**kwargs' parameter found.")

    # split into positional and keyword arguments according to idx and flatten the nested `args` and `kwargs`
    kw_idx = first_keyword_idx(param_kinds)
    pos_names = flatten_names(param_names[:kw_idx])
    kw_names = flatten_names(param_names[kw_idx:])
    pos_only = pos_only_names + cast(list[str], args) if args is not None else pos_only_names
    return StepArgs(positional=pos_names, keyword=kw_names, positional_only=pos_only)


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
        for original_name, replacement_name in alias.items():
            if original_name in params:
                params.remove(original_name)
                params.add(replacement_name)
            else:
                warn(f"Found alias '{original_name}', but '{original_name}' is not in the arguments.")


def invert_alias_step_parameters(params: dict[str, Any], alias: Optional[Mapping[str, str]]) -> None:
    """Undo renaming of function parameters, i.e. for the passed `params` change the aliased keys to original keys."""
    inverse_alias = {v: k for k, v in alias.items()} if alias is not None else {}
    keys_to_rename = (key for key in inverse_alias.keys() if key in params)
    for key in keys_to_rename:
        params[inverse_alias[key]] = params.pop(key)


def verify_map_parameter(params: set[str], map: Optional[str]) -> None:
    """Ensure that given `split` and `map` are in the (final) parameter set."""
    if map is not None:
        assert map in params, (
            f"Step parameter 'map' must refer to one of the function arguments, but found arguments"
            f"'{params}' and parameter '{map}'."
        )


def get_function_params(fn: Callable[..., Any]) -> list[inspect.Parameter]:
    """From a given `fn`, return a list of inspected function parameters."""
    return list(inspect.signature(fn).parameters.values())


def bfs_successors(graph: nx.DiGraph, node: str) -> list[str]:  # pragma: no cover (currently not used)
    """The names of all successors to `node`"""
    result: list[str] = list(nx.bfs_tree(graph, node))[1:]
    return result


def bfs_predecessors(graph: nx.DiGraph, node: str) -> list[str]:  # pragma: no cover (currently not used)
    """The names of all predecessors to `node`"""
    result: list[str] = list(nx.bfs_tree(graph.reverse(copy=False), node))[1:]
    return result


def topological_successors(graph: nx.DiGraph, node: str) -> list[list[str]]:
    """The names of all invalidated nodes (grouped in generations) if `node` changed"""
    bfs_tree = nx.bfs_tree(graph, node)
    subgraph = nx.induced_subgraph(graph, bfs_tree.nodes)
    generations = nx.topological_generations(subgraph)
    return list(generations)


def topological_predecessors(graph: nx.DiGraph, node: str) -> list[list[str]]:
    """The names of all dependency nodes (grouped in generations) for `node`"""
    generations = topological_successors(graph.reverse(copy=False), node)
    return list(reversed(generations))


def topological_generations(graph: nx.DiGraph) -> list[list[str]]:
    """The names of all nodes in the graph grouped into generations"""
    return list(nx.topological_generations(graph))
