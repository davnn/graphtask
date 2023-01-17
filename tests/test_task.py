"""
Test the functionality of a `Task`.
"""
from re import escape
from typing import get_args

import pytest
from hypothesis import given

from graphtask import Task
from graphtask._task import MapTypeT
from tests import *


def identity(data):
    return data


@given(data=basics)
def test_identity(data):
    """Applying an identity function does not change the input data"""
    task = Task()
    task.register(data=data)
    task.step(fn=identity)
    assert data == task.run()


@given(data=dict_of_iterables)
@pytest.mark.parametrize("map_type", get_args(MapTypeT))
def test_identity_map_mappable(data, map_type):
    """Splitting an identity function does not change the input data"""
    task = Task()
    task.register(data=data)
    task.step(identity, map_arg="data", map_type=map_type)
    assert data == task.run()


@given(data=list_of_iterables)
def test_identity_map_iterable(data):
    """Splitting an identity function does not change the input data"""
    task = Task()
    task.register(data=data)
    task.step(identity, map_arg="data", map_type="values")
    assert data == task.run()


@given(n_nodes=int_gt_1_lt_max, data=anything)
def test_split_nodes(n_nodes, data):
    task = Task()
    task.register(data=data)
    for n in range(n_nodes):
        task.step(fn=identity, rename=f"identity_{n}")
    result = task.run()

    # the number of results should equal the number of leaf nodes
    assert len(result) == n_nodes

    # each result should equal the input (identity)
    for r in result:
        assert r == data


@given(n_nodes=int_gt_1_lt_max, data=anything)
def test_combine_nodes(n_nodes, data):
    task = Task()
    task.register(data=data)
    nodes = [f"identity_{n}" for n in range(n_nodes)]
    for node in nodes:
        task.step(fn=identity, rename=node)

    # Combine using *args
    task.step(fn=lambda *args: list(args), args=nodes, rename="args")

    # Combine using **kwargs
    task.step(fn=lambda **kwargs: list(kwargs.values()), kwargs=nodes, rename="kwargs")

    # Run the task
    result_args, result_kwargs = task.run()

    # Make sure the output size is as expected
    assert len(result_args) == n_nodes
    assert len(result_kwargs) == n_nodes

    # Make sure the output type is as expected
    assert isinstance(result_args, list)
    assert isinstance(result_kwargs, list)

    # Make sure the values are unchanged as expected (identity fn)
    for r in [*result_args, *result_kwargs]:
        assert r == data


@given(data=text)
def test_string_args_kwargs(data):
    task = Task()
    task.register(data=data)
    task.step(lambda *args: args[0], args="data", rename="args")
    task.step(lambda **kwargs: list(kwargs.values())[0], kwargs="data", rename="kwargs")
    for result in task.run():
        assert result == data


@given(n_nodes=int_gt_1_lt_max, data=anything)
def test_run_nodes(n_nodes, data):
    task = Task()
    task.register(data=data)
    repeated_identity = lambda repeats: lambda data: tuple(data for _ in range(repeats))
    nodes = [f"identity_{n}" for n in range(n_nodes)]
    for i, node in enumerate(nodes):
        task.step(fn=repeated_identity(i), rename=node)

    # each node should return the identity repeated `i` times
    for repeats, node in enumerate(nodes):
        result = task.run(node)
        assert len(result) == repeats
        for item in result:
            assert data == item


@given(alias=text, data=anything)
def test_aliasing(alias, data):
    task = Task()
    task.register(**{alias: data})
    task.step(fn=identity, alias={"data": alias})
    assert data == task.run()


def test_assertions():
    def fn(x):
        ...

    def fn_args(x, *args):
        ...

    def fn_kwargs(x, **kwargs):
        ...

    def fn_args_kwargs(*args, **kwargs):
        ...

    with pytest.raises(AssertionError, match=escape("Variable argument '*args' requires 'args' parameter")):
        task = Task()
        task.step(fn=lambda *args: args)

    with pytest.raises(AssertionError, match=escape("Variable argument '**kwargs' requires 'kwargs' parameter")):
        task = Task()
        task.step(fn=lambda **kwargs: kwargs)

    with pytest.raises(AssertionError, match=escape("The names provided to 'args' cannot be duplicates")):
        task = Task()
        task.step(fn=fn_args, args=["x"])

    with pytest.raises(AssertionError, match=escape("The names provided to 'kwargs' cannot be duplicates")):
        task = Task()
        task.step(fn=fn_kwargs, kwargs=["x"])

    with pytest.raises(AssertionError, match=escape("There cannot be duplicate names provided to 'args' and 'kwargs'")):
        task = Task()
        task.step(fn=fn_args_kwargs, args="x", kwargs="x")

    with pytest.raises(AssertionError, match="Step argument 'map' must refer to one of the parameters"):
        task = Task()
        task.step(fn=fn, map_arg="data")

    with pytest.raises(AssertionError, match="The parameter 'map_type' must be one of"):
        task = Task()
        task.step(fn=fn, map_arg="x", map_type="nonexistant")

    with pytest.raises(AssertionError, match="Cannot verify that predicate 'is_dag' holds"):
        task = Task()
        task.step(lambda b: None, rename="a")
        task.step(lambda a: None, rename="b")

    # materialize
    with pytest.raises(AssertionError, match=f"Node 'x' not defined, but set as a dependency."):
        task = Task()
        task.step(fn=fn)
        task.run()

    with pytest.raises(AssertionError, match=f"Parameter 'map' requires an iterable input argument"):
        task = Task()
        task.register(x=1)
        task.step(fn=fn, map_arg="x")
        task.run()

    with pytest.raises(AssertionError, match=f"Cannot use 'map_type=keys' on non-mappable argument"):
        task = Task()
        task.register(data=[1, 2, 3])
        task.step(identity, map_arg="data", map_type="keys")
        task.run()

    with pytest.raises(AssertionError, match=f"Cannot use 'map_type=items' on non-mappable argument"):
        task = Task()
        task.register(data=[1, 2, 3])
        task.step(identity, map_arg="data", map_type="items")
        task.run()

    # run
    with pytest.raises(AssertionError, match="The 'node' must be in Task"):
        task = Task()
        task.register(data=[1, 2, 3])
        task.run(node="missing")

    with pytest.raises(AssertionError, match="Cannot verify that predicate 'is_dag' holds"):
        # should only happen if a user messes with the underlying `_graph`, otherwise should already fail in `step`
        task = Task()
        task._graph.add_edges_from([("cyclic", "cyclic")])
        task.run()


def test_warnings():
    with pytest.warns(UserWarning, match=escape("Provided 'args' argument for 'step', but no '*args'")):
        task = Task()
        task.step(fn=lambda: None, args=["x"], rename="test")

    with pytest.warns(UserWarning, match=escape("Provided 'kwargs' argument for 'step', but no '**kwargs'")):
        task = Task()
        task.step(fn=lambda: None, kwargs=["x"], rename="test")
