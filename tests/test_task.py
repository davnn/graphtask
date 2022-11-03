"""
Test the functionality of a `Task`.
"""
from re import escape

import pytest
from hypothesis import given

from graphtask import Task
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


@given(data=list_of_iterables)
def test_identity_split(data):
    """Splitting an identity function does not change the input data"""
    task = Task()
    task.register(data=data)
    task.step(identity, split="data")
    assert data == task.run()


@given(data=dict_of_iterables)
def test_identity_map(data):
    """Mapping an identity function does not change the input data"""
    task = Task()
    task.register(data=data)
    task.step(fn=identity, map="data")
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


def test_assertions():
    task = Task()

    def fn(x):
        ...

    def fn_args(x, *args):
        ...

    def fn_kwargs(x, **kwargs):
        ...

    # step
    with pytest.raises(AssertionError, match="Cannot name node '<lambda>'"):
        task = Task()
        task.step(fn=lambda: None)

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

    with pytest.raises(AssertionError, match="Cannot combine 'split' and 'map'"):
        task = Task()
        task.step(fn=fn, split="data", map="data")

    with pytest.raises(AssertionError, match="Step argument 'split' must refer to one of the parameters"):
        task = Task()
        task.step(fn=fn, split="data")

    with pytest.raises(AssertionError, match="Step argument 'map' must refer to one of the parameters"):
        task = Task()
        task.step(fn=fn, map="data")

    with pytest.raises(AssertionError, match="Cannot verify that predicate 'is_dag' holds"):
        task = Task()
        task.step(lambda: None, rename="cyclic")
        task.step(lambda cyclic: None, rename="cyclic")

    # materialize
    with pytest.raises(AssertionError, match=f"Node 'x' not defined, but set as a dependency."):
        task = Task()
        task.step(fn=fn)
        task.run()

    # run
    with pytest.raises(AssertionError, match="The 'node' must be in Task"):
        task = Task()
        task.run(node="missing")

    with pytest.raises(AssertionError, match="Cannot verify that predicate 'is_dag' holds"):
        # this should only happen if a user messes with the underlying `_graph`
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
