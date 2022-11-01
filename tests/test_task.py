import pytest
from hypothesis import given
from hypothesis import strategies as s

from graphtask import Task

# defines test strategies
n_nodes = s.integers(min_value=2, max_value=100)
any_value = s.one_of(s.text(), s.lists(s.text()), s.dictionaries(keys=s.text(), values=s.text()))
split_data = s.lists(any_value, max_size=5)
map_data = s.dictionaries(keys=s.text(), values=split_data, max_size=5)
any_data = s.one_of(any_value, split_data, map_data)


def identity(data):
    return data


@given(data=any_data)
def test_identity_any(data):
    """Applying an identity function does not change the input data"""
    task = Task()
    task.register(data=data)
    task.step(fn=identity)
    assert data == task.run()


@given(data=split_data)
def test_identity_split(data):
    """Splitting an identity function does not change the input data"""
    task = Task()
    task.register(data=data)
    task.step(identity, split="data")
    assert data == task.run()


@given(data=map_data)
def test_identity_map(data):
    """Mapping an identity function does not change the input data"""
    task = Task()
    task.register(data=data)
    task.step(fn=identity, map="data")
    assert data == task.run()


@given(n_nodes=n_nodes, data=any_data)
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


@given(n_nodes=n_nodes, data=any_data)
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


def test_step_assertions():
    task = Task()

    def fn(x):
        ...

    with pytest.raises(AssertionError, match="Variable arguments.+"):
        # Cannot use *args without step(args=...)
        task.step(fn=lambda *args: args)

    with pytest.raises(AssertionError, match="Variable keyword"):
        # Cannot use **kwargs without step(kwargs=...)
        task.step(fn=lambda **kwargs: kwargs)

    with pytest.raises(AssertionError, match="Cannot combine `split` and `map`"):
        task.step(fn=fn, split="data", map="data")

    with pytest.raises(AssertionError, match="Argument `split` must refer to one of the parameters"):
        task.step(fn=fn, split="data")

    with pytest.raises(AssertionError, match="Argument `map` must refer to one of the parameters"):
        task.step(fn=fn, map="data")

    with pytest.raises(AssertionError, match="Cannot name node '<lambda>'"):
        task.step(fn=lambda: None)

    with pytest.raises(AssertionError, match="Cannot find 'x' in the graph"):
        task.step(fn=fn)

    with pytest.raises(AssertionError, match="Cannot verify that predicate 'is_dag' holds"):
        task.step(lambda: None, rename="cyclic")
        task.step(lambda cyclic: None, rename="cyclic")


def test_run_assertions():
    task = Task()

    with pytest.raises(AssertionError, match="The 'node' must be in Task"):
        task.run(node="missing")

    with pytest.raises(AssertionError, match="Cannot verify that predicate 'is_dag' holds"):
        task._graph.add_edges_from([("cyclic", "cyclic")])
        task.run()
