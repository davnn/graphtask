"""
Test the representation of a `Task`
"""
from graphtask import Task


def test_str():
    task = Task()
    assert str(task) == "Task(n_jobs=1)"


def test_repr():
    # empty nodes
    task = Task()
    assert repr(task) == str(task)

    # nodes without edges
    task = Task()
    task.register(a=1, b=2)
    assert repr(task) == "Task(n_jobs=1)\no    a,b"

    # nodes and edges
    task = Task()
    task.register(a=1)
    task.step(fn=lambda a: a, rename="b")
    assert repr(task) == "Task(n_jobs=1)\no    a\n|\no    b"
