"""
Test the representation of a `Task`
"""
from graphtask import Task


def test_str():
    task = Task()
    assert str(task) == "Task(n_jobs=1, backend=ThreadingBackend)"


def test_empty():
    # empty nodes
    task = Task()
    assert repr(task) == str(task)


def test_nodes():
    # nodes without edges
    task = Task()
    task.register(a=1, b=2)
    assert repr(task) == "Task(n_jobs=1, backend=ThreadingBackend)\no    a,b"


def test_nodes_edges():
    # nodes and edges
    task = Task()
    task.register(a=1)
    task.step(fn=lambda a: a, rename="b")
    assert repr(task) == "Task(n_jobs=1, backend=ThreadingBackend)\no    a\n|\no    b"


def test_subtask():
    task = Task()
    task.register(a=1)
    task.step(fn=lambda a: a, rename="b")
    task.step(fn=lambda b: b, rename="c")
    assert repr(task) == "Task(n_jobs=1, backend=ThreadingBackend)\no    a\n|\no    b\n|\no    c"
    assert repr(task["b"]) == "Task(n_jobs=1, backend=ThreadingBackend)\no    a\n|\no    b"
    assert repr(task["a"]) == "Task(n_jobs=1, backend=ThreadingBackend)\no    a"
