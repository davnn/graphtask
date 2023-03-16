"""
Test the equivalence of `step` and the `@step` decorator.
"""
from collections.abc import Callable

import pytest
from hypothesis import given

from graphtask import Task, step
from graphtask._step import InvalidStepArgumentError
from tests import *


@given(a=ints, b=ints, c=ints, d=ints)
def test_fn_signatures(a, b, c, d):
    def fn1(a, b, c, d):
        return a + b * c + d

    def fn2(*, a, b, c, d):
        return a + b * c + d

    def fn3(a, b, *, c, d):
        return a + b * c + d

    def fn4(a, b, c, d, /):
        return a + b * c + d

    def fn5(a, b, /, c, d):
        return a + b * c + d

    def fn6(a, /, b, *, c, d):
        return a + b * c + d

    def fn7(a, b, /, c, *, d):
        return a + b * c + d

    def fn8(a, b, /, *, c, d):
        return a + b * c + d

    def fn9(a, /, b, *c, **d):
        return a + b * c[0] + list(d.values())[0]

    def fn10(a, b, /, *c, **d):
        return a + b * c[0] + list(d.values())[0]

    def fn11(a, /, *c, b, **d):
        return a + b * c[0] + list(d.values())[0]

    task = Task()
    task.register(a=a, b=b, c=c, d=d)
    for fn in [fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8]:
        task.step(fn=fn)

    for fn in [fn9, fn10, fn11]:
        task.step(fn=fn, args=["c"], kwargs=["d"])

    for result in task():
        assert a + b * c + d == result


@given(a=ints, b=ints, c=ints, d=ints)
def test_decorator_signatures(a, b, c, d):
    task = Task()
    task.register(a=a, b=b, c=c, d=d)

    @task.step
    def fn1(a, b, c, d):
        return a + b * c + d

    @task.step
    def fn2(*, a, b, c, d):
        return a + b * c + d

    @task.step
    def fn3(a, b, *, c, d):
        return a + b * c + d

    @task.step
    def fn4(a, b, c, d, /):
        return a + b * c + d

    @task.step
    def fn5(a, b, /, c, d):
        return a + b * c + d

    @task.step
    def fn6(a, /, b, *, c, d):
        return a + b * c + d

    @task.step
    def fn7(a, b, /, c, *, d):
        return a + b * c + d

    @task.step
    def fn8(a, b, /, *, c, d):
        return a + b * c + d

    @task.step(args=["c"], kwargs=["d"])
    def fn9(a, /, b, *c, **d):
        return a + b * c[0] + list(d.values())[0]

    @task.step(args=["c"], kwargs=["d"])
    def fn10(a, b, /, *c, **d):
        return a + b * c[0] + list(d.values())[0]

    @task.step(args=["c"], kwargs=["d"])
    def fn11(a, /, *c, b, **d):
        return a + b * c[0] + list(d.values())[0]

    steps = [fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8, fn9, fn10, fn11]
    for step, result in zip(steps, task()):
        assert a + b * c + d == result == step()


def test_invalid_signature_raises():
    task_fun = Task()
    task_fun.register(a=0)

    @task_fun.step
    def pos_or_kw(a):
        return a

    @task_fun.step
    def pos_only(a, /):
        return a

    @task_fun.step
    def kw_only(*, a):
        return a

    class T(Task):
        @step
        def pos_or_kw(self, a):
            return a

        @step
        def pos_only(self, a, /):
            return a

        @step
        def kw_only(self, *, a):
            return a

    task_cls = T()
    task_cls.register(a=0)

    def raises_invalid_step(match: str, *examples: Callable):
        """Check if a given callable exception raises an ``InvalidStepArgumentError`` for a given ``match``"""
        for example in examples:
            with pytest.raises(InvalidStepArgumentError, match=match):
                example()

    raises_invalid_step(
        "Step takes 1 positional arguments",
        lambda: pos_or_kw(1, 2),
        lambda: task_cls.pos_or_kw(1, 2)
    )

    raises_invalid_step(
        "Step takes 0 positional arguments",
        lambda: kw_only(0),
        lambda: task_cls.kw_only(0)
    )

    raises_invalid_step(
        "Step got unexpected keyword arguments",
        lambda: pos_or_kw(nonexistant=0),
        lambda: task_cls.pos_or_kw(nonexistant=0)
    )

    raises_invalid_step(
        "Step got positional-only arguments passed as keyword",
        lambda: pos_only(a=0),
        lambda: task_cls.pos_only(a=0)
    )

    raises_invalid_step(
        "Step got duplicate arguments",
        lambda: pos_or_kw(0, a=0),
        lambda: task_cls.pos_or_kw(0, a=0)
    )


@given(a=ints, b=ints, c=ints, d=ints)
def test_method_signatures(a, b, c, d):
    class T(Task):
        @step
        def fn1(self, a, b, c, d):
            return a + b * c + d

        @step
        def fn2(self, *, a, b, c, d):
            return a + b * c + d

        @step
        def fn3(self, a, b, *, c, d):
            return a + b * c + d

        @step
        def fn4(self, a, b, c, d, /):
            return a + b * c + d

        @step
        def fn5(self, a, b, /, c, d):
            return a + b * c + d

        @step
        def fn6(self, a, /, b, *, c, d):
            return a + b * c + d

        @step
        def fn7(self, a, b, /, c, *, d):
            return a + b * c + d

        @step
        def fn8(self, a, b, /, *, c, d):
            return a + b * c + d

        @step(args=["c"], kwargs=["d"])
        def fn9(self, a, /, b, *c, **d):
            return a + b * c[0] + list(d.values())[0]

        @step(args=["c"], kwargs=["d"])
        def fn10(self, a, b, /, *c, **d):
            return a + b * c[0] + list(d.values())[0]

        @step(args=["c"], kwargs=["d"])
        def fn11(self, a, /, *c, b, **d):
            return a + b * c[0] + list(d.values())[0]

    task = T()
    task.register(a=a, b=b, c=c, d=d)
    for result in task():
        assert a + b * c + d == result


@given(data=basics)
def test_equivalence(data):
    """Test if different syntactic approaches yield the same result"""

    # functional approach
    task_fun = Task()

    def identity(data):
        return data

    task_fun.register(data=data)
    task_fun.step(fn=identity)

    # decorator approach
    task_dec = Task()
    task_dec.register(data=data)

    @task_dec.step
    def identity(data):
        return data

    # class approach
    class T(Task):
        @step
        def identity(self, data):
            return data

    task_cls = T()
    task_cls.register(data=data)

    assert task_fun() == task_dec() == task_cls()


@given(data=create_nonempty_dict(basics))
def test_map_items_raises(data):
    with pytest.raises(AssertionError, match="Must return a 'key, value' tuple from function mapping over `items`"):
        task = Task()
        task.register(data=data)
        task.step(lambda data: data[0], map="data", map_type="items")
        task()


def test_wrong_step_kind_raises():
    with pytest.raises(AssertionError, match="The parameter 'map_type' must be one of"):
        task = Task()
        task.step(lambda data: data, map="data", map_type="wrong")
        task()


def test_map_noniterable_raises():
    with pytest.raises(AssertionError, match=f"Data for 'map' argument 'x' must be iterable"):
        task = Task()
        task.register(x=1)
        task.step(fn=lambda x: x, map="x")
        task()
