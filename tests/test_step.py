"""
Test the equivalence of `step` and the `@step` decorator.
"""
from hypothesis import given

from graphtask import Task, step
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
    for fn in [fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8, fn9, fn10, fn11]:
        task.step(fn=fn, args=["c"], kwargs=["d"])

    for result in task.run():
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

    for result in task.run():
        assert a + b * c + d == result


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
    for result in task.run():
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

    assert task_fun.run() == task_dec.run() == task_cls.run()
