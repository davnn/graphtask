API
===

The public interface of *graphtask* consists of a ``Task`` and ``step``. A ``Task`` makes it possible to conveniently
create a DAG of functions or methods. The DAG does not have to be specified, it is implicitly created based on the
dependencies of the functions and their arguments. The following examples implicitly create a DAG with three nodes
``(a, b, c)`` and three edges ``((a, b), (a, c), (b, c))``.

>>> task = Task()
>>> @task.step
... def a(): return 1
>>> @task.step
... def b(a): return a * 2
>>> @task.step
... def c(a,b): return a * b ** 2
>>> task.run()
4

It is also possible to inherit from ``Task``, but then it is not possible to use the ``step`` decorator defined on the
inherited ``Task``, because it would only be defined after class instantiation. We could, however, add the steps
manually directly in the ``__init__`` method of the inherited class.

>>> class MyTask(Task):
...     def __init__(self):
...         super().__init__()
...         self.step(self.a)
...         self.step(self.b)
...         self.step(self.c)
...     def a(self):
...         return 1
...     def b(self, a):
...         return a * 2
...     def c(self, a, b):
...         return a * b ** 2
>>> task = MyTask()
>>> task.run()
4

Another possible option is to use the exported ``step`` decorator provided by *graphtask*. The step decorator marks
a method such that it automatically get added as a ``step`` during class instantiation.

>>> class MyTask(Task):
...     @step
...     def a(self):
...         return 1
...     @step
...     def b(self, a):
...         return a * 2
...     @step
...     def c(self, a, b):
...         return a * b ** 2
>>> task = MyTask()
>>> task.run()
4

Module
------

.. automodule:: graphtask
    :members:
