Tutorial
========

Graphtask is a lightweight `Python <https://www.python.org/>`_ package designed to simplify common processing tasks by
implicitly modeling them as acyclic, directed graphs (DAGs). DAGs are a very general concept and apply to a lot of
different computing tasks.

Traditionally, you would have to define your processing tasks *explicitly* as a DAG by specifying how different steps
of your tasks relate to each other. Graphtask, however, can infer the structure of the DAG, without you having to
define it. This results in *pure*, *testable* and *declarative* functions and the complex relationships between the
functions is automatically inferred for you.

Creating your first Task
------------------------

There are two ways to create task:

1. Instantiate a :class:`graphtask.Task` object which stores the created DAG.
2. Inherit from :class:`graphtask.Task` to create a DAG for your own objects.

Let's start by instantiating a :class:`graphtask.Task`.

>>> task = Task()

Once the task is instantiated, we can add a :meth:`graphtask.Task.step` to the task. A step is just a function
that produces some value based on its input arguments. The input arguments are referred to as the function's
*dependencies* in the graph.

>>> @task.step
... def my_fun(a, b, c):
...     return a * b * c

We have now implicitly added a node ``my_fun`` to the task graph with three dependencies ``a``, ``b`` and ``c``. The
dependencies are other nodes in the graph, but the specified dependencies do not yet exist. Let's add the nodes ``a``,
``b`` and ``c`` to the graph using :meth:`graphtask.Task.register`.

>>> task.register(a=1, b=2, c=3)

:meth:`graphtask.Task.register` lets us add arbitrary values to the graph. Remember that a node is always just a
function; :meth:`graphtask.Task.register` can thus be seen as a shorthand for:

>>> @task.step
... def a():
...     return 1
...
>>> @task.step
... def b():
...     return 2
...
>>> @task.step
... def c():
...     return 3

We can now run the :class:`graphtask.Task` using :meth:`graphtask.Task.run`.

>>> task.run()
6

Graphtask knows how the functions relate and runs them in order such that the results of running the functions
``a``, ``b`` and ``c`` are ready before ``my_fun`` is run. We can have a look at the graph with ``repr(task)``:

.. code-block:: text

    Task(n_jobs=1)
    o o o    a,b,c
    |/_/
    o    my_fun
