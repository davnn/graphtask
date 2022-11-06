Installation
============

Graphtask is a lightweight `Python <https://www.python.org/>`_-only package with minimal dependencies. Use one of the
following commands to install the package.

.. grid:: 2

    .. grid-item-card::

        **Install using conda**
        ^^^
        *graphtask* is part of `conda-forge <https://anaconda.org/conda-forge/graphtask/>`_ and can be installed with:
        +++
        .. code-block:: bash

            conda install -c conda-forge graphtask

    .. grid-item-card::

        **Install using pip**
        ^^^
        *graphtask* is registered at `PyPI <https://pypi.org/project/graphtask/>`_ and can be installed with:
        +++
        .. code-block:: bash

            pip install graphtask

If you install the package using pip, the `PyGraphviz <https://pygraphviz.github.io/>`_ dependency is optional
and can be added with ``pip install graphtask[visualize]``. This is because PyGraphviz is not trivial to set up
and most users will not have `Graphviz <https://graphviz.org/>`_ installed. With conda, the package is automatically
added because PyGraphviz is bundled and pre-built on conda.
