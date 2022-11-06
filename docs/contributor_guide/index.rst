Contributor Guide
=================

Thanks for contributing on graphtask! Before implementing features and changes, please
`submit an issue <https://github.com/davnn/graphtask/issues/new/choose>`_ to discuss the proposed changes.

How to submit a pull request
----------------------------

1. `Fork this repository <https://github.com/davnn/graphtask/fork>`_
2. Clone the forked repository and add a new branch with the feature name.

Before submitting your code as a pull request please do the following steps:

1. Add any changes you want
2. Add tests for the new changes
3. Edit documentation if you have changed something significant
4. Run ``make format`` to format your changes.
5. Run ``make test`` to verify that all tests are passing.
6. Run ``make lint`` to ensure types, security and docstrings are okay.

Conveniently, you can run ``make submit``, to combine all the mentioned commands. If you do not have ``make`` installed,
you can directly run the commands specified in ``Makefile``.

For more information on how to contribute, see `CONTRIBUTING.md <https://github.com/davnn/graphtask/blob/main/.github/CONTRIBUTING.md>`_.
