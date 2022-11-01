# Contributing to `graphtask`

Thanks for contributing on graphtask! Before implementing features and changes, please submit an issue to discuss
the proposed changes.

## How to submit a pull request

1. [Fork this repository](https://github.com/davnn/graphtask/fork).
2. Clone the forked repository and add a new branch with the feature name.

Before submitting your code as a pull request please do the following steps:

1. Add any changes you want
2. Add tests for the new changes
3. Edit documentation if you have changed something significant
4. Run `make format` to format your changes.
5. Run `make test` to verify that all tests are passing.
6. Run `make lint` to ensure types, security and docstrings are okay.

Conveniently, you can run `make submit`, to combine all the mentioned commands. We use [gitmoji](https://gitmoji.dev/)
to categorize different kinds of commits.

## Contributing without `make`

We use [make](https://www.gnu.org/software/make/) to provide pre-configured CLI commands for the project, but `make` is
not required, you can also run the commands directly from the CLI. Have a look at `Makefile` for a reference of
commands.

## Set up a poetry environment

We use [miniconda](https://docs.conda.io/en/latest/miniconda.html) to set up a Python environment
for [poetry](https://python-poetry.org/). Make sure that conda is installed and available. Use `make conda-env-create`
to create an empty Python 3.9 environment named `graphtask`. After you have successfully created the conda environment
activate it with `conda activate graphtask`. If `poetry` is not already installed, run `make poetry-install`. Using the
activated `graphtask` environment check if poetry is using the right environment with `poetry env info`. Once the
poetry setup is complete, you are ready to install the dependencies.

Note that Conda is not necessary, you can use a tool of your choice to manage your Python environment. One benefit of
using Conda is that we can override packages that are not easy to install with `pip`, for example `pygraphviz`.

## Install dependencies

We use [`poetry`](https://github.com/python-poetry/poetry) to manage the dependencies. With an active poetry env,
run `make install` to install all dependencies into environment. After the dependencies are installed, run
`make pre-commit-install` to add the [pre-commit](https://pre-commit.com/) hooks.

## Other help

You can contribute by spreading a word about this library. It would also be a huge contribution to write a short article
on how you are using this project. You can also share your best practices with us.
