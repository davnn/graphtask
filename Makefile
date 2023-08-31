#* Variables
SHELL := /usr/bin/env bash
PYTHON := python
PYTHONPATH := `pwd`
PROJECT := graphtask
CONDA := micromamba

#* Poetry installation
.PHONY: poetry-install
poetry-install:
	curl -sSL https://install.python-poetry.org | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://install.python-poetry.org | $(PYTHON) - --uninstall

#* Python environment
.PHONY: env-create
env-create:
	$(CONDA) env create --yes --name $(PROJECT) conda-forge::python=3.10

.PHONY: env-remove
env-remove:
	$(CONDA) env remove --name $(PROJECT)

#* Package installation
.PHONY: update
update:
	poetry update --no-interaction

.PHONY: install
install:
	poetry install --all-extras --no-interaction

.PHONY: fix
fix:
	$(CONDA) install --yes --freeze-installed conda-forge::pygraphviz=1.11.0

#* Pre-commit
.PHONY: pre-commit-install
pre-commit-install:
	poetry run pre-commit install

.PHONY: pre-commit
pre-commit:
	poetry run pre-commit run --all-files

#* Formatting
.PHONY: format
format:
	poetry run ruff check $(PROJECT) --fix
	poetry run black --config pyproject.toml $(PROJECT)

#* Testing
.PHONY: test
test:
	PYTHONPATH=$(PYTHONPATH) poetry run pytest -c pyproject.toml --cov-report=html --cov=$(PROJECT) tests/
	poetry run coverage-badge -o assets/images/coverage.svg -f

#* Linting
.PHONY: codestyle
codestyle:
	poetry run ruff check $(PROJECT)
	poetry run black --diff --check --config pyproject.toml $(PROJECT)

.PHONY: pyright
pyright:
	poetry run pyright

.PHONY: safety
safety:
	poetry run safety check --full-report
	poetry run bandit -ll --recursive $(PROJECT) $(TESTS)

.PHONY: lint
lint: codestyle pyright safety

#* Combined checks for pull requests
.PHONY: submit
submit: format test lint

#* Building and publishing
.PHONY: build
build:
	poetry build

.PHONY: publish
publish:
	poetry publish --skip-existing

#* Documentation
.PHONY: docs
docs:
	poetry run sphinx-build -b html docs docs/_build/html

doctest:
	poetry run sphinx-build -b doctest docs docs/_build/doctest

autodocs:
	poetry run sphinx-autobuild docs docs/_build/html --watch $(PROJECT) --watch examples --ignore docs/_gallery --open-browser

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove ipynbcheckpoints-remove pytestcache-remove build-remove

#* Debug information
.PHONY: sysinfo
sysinfo:
	poetry run python .github/system_info.py
