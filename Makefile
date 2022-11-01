#* Variables
SHELL := /usr/bin/env bash
PYTHON := python3
PYTHONPATH := `pwd`
PROJECT := graphtask

#* Poetry installation
.PHONY: poetry-install
poetry-install:
	curl -sSL https://install.python-poetry.org | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://install.python-poetry.org | $(PYTHON) - --uninstall

#* Conda installation
.PHONY: conda-env-create
conda-env-create:
	conda create --yes --name $(PROJECT) python=3.9

.PHONY: conda-env-remove
conda-env-remove:
	conda env remove --name $(PROJECT)

#* Package installation
.PHONY: install
install:
	poetry install --all-extras --no-interaction

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
	poetry run pyupgrade --exit-zero-even-if-changed --py39-plus **/*.py
	poetry run isort --settings-path pyproject.toml ./
	poetry run black --config pyproject.toml ./

#* Testing
.PHONY: test
test:
	PYTHONPATH=$(PYTHONPATH) poetry run pytest -c pyproject.toml --cov-report=html --cov=graphtask tests/
	poetry run coverage-badge -o assets/images/coverage.svg -f

#* Linting
.PHONY: codestyle
codestyle:
	poetry run isort --diff --check-only --settings-path pyproject.toml ./
	poetry run black --diff --check --config pyproject.toml ./
	poetry run darglint --verbosity 2 graphtask tests

.PHONY: pyright
pyright:
	poetry run pyright

.PHONY: safety
safety:
	poetry check
	poetry run safety check --full-report
	poetry run bandit -ll --recursive graphtask tests

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
