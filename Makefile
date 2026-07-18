SHELL := /bin/sh

MYPY_CONFIG := $(CURDIR)/pyproject.toml
TEX_DIRS := proposal pres
TEX_MAIN ?= main

.DEFAULT_GOAL := help

.PHONY: help
help: ## Print this help message.
	@printf "Available make targets:\n"
	@awk 'BEGIN {FS = ":.*## "}; /^[A-Za-z0-9_.-]+:.*## / {printf "  %-30s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: install
install: ## Sync all Python dependencies with uv.
	uv sync

.PHONY: ruff-format
ruff-format: ## Check Python and notebook formatting with ruff.
	uv run ruff format --check .

.PHONY: ruff-format-fix
ruff-format-fix: ## Format Python and notebook files with ruff.
	uv run ruff format .

.PHONY: ruff-lint
ruff-lint: ## Lint Python and notebook files with ruff.
	uv run ruff check .

.PHONY: ruff-fix
ruff-fix: ## Lint and auto-fix Python and notebook files with ruff.
	uv run ruff check --fix .

.PHONY: ruff
ruff: ruff-format ruff-lint ## Run all ruff validation checks.

.PHONY: mypy
mypy: ## Run mypy over the Python sources.
	uv run mypy --config-file "$(MYPY_CONFIG)" .

.PHONY: requirements
requirements: ## Export the locked dependencies to requirements.txt for DataSphere jobs.
	uv export --quiet --frozen --no-hashes --no-emit-project --no-annotate --no-header \
		| uv run --quiet python -c 'import sys; \
		from packaging.requirements import Requirement; \
		environment = {"sys_platform": "linux", "platform_system": "Linux", "os_name": "posix", \
		               "platform_machine": "x86_64", "platform_python_implementation": "CPython", \
		               "implementation_name": "cpython", "python_version": "3.12", \
		               "python_full_version": "3.12.13"}; \
		[print(line.split(";")[0].strip()) for line in map(str.strip, sys.stdin) \
		 if line and ((marker := Requirement(line).marker) is None or marker.evaluate(environment))]' \
		> requirements.txt

.PHONY: pdf
pdf: ## Compile every LaTeX document into its PDF.
	@set -eu; \
	for dir in $(TEX_DIRS); do \
		(cd "$$dir" && latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX_MAIN).tex); \
	done

.PHONY: quality
quality: ruff mypy ## Run all quality checks.
