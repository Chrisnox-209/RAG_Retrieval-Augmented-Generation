MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
--disallow-untyped-defs --check-untyped-defs
MYPY_STRICT = --strict
FLAKE_STRICT = --max-complexity=25


.PHONY: all install run clean lint lint-strict debug

all: run

install:
	@uv sync

run:
	@uv run python -m src

clean:
	@rm -Rf .venv
	@rm -Rf __pycache__
	@rm -Rf .mypy_cache
	@find data -mindepth 1 -maxdepth 1 ! -name 'datasets' ! -name 'raw' -exec rm -rf -- {} +
	@echo "All code clean"
lint:
	@uv run python -m mypy src $(MYPY_FLAGS)
	@uv run python -m flake8 --exclude .venv src

lint-strict:
	@uv run python -m mypy src  $(MYPY_FLAGS) $(MYPY_STRICT)
	@uv run python -m flake8 --exclude .venv src $(FLAKE_STRICT)

debug:
	@uv run python -m pdb $(MAIN)
