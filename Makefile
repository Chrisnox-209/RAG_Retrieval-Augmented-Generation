MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
--disallow-untyped-defs --check-untyped-defs
MYPY_STRICT = --strict
FLAKE_STRICT = --max-complexity=25


.PHONY: all install run clean lint lint-strict debug index search moulinette full

all: run

install:
	@uv sync

run:
	@uv run python -m src

clean:
	@rm -Rf .venv
	@rm -Rf __pycache__
	@rm -Rf src/__pycache__
	@rm -Rf test/__pycache__
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
	@uv run python -m pdb -m src

index:
	@uv run python3 -m src index \
		--max_chunk_size 2000 \
		--chunk_overlap_percent 17 \
		--source_path data/raw/vllm-0.10.1 \
		--processed_directory data/processed

search:
	@uv run python3 -m src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
		--k 10 \
		--save_directory data/output/search_results/UnansweredQuestions
	@uv run python3 -m src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
		--k 10 \
		--save_directory data/output/search_results/UnansweredQuestions

moulinette:
	@echo "\n====== DOCS ======"
	@./moulinette-ubuntu evaluate_student_search_results \
		data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
		data/datasets/AnsweredQuestions/dataset_docs_public.json \
		--k 10 \
		--max_context_length 2000
	@echo "\n====== CODE ======"
	@./moulinette-ubuntu evaluate_student_search_results \
		data/output/search_results/UnansweredQuestions/dataset_code_public.json \
		data/datasets/AnsweredQuestions/dataset_code_public.json \
		--k 10 \
		--max_context_length 2000

full: index search moulinette