MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
--disallow-untyped-defs --check-untyped-defs
MYPY_STRICT = --strict
FLAKE_STRICT = --max-complexity=25


.PHONY: all install run clean lint lint-strict debug index search moulinette full evaluate

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


# ### index
K = 5
max_chunk_size = 2000
chunk_overlap_percent = 33
source_path = data/raw/vllm-0.10.1
processed_directory = data/processed

# ### search
dataset_path_docs = data/datasets/UnansweredQuestions/dataset_docs_public.json
dataset_path_code = data/datasets/UnansweredQuestions/dataset_code_public.json
save_directory_search = data/output/search_results/UnansweredQuestions

# ### moulinette && evaluate
moulinette_path_docs = data/output/search_results/UnansweredQuestions/dataset_docs_public.json 
moulinette_path_code = data/output/search_results/UnansweredQuestions/dataset_code_public.json 
moulinette_dataset_docs = data/datasets/AnsweredQuestions/dataset_docs_public.json
moulinette_dataset_code = data/datasets/AnsweredQuestions/dataset_code_public.json

# ### answer
student_search_results_path_docs = data/output/search_results/UnansweredQuestions/dataset_docs_public.json
student_search_results_path_code = data/output/search_results/UnansweredQuestions/dataset_code_public.json
save_directory = data/output/search_results_and_answer/UnansweredQuestions


index:
	@uv run python3 -m src index \
		--max_chunk_size ${max_chunk_size} \
		--chunk_overlap_percent ${chunk_overlap_percent} \
		--source_path $(source_path) \
		--processed_directory $(processed_directory)

search:
	@uv run python3 -m src search_dataset \
		--dataset_path $(dataset_path_docs) \
		--k $(K) \
		--save_directory $(save_directory_search)
	@uv run python3 -m src search_dataset \
		--dataset_path $(dataset_path_code) \
		--k $(K) \
		--save_directory $(save_directory_search)

moulinette:
	@echo "\n====== DOCS ======"
	@./moulinette-ubuntu evaluate_student_search_results \
		$(moulinette_path_docs) \
		$(moulinette_dataset_docs) \
		--k $(K) \
		--max_context_length ${max_chunk_size}

	@echo "\n====== CODE ======"
	@./moulinette-ubuntu evaluate_student_search_results \
		$(moulinette_path_code) \
		$(moulinette_dataset_code) \
		--k $(K) \
		--max_context_length ${max_chunk_size}

answer:
	@uv run python3 -m src answer_dataset \
		--student_search_results_path $(student_search_results_path_docs) \
		--save_directory $(save_directory)

	@uv run python3 -m src answer_dataset \
		--student_search_results_path $(student_search_results_path_code) \
		--save_directory $(save_directory)

evaluate:
	@uv run python3 -m src evaluate \
		--student_search_results_path $(moulinette_path_docs) \
		--dataset_path $(moulinette_dataset_docs)

	@uv run python3 -m src evaluate \
		--student_search_results_path $(moulinette_path_code) \
		--dataset_path $(moulinette_dataset_code)

full: index search moulinette