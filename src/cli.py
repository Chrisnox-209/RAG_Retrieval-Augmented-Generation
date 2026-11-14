from src.models import (ChunkModel, MinimalSource,
                        SourceFile, RagDataset,
                        MinimalSearchResults, StudentSearchResults)
from src.parse import parse_directory, parse_dataset
from src.chunker import files_distribution
from src.indexer import create_corpus, build_index, save_index
from src.retriever import load_index, search
from src.tests import test_check_chunk_positions
from src.utils import Color, build_minimal_sources
from src.generator import build_context, generate_answer, load_model
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import json
from tqdm import tqdm
from pathlib import Path
from bm25s import BM25
import sys


def run_indexation(path: Path, chunk_size: int, chunk_overlap_percent: int,
                   directory_info_index: str,
                   file_info_index: str,
                   directory_index: str,
                   ) -> tuple[BM25, list[dict[str, str | int]]]:
    try:
        files: list[SourceFile] = parse_directory(str(path))
        print(f"Recovered files : {len(files)}")

    except OSError as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(filesystem) "
              f"{Color.RST.value}{error}")
        sys.exit(1)

    try:
        list_chunk: list[ChunkModel] = files_distribution(
            files, chunk_size,
            chunk_overlap_percent)
        print(f"Generated chunks : {len(list_chunk)}")
    except Exception as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(chunking) "
              f"{Color.RST.value}{error}")
        sys.exit(1)

    try:
        chunk_info: list[dict[str, str | int]] = []
        corpus: list[str] = []
        corpus, chunk_info = create_corpus(list_chunk)
        retriever: BM25 = build_index(corpus)
    except Exception as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(corpus) "
              f"{Color.RST.value}{error}")
        sys.exit(1)

    try:
        save_index(directory_info_index, directory_index,
                   file_info_index, retriever,
                   corpus, chunk_info)
    except Exception as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(build_index) "
              f"{Color.RST.value}{error}")
        sys.exit(1)

    return retriever, chunk_info


def run_search(directory_info_index: str, file_info_index: str,
               directory_index: str, query: str, k: int,
               test: bool = False) -> list[dict[str, str | int | float]]:

    retriever: BM25
    chunk_info: list[dict[str, str | int]] = []

    try:
        file_info: str = directory_info_index + "/" + file_info_index
        retriever, chunk_info = load_index(file_info, directory_index)
    except Exception as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(reading_index) "
              f"{Color.RST.value}{error}")
        sys.exit(1)

    try:
        list_dict_struct: list[dict[str, str | int | float]] = search(
            chunk_info, query, retriever, k)
    except Exception as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(search_index) "
              f"{Color.RST.value}{error}")
        sys.exit(1)

    if test:
        if test_check_chunk_positions(list_dict_struct):
            print(f"{Color.GREEN.value}[VALID] {Color.BLUE.value}The "
                  "retriever returns correct paths and positions."
                  f"{Color.RST.value}")
        else:
            print(f"{Color.RED.value}[NOT VALID] {Color.BLUE.value}The "
                  f"retriever does not correctly retrace paths or return "
                  f"to positions.{Color.RST.value}")
            sys.exit(1)

    return list_dict_struct


def start_search_dataset(dataset_path: str, k: int = 5,
                         save_directory: str = (
                             "data/output/search_results/UnansweredQuestions")
                         ) -> None:
    dataset: RagDataset = parse_dataset(dataset_path)
    processed_directory: str = "data/processed"
    directory_info_index: str = str(Path(processed_directory) / "info")
    file_info_index: str = "chunk_info.json"
    directory_index: str = str(Path(processed_directory) / "bm25")
    lst_search_results: list[MinimalSearchResults] = []
    retriever: BM25
    chunk_info: list[dict[str, str | int]]

    try:
        file_info: str = directory_info_index + "/" + file_info_index
        retriever, chunk_info = load_index(file_info, directory_index)
    except Exception as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(reading_index) "
              f"{Color.RST.value}{error}")
        sys.exit(1)

    for query in tqdm(dataset.rag_questions,
                      desc="Searching",
                      unit="question",
                      colour="green",
                      dynamic_ncols=True):

        try:
            list_dict_struct: list[dict[str, str | int | float]] = search(
                chunk_info, query.question, retriever, k)
        except Exception as error:
            print(f"{Color.RED.value}[ERROR]: "
                  f"{Color.YELLOW.value}(search_index) "
                  f"{Color.RST.value}{error}")
            sys.exit(1)

        search_results: MinimalSearchResults = MinimalSearchResults(
            question_id=query.question_id,
            question=query.question,
            retrieved_sources=build_minimal_sources(list_dict_struct)
        )

        lst_search_results.append(search_results)
    student_search_results: StudentSearchResults = StudentSearchResults(
        search_results=lst_search_results,
        k=k
    )

    file_path: Path = Path(save_directory) / Path(dataset_path).name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        student_search_results.model_dump_json(indent=2),
        encoding="utf-8"
    )


def answer(query: str,
           k: int = 5,
           processed_directory: str = "data/processed",
           test: bool = False,
           llm: str = "Qwen/Qwen3-0.6B",) -> str:

    top_sourcer: list[MinimalSource] = start_search(query,
                                                    k,
                                                    processed_directory,
                                                    test)

    str_source: str = build_context(top_sourcer)
    tokenizer: PreTrainedTokenizerBase
    modele: PreTrainedModel
    tokenizer, modele = load_model(llm)
    answer_llm: str = generate_answer(query, str_source, tokenizer, modele)

    return answer_llm


def answer_dataset(
    results_file: str = "data/output/search_results/UnansweredQuestions/dataset_code_public.json",
    save_directory: str = (
        "data/output/search_results_and_answer/UnansweredQuestions"
    ),
    llm: str = "Qwen/Qwen3-0.6B") -> StudentSearchResultsAndAnswer:

    try:
        with open(results_file, "r", encoding="utf-8") as c:
            json_content: dict[str, list[dict[str, str]]] = json.load(c)

    except FileNotFoundError:
        raise ValueError(f"{Color.MAGENTA.value}The file ["
                         f"{Color.CYAN.value}{results_file}"
                         f"{Color.MAGENTA.value}{Color.CYAN.value}] cannot "
                         f"be found.{Color.RST.value}")

    except json.JSONDecodeError:
        raise ValueError(f"{Color.MAGENTA.value}The file ["
                         f"{Color.CYAN.value}{results_file}"
                         f"{Color.MAGENTA.value}{Color.CYAN.value}] content "
                         f"is not valid JSON.{Color.RST.value}")

    object:StudentSearchResults = (
        StudentSearchResults.model_validate_json(json_content))


def start_search(
    query: str,
    k: int = 5,
    processed_directory: str = "data/processed",
    test: bool = False
) -> list[MinimalSource]:

    directory_info_index: str = str(Path(processed_directory) / "info")
    file_info_index: str = "chunk_info.json"
    directory_index: str = str(Path(processed_directory) / "bm25")

    results: list[dict[str, str | int | float]] = run_search(
        directory_info_index,
        file_info_index,
        directory_index,
        query,
        k,
        test
    )

    return build_minimal_sources(results)


def start_indexation(
    max_chunk_size: int = 2000,
    chunk_overlap_percent: int = 15,
    source_path: str = "data/raw/vllm-0.10.1",
    processed_directory: str = "data/processed"
) -> None:
    """
    Build and save the BM25 index from the vLLM source files.

    Args:
        max_chunk_size: Maximum number of characters per chunk.
        chunk_overlap_percent: Percentage of overlap between chunks.
    """
    path: Path = Path(source_path)
    directory_info_index: str = str(Path(processed_directory) / "info")
    file_info_index: str = "chunk_info.json"
    directory_index: str = str(Path(processed_directory) / "bm25")

    run_indexation(path, max_chunk_size, chunk_overlap_percent,
                   directory_info_index, file_info_index,
                   directory_index)
