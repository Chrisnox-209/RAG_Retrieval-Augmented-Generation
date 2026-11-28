"""Coordinate indexing, search, and answer generation for the CLI."""

from src.models import (ChunkModel, MinimalSource,
                        SourceFile, RagDataset, UnansweredQuestion,
                        MinimalSearchResults, StudentSearchResults,
                        MinimalAnswer, StudentSearchResultsAndAnswer)
from src.parse import parse_directory, parse_dataset
from src.chunker import files_distribution
from src.indexer import create_corpus, build_index, save_index
from src.retriever import load_index, search
from src.tests import test_check_chunk_positions
from src.utils import Color, build_minimal_sources
from src.generator import build_context, generate_answer, load_model
from src.cache import (generate_cache_key, save_cache,
                       load_cache, invalidate_cache)
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from typing import Any
from tqdm import tqdm
from pathlib import Path
from bm25s import BM25
import json
import sys


def run_indexation(path: Path, chunk_size: int, chunk_overlap_percent: int,
                   directory_info_index: str,
                   file_info_index: str,
                   directory_index: str,
                   debug: bool
                   ) -> tuple[BM25, list[dict[str, str | int]]]:
    """Parse source files, build a BM25 index, and save it.

    Args:
        path: Root directory containing the source corpus.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap_percent: Desired overlap percentage between chunks.
        directory_info_index: Destination directory for chunk metadata.
        file_info_index: Filename for the chunk metadata JSON.
        directory_index: Destination directory for the BM25 index.

    Returns:
        Built retriever and its chunk metadata in corpus order.

    Raises:
        SystemExit: If reading, chunking, indexing, or saving fails.
    """
    cache_directory: str = str(Path(directory_index).parent / "cache")

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
    except Exception as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(corpus) "
              f"{Color.RST.value}{error}")
        sys.exit(1)

    try:
        retriever: BM25 = build_index(corpus)
    except Exception as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(build_index) "
              f"{Color.RST.value}{error}")
        sys.exit(1)

    try:
        invalidate_cache(cache_directory, debug)
        save_index(directory_info_index, directory_index,
                   file_info_index, retriever,
                   corpus, chunk_info)
    except Exception as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(save_index) "
              f"{Color.RST.value}{error}")
        sys.exit(1)

    return retriever, chunk_info


def run_search(directory_info_index: str, file_info_index: str,
               directory_index: str, query: str, k: int,
               test: bool = False) -> list[dict[str, str | int | float]]:
    """Search a saved index and optionally check the retrieved source slices.

    Args:
        directory_info_index: Directory containing the chunk metadata.
        file_info_index: Filename of the chunk metadata JSON.
        directory_index: Directory containing the saved BM25 index.
        query: Question or search text.
        k: Number of chunks to retrieve.
        test: Whether to verify paths, bounds, and text against source files.

    Returns:
        Ranked retrieval dictionaries with text, score, path, and offsets.

    Raises:
        SystemExit: If loading or searching fails, or source checks return
            False.
    """

    retriever: BM25 | None = None
    chunk_info: list[dict[str, str | int]] = []
    list_dict_struct: list[dict[str, str | int | float]] | None
    cache_directory: str = str(Path(directory_index).parent / "cache")

    key: str = generate_cache_key(query, k)
    list_dict_struct = load_cache(cache_directory, key)

    if list_dict_struct is None:
        if test:
            print(f"{Color.CYAN.value}[INFO]: "
                  f"{Color.YELLOW.value}(CACHE MISS) "
                  f"{Color.BLUE.value}Searching BM25 index."
                  f"{Color.RST.value}")

        try:
            file_info: str = str(Path(directory_info_index) / file_info_index)
            retriever, chunk_info = load_index(file_info, directory_index)
        except Exception as error:
            print(f"{Color.RED.value}[ERROR]: "
                  f"{Color.YELLOW.value}(reading_index) "
                  f"{Color.RST.value}{error}")
            sys.exit(1)

        try:
            list_dict_struct = search(chunk_info, query, retriever, k)
        except Exception as error:
            print(f"{Color.RED.value}[ERROR]: "
                  f"{Color.YELLOW.value}(search_index) "
                  f"{Color.RST.value}{error}")
            sys.exit(1)

        try:
            save_cache(cache_directory, key, list_dict_struct)
            if test:
                print(f"{Color.CYAN.value}[INFO]: "
                      f"{Color.YELLOW.value}(CACHE SAVED) "
                      f"{Color.GREEN.value}Caching of the key ["
                      f"{Color.MAGENTA.value}{key}{Color.GREEN.value}] "
                      f"has been successfully completed.{Color.RST.value}")
        except Exception as error:
            print(f"{Color.ORANGE.value}[WARNING]: "
                  f"{Color.YELLOW.value}(save_cache) "
                  f"{Color.RST.value}{error}")

    elif test:
        print(f"{Color.CYAN.value}[INFO]: "
              f"{Color.YELLOW.value}(CACHE HIT) "
              f"{Color.BLUE.value}Results loaded from cache."
              f"{Color.RST.value}")

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
                             "data/output/search_results/UnansweredQuestions"),
                         processed_directory: str = "data/processed",
                         debug: bool = False
                         ) -> None:
    """Search every dataset question and save the results as JSON.

    Load the index once and write the output using the input dataset filename.

    Args:
        dataset_path: Path to the answered or unanswered question dataset.
        k: Number of sources to retrieve per question.
        save_directory: Directory for the search results JSON.
        processed_directory: Directory containing the saved index and metadata.

    Raises:
        ValueError: If the dataset is missing or has invalid JSON or fields.
        OSError: If the dataset cannot be read or the output cannot be written.
        SystemExit: If loading the index or searching a question fails.
    """

    dataset: RagDataset = parse_dataset(dataset_path)
    directory_info_index: str = str(Path(processed_directory) / "info")
    file_info_index: str = "chunk_info.json"
    directory_index: str = str(Path(processed_directory) / "bm25")
    cache_directory: str = str(Path(directory_index).parent / "cache")
    lst_search_results: list[MinimalSearchResults] = []
    retriever: BM25 | None = None
    chunk_info: list[dict[str, str | int]]
    list_dict_struct: list[dict[str, str | int | float]] | None

    cache_hits: int = 0
    cache_misses: int = 0

    for query in tqdm(dataset.rag_questions,
                      desc="Searching",
                      unit="question",
                      colour="green",
                      dynamic_ncols=True):
        key: str = generate_cache_key(query.question, k)
        list_dict_struct = load_cache(cache_directory, key)

        if list_dict_struct is None:
            cache_misses += 1

            if retriever is None:
                try:
                    file_info: str = str(
                        Path(directory_info_index) / file_info_index)
                    retriever, chunk_info = load_index(
                        file_info, directory_index)
                except Exception as error:
                    print(f"{Color.RED.value}[ERROR]: "
                          f"{Color.YELLOW.value}(reading_index) "
                          f"{Color.RST.value}{error}")
                    sys.exit(1)

            try:
                list_dict_struct = search(
                    chunk_info, query.question, retriever, k)
            except Exception as error:
                print(f"\n{Color.RED.value}[ERROR]: "
                      f"{Color.YELLOW.value}(search_index) "
                      f"{Color.RST.value}{error}")
                sys.exit(1)

            try:
                save_cache(cache_directory, key, list_dict_struct)
            except Exception as error:
                print(f"{Color.ORANGE.value}[WARNING]: "
                      f"{Color.YELLOW.value}(save_cache) "
                      f"{Color.RST.value}{error}")

        else:
            cache_hits += 1

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

    if debug:
        print(f"{Color.CYAN.value}[INFO]: "
              f"{Color.GREEN.value}(CACHING) "
              f"{Color.MAGENTA.value}[{Color.RST.value}{cache_hits}"
              f"{Color.MAGENTA.value}] {Color.RST.value}cached file.")

        print(f"{Color.CYAN.value}[INFO]: "
              f"{Color.YELLOW.value}(ALREADY IN CACHING): "
              f"{Color.MAGENTA.value}[{Color.RST.value}{cache_misses}"
              f"{Color.MAGENTA.value}] {Color.RST.value}files already "
              "in cache.")


def answer(query: str,
           k: int = 5,
           processed_directory: str = "data/processed",
           test: bool = False,
           llm: str = "Qwen/Qwen3-0.6B",
           save_directory: str = (
               "data/output/search_results_and_answer/SingleQuestion")
           ) -> str:
    """Generate and save the answer to one question.

    Write the answer and its sources to dataset_single_answer.json in the
    selected output directory.

    Args:
        query: Question to answer.
        k: Number of sources to retrieve for the context.
        processed_directory: Directory containing the saved index and metadata.
        test: Whether to verify retrieved source slices against the files.
        llm: Hugging Face model identifier or local model directory.
        save_directory: Directory for the answer JSON.

    Returns:
        Generated answer text for display in the terminal.

    Raises:
        SystemExit: If retrieval or answer generation fails.
        ValueError: If a source file cannot be opened at its given path.
        OSError: If source text cannot be read or the output cannot be written.
    """

    top_sources: list[MinimalSource] = search_sources(
        query, k, processed_directory, test)

    str_source: str = build_context(top_sources)
    tokenizer: PreTrainedTokenizerBase
    modele: PreTrainedModel

    tokenizer, modele = load_model(llm)

    try:
        answer_llm: str = generate_answer(query, str_source, tokenizer, modele)
    except Exception as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(Answer) "
              f"{Color.RST.value}{error}")
        sys.exit(1)

    question: UnansweredQuestion = UnansweredQuestion(question=query)
    result: MinimalAnswer = MinimalAnswer(
        question_id=question.question_id,
        question=question.question,
        retrieved_sources=top_sources,
        answer=answer_llm
    )
    student_answers: StudentSearchResultsAndAnswer = (
        StudentSearchResultsAndAnswer(search_results=[result], k=k))

    file_path: Path = Path(save_directory) / "dataset_single_answer.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        student_answers.model_dump_json(indent=2),
        encoding="utf-8"
    )
    print(f"Saved student_search_results_and_answer to {file_path}")

    return answer_llm


def answer_dataset(
        student_search_results_path: str = (
            "data/output/search_results/UnansweredQuestions/"
            "dataset_code_public.json"),
        save_directory: str = (
            "data/output/search_results_and_answer/UnansweredQuestions"
        ), llm: str = "Qwen/Qwen3-0.6B",
        verbose: bool = False) -> None:
    """Generate answers from saved search results and write them as JSON.

    Load the model once for the batch and reuse the input filename for the
    output. Preserve the question identifiers, sources, and requested k.

    Args:
        student_search_results_path: Path to a StudentSearchResults JSON file.
        save_directory: Directory for the generated answer JSON.
        llm: Hugging Face model identifier or local model directory.

    Raises:
        ValueError: If input JSON or fields are invalid, a source is
            inaccessible, or a prompt exceeds the model context limit.
        OSError: If input files cannot be read or the output cannot be written.
    """

    try:
        with open(student_search_results_path, "r", encoding="utf-8") as c:
            json_content: dict[str, Any] = json.load(c)

    except FileNotFoundError:
        raise ValueError(f"{Color.MAGENTA.value}The file ["
                         f"{Color.CYAN.value}{student_search_results_path}"
                         f"{Color.MAGENTA.value}] cannot "
                         f"be found.{Color.RST.value}")

    except json.JSONDecodeError:
        raise ValueError(f"{Color.MAGENTA.value}The file ["
                         f"{Color.CYAN.value}{student_search_results_path}"
                         f"{Color.MAGENTA.value}] content "
                         f"is not valid JSON.{Color.RST.value}")

    student_results: StudentSearchResults = (
        StudentSearchResults.model_validate(json_content))

    tokenizer: PreTrainedTokenizerBase
    modele: PreTrainedModel
    tokenizer, modele = load_model(llm)

    lst_answers: list[MinimalAnswer] = []

    for data in tqdm(student_results.search_results,
                     desc="Generating answers",
                     unit="question",
                     colour="red",
                     dynamic_ncols=True,
                     disable=verbose):

        sources: str = build_context(data.retrieved_sources)
        answer_llm: str = generate_answer(data.question,
                                          sources,
                                          tokenizer,
                                          modele)

        if verbose:
            print(f"{Color.CYAN.value}[QUESTION]: {Color.MAGENTA.value}"
                  f"{data.question}")
            print(f"{Color.BLUE.value}[LLM RESPONSE]: "
                  f"{Color.RST.value}{answer_llm}\n")

        result: MinimalAnswer = MinimalAnswer(
            question_id=data.question_id,
            question=data.question,
            retrieved_sources=data.retrieved_sources,
            answer=answer_llm
        )
        lst_answers.append(result)

    student_answers: StudentSearchResultsAndAnswer = (
        StudentSearchResultsAndAnswer(
            search_results=lst_answers,
            k=student_results.k))

    file_path: Path = (
        Path(save_directory) / Path(student_search_results_path).name)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        student_answers.model_dump_json(indent=2),
        encoding="utf-8"
    )


def search_sources(
    query: str,
    k: int,
    processed_directory: str,
    test: bool
) -> list[MinimalSource]:
    """Load the index and return the sources for one question.

    Args:
        query: Question or search text.
        k: Number of sources to retrieve.
        processed_directory: Directory containing the saved index and metadata.
        test: Whether to verify retrieved source slices against the files.

    Returns:
        Validated source locations in retrieval order.

    Raises:
        SystemExit: If index loading, retrieval, or source validation fails.
    """

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


def start_search(
    query: str,
    k: int = 5,
    processed_directory: str = "data/processed",
    test: bool = False,
    save_directory: str = "data/output/search_results/SingleQuestion",
    verbose: bool = False
) -> list[MinimalSource] | None:
    """Save the sources for one question and display them in verbose mode.

    Write dataset_single.json in the selected output directory.

    Args:
        query: Question or search text.
        k: Number of sources to retrieve.
        processed_directory: Directory containing the saved index and metadata.
        test: Whether to verify retrieved source slices against the files.
        save_directory: Directory for the search results JSON.
        verbose: Whether to return sources for display by Python Fire.

    Returns:
        Retrieved sources when verbose is True, otherwise None.

    Raises:
        SystemExit: If index loading, retrieval, or source validation fails.
        OSError: If the output directory or JSON file cannot be written.
    """

    top_sources: list[MinimalSource] = search_sources(
        query, k, processed_directory, test)
    question: UnansweredQuestion = UnansweredQuestion(question=query)
    search_results: MinimalSearchResults = MinimalSearchResults(
        question_id=question.question_id,
        question=question.question,
        retrieved_sources=top_sources
    )
    student_search_results: StudentSearchResults = StudentSearchResults(
        search_results=[search_results],
        k=k
    )

    file_path: Path = Path(save_directory) / "dataset_single.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        student_search_results.model_dump_json(indent=2),
        encoding="utf-8"
    )

    print(
        f"{Color.GREEN.value}[SUCCESS] "
        f"{Color.BLUE.value}Saved student_search_results to: "
        f"{Color.YELLOW.value}{file_path}{Color.RST.value}"
    )

    if verbose:
        return top_sources

    return None


def start_indexation(max_chunk_size: int = 2000,
                     chunk_overlap_percent: int = 15,
                     source_path: str = "data/raw/vllm-0.10.1",
                     processed_directory: str = "data/processed",
                     debug: bool = False) -> None:

    """Build and save the BM25 index from the vLLM source files.

    Args:
        max_chunk_size: Maximum number of characters per chunk.
        chunk_overlap_percent: Percentage of overlap between chunks.
        source_path: Root directory containing the source corpus.
        processed_directory: Destination directory for the index and metadata.

    Raises:
        SystemExit: If reading, chunking, indexing, or saving fails.
    """
    path: Path = Path(source_path)
    directory_info_index: str = str(Path(processed_directory) / "info")
    file_info_index: str = "chunk_info.json"
    directory_index: str = str(Path(processed_directory) / "bm25")

    run_indexation(path, max_chunk_size, chunk_overlap_percent,
                   directory_info_index, file_info_index,
                   directory_index, debug)
