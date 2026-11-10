from src.models import ChunkModel, SourceFile
from src.parse import parse_directory
from src.chunker import files_distribution
from src.indexer import create_corpus, build_index, save_index
from src.retriever import load_index, search
from src.tests import test_check_chunk_positions
from src.utils import Color
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
               directory_index: str, query: str, k: int
               ) -> list[dict[str, str | int | float]]:

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
        list_dict_struct: list[dict[str, str | int | float]] = []
        list_dict_struct = search(chunk_info, query, retriever, k)
    except Exception as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(search_index) "
              f"{Color.RST.value}{error}")
        sys.exit(1)

    if test_check_chunk_positions(list_dict_struct):
        print(f"{Color.GREEN.value}[VALID] {Color.BLUE.value}The retriever "
              "returns correct paths and positions."
              f"{Color.RST.value}")
    else:
        print(f"{Color.RED.value}[NOT VALID] {Color.BLUE.value}The retriever "
              f"does not correctly retrace paths or return to positions."
              f"{Color.RST.value}")
        sys.exit(1)

    return list_dict_struct


def start_search(query: str, k: int = 5) -> list[dict[str, str | int | float]]:

    directory_info_index: str = "data/index/info"
    file_info_index: str = "chunk_info.json"
    directory_index: str = "data/index/bm25"

    return run_search(
        directory_info_index,
        file_info_index,
        directory_index,
        query,
        k
    )


def start_indexation(max_chunk_size: int = 2000,
                     chunk_overlap_percent: int = 15) -> None:
    """
    Build and save the BM25 index from the vLLM source files.

    Args:
        max_chunk_size: Maximum number of characters per chunk.
        chunk_overlap_percent: Percentage of overlap between chunks.
    """
    directory_info_index: str = "data/index/info"
    file_info_index: str = "chunk_info.json"
    directory_index: str = "data/index/bm25"
    path = Path("./data/raw/vllm-0.10.1")

    run_indexation(path, max_chunk_size, chunk_overlap_percent,
                   directory_info_index, file_info_index,
                   directory_index)
