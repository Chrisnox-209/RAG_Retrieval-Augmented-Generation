from src.models import ChunkModel, SourceFile
from src.parse import parse_directory
from src.chunker import files_distribution
from src.indexer import create_corpus, build_index, save_index
from src.retriever import load_index, search
from src.utils import Color
from pathlib import Path
from bm25s import BM25
import sys


def main() -> None:
    chunk_size: int = 2000
    chunk_overlap_percent: int = 15
    directory_info_index: str = "data/index/info"
    file_info_index: str = "chunk_info.json"
    directory_index: str = "data/index/bm25"

    list_chunk: list[ChunkModel] = []
    path = Path("./data/raw/vllm-0.10.1")

    try:
        files: list[SourceFile] = parse_directory(str(path))
        print(f"Recovered files : {len(files)}")

    except OSError as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(filesystem) "
              f"{Color.RST.value}{error}")
        sys.exit(1)

    list_chunk = files_distribution(files, chunk_size, chunk_overlap_percent)

    print(f"Generated chunks : {len(list_chunk)}")

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

    try:
        file_info: str = directory_info_index + "/" + file_info_index
        retriever, chunk_info = load_index(file_info, directory_index)
    except Exception as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(reading_index) "
              f"{Color.RST.value}{error}")
        sys.exit(1)

    list_dict_struct: list[dict[str, str | int | float]] = []
    list_dict_struct = search(chunk_info, "KV cache", retriever, 5)

    print(list_dict_struct)


if __name__ == "__main__":

    try:
        main()
    except Exception as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(unexpected) "
              f"{Color.MAGENTA.value}{error}"
              f"{Color.RST.value}")
        sys.exit(1)
