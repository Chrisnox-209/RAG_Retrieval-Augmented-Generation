from src.models import ChunkModel, SourceFile
from src.parse import parse_directory
from src.chunker import files_distribution
from src.indexer import create_corpus, build_index, save_index
from src.utils import Color
from pathlib import Path
from bm25s import BM25
import sys


def main() -> None:
    chunk_size: int = 2000
    chunk_overlap_percent: int = 15
    directory_info_index = "data/index/info"
    directory_index = "data/ko/bm25"

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
                   retriever, corpus, chunk_info)
    except Exception as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(build_index) "
              f"{Color.RST.value}{error}")
        sys.exit(1)


if __name__ == "__main__":

    try:
        main()
    except Exception as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(unexpected) "
              f"{Color.MAGENTA.value}{error}"
              f"{Color.RST.value}")
        sys.exit(1)
