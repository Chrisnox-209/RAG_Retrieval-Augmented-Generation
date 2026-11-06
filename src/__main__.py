from src.models import ChunkModel, SourceFile
from src.parse import parse_directory
from src.chunker import files_distribution
from pathlib import Path
import sys


def main() -> None:
    chunk_size: int = 200
    chunk_overlap_percent: int = 15
    list_chunk: list[ChunkModel] = []
    path = Path("./data/raw/vllm-0.10.1")

    try:
        files: list[SourceFile] = parse_directory(str(path))
        print(f"Fichiers récupérés : {len(files)}")
    except OSError as error:
        print(f"[ERROR]: {error}")
        sys.exit(1)

    list_chunk = files_distribution(files, chunk_size, chunk_overlap_percent)
    print(f"Fichiers récupérés : {len(files)}")
    print(f"Chunks générés : {len(list_chunk)}")
    if list_chunk:
        print(list_chunk[0])


if __name__ == "__main__":

    try:
        main()
    except Exception as error:
        print(f"[ERROR]: {error}")
        sys.exit(1)
