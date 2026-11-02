from src.models import SourceFile
from src.parse import parse_directory
from pathlib import Path
import sys


def main() -> None:
    path = Path("./data/raw/vllm-0.10.1")

    try:
        files: list[SourceFile] = parse_directory(str(path))
        print(f"Fichiers récupérés : {len(files)}")
    except OSError as error:
        print(f"[ERROR]: {error}")
        sys.exit(1)


if __name__ == "__main__":

    try:
        main()
    except Exception as error:
        print(f"[ERROR]: {error}")
        sys.exit(1)
