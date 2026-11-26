"""Read source files and validate question datasets."""

from os.path import relpath
from pathlib import Path
from typing import Literal
from src.utils import Color
from src.models import (SourceFile, RagDataset)
import json


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def get_file_type(path: Path) -> Literal['python', 'text'] | None:
    """Return the supported file type.

    Args:
        path: File path whose suffix determines the chunking strategy.

    Returns:
        "python" for .py files, "text" for .md or .txt files, or None for
        unsupported suffixes.
    """

    if path.suffix == ".py":
        return "python"

    if path.suffix in {".md", ".txt"}:
        return "text"

    return None


def parse_dataset(file: str) -> RagDataset:
    """Load a question dataset from JSON and validate its structure.

    Args:
        file: Path to the dataset JSON file.

    Returns:
        Validated dataset containing answered or unanswered questions.

    Raises:
        ValueError: If the file is missing or its JSON or schema is invalid.
        OSError: If the file cannot be read for another filesystem reason.
    """

    try:
        with open(file, "r", encoding="utf-8") as content:
            json_content: dict[str, list[dict[str, str]]] = json.load(content)

    except FileNotFoundError:
        raise ValueError(f"{Color.MAGENTA.value}The file ["
                         f"{Color.CYAN.value}{file}{Color.MAGENTA.value}"
                         f"{Color.CYAN.value}] cannot be found."
                         f"{Color.RST.value}")

    except json.JSONDecodeError:
        raise ValueError(f"{Color.MAGENTA.value}The file ["
                         f"{Color.CYAN.value}{file}{Color.MAGENTA.value}"
                         f"{Color.CYAN.value}] content is not valid JSON."
                         f"{Color.RST.value}")

    return RagDataset.model_validate(json_content)


def parse_file(path: Path) -> SourceFile | None:
    """Read one supported source file and store its project-relative path.

    Args:
        path: Absolute or relative path to the source file.

    Returns:
        Source file model, or None when the suffix is unsupported, the file
        cannot be read, or its contents are not valid UTF-8 text.
    """

    file_type: Literal['python', 'text'] | None = get_file_type(path)

    if file_type is None:
        return None

    try:
        content: str = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    file_path: str = Path(relpath(path, PROJECT_ROOT)).as_posix()

    return SourceFile(
        file_path=file_path,
        content=content,
        file_type=file_type,
    )


def parse_directory(directory: str) -> list[SourceFile]:
    """Read all supported files from a directory.

    Args:
        directory: Root directory to traverse recursively.

    Returns:
        Readable Python, Markdown, and text files with project-relative paths.

    Raises:
        NotADirectoryError: If the given path is not an existing directory.
    """

    source_files: list[SourceFile] = []
    root = Path(directory)

    if not root.is_dir():
        raise NotADirectoryError(f"{Color.MAGENTA.value}The path["
                                 f"{Color.CYAN.value}{root}"
                                 f"{Color.MAGENTA.value}] "
                                 f"is not a valid directory.{Color.RST.value}")

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        source_file: SourceFile | None = parse_file(path)

        if source_file is not None:
            source_files.append(source_file)

    return source_files
