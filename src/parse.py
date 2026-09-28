from pathlib import Path
from src.models import SourceFile


def get_file_type(path: Path) -> str | None:
    """Return the supported file type."""

    if path.suffix == ".py":
        return "python"

    if path.suffix in {".md", ".txt"}:
        return "text"

    return None


def parse_file(path: Path) -> SourceFile | None:
    """Read one supported source file."""

    file_type: str | None = get_file_type(path)

    if file_type is None:
        return None

    try:
        content: str = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    return SourceFile(
        file_path=str(path),
        content=content,
        file_type=file_type,
    )


def parse_directory(directory: str) -> list[SourceFile]:
    """Read all supported files from a directory."""

    source_files: list[SourceFile] = []
    root = Path(directory)

    if not root.exists():
        return source_files

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        source_file = parse_file(path)

        if source_file is not None:
            source_files.append(source_file)

    return source_files
