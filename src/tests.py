"""Check that retrieved chunks match their original file contents."""

from typing import Any


def test_check_chunk_positions(
        list_dict_struct: list[dict[str, str | int | float]]) -> bool:
    """Check retrieved paths, character bounds, and text against source files.

    Args:
        list_dict_struct: Retrieval dictionaries containing path, first_index,
            last_index, and text.

    Returns:
        True if every source slice matches its retrieved text. Return False
        for empty results, invalid bounds, or mismatched text.

    Raises:
        OSError: If a source file cannot be read.
        UnicodeDecodeError: If a source file is not valid UTF-8 text.
    """

    if list_dict_struct == []:
        return False

    for i, dct in enumerate(list_dict_struct):

        start: Any = dct["first_index"]
        end: Any = dct["last_index"]
        path: Any = dct["path"]

        with open(path, "r", encoding="utf-8") as f:
            lignes: str = f.read()

        if (start < 0 or start >= len(lignes) or
           end <= start or end > len(lignes)):
            return False

        data: str = lignes[start:end]

        if data != dct["text"]:
            return False
    return True
