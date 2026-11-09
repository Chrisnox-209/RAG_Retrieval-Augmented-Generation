from typing import Any


def test_check_chunk_positions(
        list_dict_struct: list[dict[str, str | int | float]]) -> bool:

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
