"""Provide terminal colors and conversion to minimal source models."""

from enum import Enum
from src.models import MinimalSource


class Color(Enum):
    """Represent ANSI color escape codes for terminal messages."""

    BLUE = "\033[1;94m"
    ORANGE = "\033[38;5;208m"
    RED = "\033[1;91m"
    WHITE = "\033[1;97m"
    YELLOW = "\033[1;93m"
    GREEN = "\033[1;92m"
    MAGENTA = "\033[1;95m"
    CYAN = "\033[1;96m"
    RST = "\033[0m"


def build_minimal_sources(results: list[dict[str, str | int | float]]
                          ) -> list[MinimalSource]:
    """Convert retrieval dictionaries into validated source locations.

    Args:
        results: Ranked dictionaries with path, first_index, and last_index.

    Returns:
        MinimalSource models in the same order as the input results.

    Raises:
        KeyError: If a required dictionary key is missing.
        ValueError: If Pydantic rejects a source field.
    """

    lst_objects: list[MinimalSource] = []

    for data in results:

        object: MinimalSource = MinimalSource.model_validate({
            "file_path": data["path"],
            "first_character_index": data["first_index"],
            "last_character_index": data["last_index"],
        })
        lst_objects.append(object)

    return lst_objects
