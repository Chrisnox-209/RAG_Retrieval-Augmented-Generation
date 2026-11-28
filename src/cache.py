from _hashlib import HASH
from pathlib import Path
from src.utils import Color
import hashlib
import json


def generate_cache_key(query: str, k: int) -> str:
    data: dict[str, str | int] = {"query": query, "k": k}
    data_str: str = json.dumps(data, sort_keys=True)

    hash_object: HASH = hashlib.sha256(data_str.encode())
    hex_digest: str = hash_object.hexdigest()

    return hex_digest


def save_cache(cache_directory: str, cache_key: str,
               results: list[dict[str, str | int | float]]
               ) -> None:
    file_path: Path = Path(cache_directory) / f"{cache_key}.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8")


def load_cache(cache_directory: str, cache_key: str
               ) -> list[dict[str, str | int | float]] | None:
    file_path: Path = Path(cache_directory) / f"{cache_key}.json"
    try:
        with open(file_path, "r", encoding="utf-8") as c:
            json_content: list[dict[str, str | int | float]] = json.load(c)

    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None

    return json_content


def invalidate_cache(cache_directory: str, debug: bool = False) -> None:
    root = Path(cache_directory)
    count: int = 0

    if not root.exists():
        if debug:
            print(f"{Color.CYAN.value}[INFO]: "
                  f"{Color.YELLOW.value}(CACHE NOT FOUND) "
                  f"{Color.RST.value}The cache directory ["
                  f"{Color.MAGENTA.value}{cache_directory}{Color.RST.value}] "
                  f"does not exist.{Color.RST.value}")
    else:
        for file in root.glob("*.json"):
            if file.is_file():
                file.unlink()
                count += 1
        if debug:
            print(f"{Color.CYAN.value}[INFO]: "
                  f"{Color.YELLOW.value}(CACHE INVALIDATED) "
                  f"{Color.RST.value}[{Color.MAGENTA.value}"
                  f"{count}{Color.RST.value}] "
                  f"cache files have been removed.")
