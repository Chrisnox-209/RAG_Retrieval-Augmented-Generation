from typing import List
from pathlib import Path
from bm25s.tokenization import Tokenized
from src.models import ChunkModel
from src.utils import Color
import bm25s
import json


def create_corpus(list_chunk: list[ChunkModel]
                  ) -> tuple[List[str], list[dict[str, str | int]]]:
    chunk_info: list[dict[str, str | int]] = []
    corpus: list[str] = []
    for chunk in list_chunk:
        info: dict[str, str | int] = {}
        corpus.append(chunk.content)
        info["path"] = chunk.file_path
        info["first_index"] = chunk.first_character_index
        info["last_index"] = chunk.last_character_index
        chunk_info.append(info)

    if len(corpus) != len(chunk_info):
        raise ValueError(f"{Color.MAGENTA.value}The number of chunk_info"
                         f"[{Color.CYAN.value}{len(chunk_info)}"
                         f"{Color.MAGENTA.value}] does not correspond "
                         "to the generated chunk"
                         f"[{Color.CYAN.value}{len(corpus)}"
                         f"{Color.MAGENTA.value}].{Color.RST.value}")

    return corpus, chunk_info


def build_index(corpus: list[str]) -> bm25s.BM25:
    corpus_tokens: List[List[str]] | Tokenized = bm25s.tokenize(corpus)
    retriever: bm25s.BM25 = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens)

    return retriever


def save_index(directory_info: str,
               directory_index: str,
               file_info_index: str,
               retriever: bm25s.BM25,
               corpus: list[str],
               chunk_info: list[dict[str, str | int]]) -> None:

    root_info = Path(directory_info)
    path_file_info: Path = root_info / file_info_index
    root_index = Path(directory_index)

    try:
        root_index.mkdir(parents=True, exist_ok=True)
        retriever.save(root_index, corpus=corpus)
    except OSError as error:
        raise OSError(
            f"{Color.MAGENTA.value}Failed to save data to "
            f"{Color.CYAN.value}[{root_index}]\n"
            f"{Color.ORANGE.value}{error}{Color.RST.value}"
        ) from error

    try:
        root_info.mkdir(parents=True, exist_ok=True)
        with open(path_file_info, "w", encoding="utf-8") as f:
            json.dump(chunk_info, f, indent=4, ensure_ascii=False)
    except OSError as error:
        raise OSError(
            f"{Color.MAGENTA.value}Failed to save data to "
            f"{Color.CYAN.value}[{path_file_info}]\n"
            f"{Color.ORANGE.value}{error}{Color.RST.value}"
        ) from error
