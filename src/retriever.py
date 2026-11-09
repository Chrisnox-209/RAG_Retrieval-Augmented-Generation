from bm25s.tokenization import Tokenized
import numpy as np
from numpy.typing import NDArray
from src.utils import Color
from typing import List
import json
import bm25s


def load_index(file_info: str, directory_index: str
               ) -> tuple[bm25s.BM25, list[dict[str, str | int]]]:
    retriever: bm25s.BM25 = bm25s.BM25.load(
        directory_index,
        load_corpus=True
    )

    with open(file_info, encoding="utf-8") as file:
        chunk_info: list[dict[str, str | int]] = json.load(file)

    if retriever.corpus is None:
        raise ValueError(f"{Color.MAGENTA.value}Corpus not loaded"
                         f"{Color.RST.value}")

    if len(retriever.corpus) != len(chunk_info):
        raise ValueError(f"{Color.MAGENTA.value}Corpus and metadata count"
                         f"mismatch{Color.RST.value}")

    return retriever, chunk_info


def search(chunk_info: list[dict[str, str | int]], query: str,
           retriever: bm25s.BM25, k: int = 5
           ) -> list[dict[str, str | int | float]]:

    if query.strip() == "":
        raise ValueError(f"{Color.MAGENTA.value}No request has been entered."
                         f"{Color.RST.value}")

    if k <= 0 or k > len(chunk_info):
        raise ValueError(f"{Color.MAGENTA.value}The value of k["
                         f"{Color.CYAN.value}{k}{Color.MAGENTA.value}] "
                         f"is not valid.{Color.RST.value}")

    query_tokens: List[List[str]] | Tokenized = bm25s.tokenize(query)
    docs: NDArray[np.object_]
    scores: NDArray[np.float32]
    docs, scores = retriever.retrieve(query_tokens, k=k)

    list_dict_struct: list[dict[str, str | int | float]] = []
    for lst in docs:
        for i, dct in enumerate(lst):

            dict_struct: dict[str, str | int | float] = {}

            dict_struct["text"] = dct["text"]
            dict_struct["score"] = float(scores[0][i])

            if dct["id"] <= 0 or dct["id"] >= len(chunk_info):
                raise ValueError(f"{Color.MAGENTA.value}Invalid BM25 document "
                                 f"ID [{Color.CYAN.value}{dct['id']}"
                                 f"{Color.MAGENTA.value}]. "
                                 "Expected an ID between 0 and "
                                 f"{len(chunk_info) - 1}.{Color.RST.value}")

            for key, values in chunk_info[dct["id"]].items():
                if key == "path":
                    dict_struct["path"] = values
                if key == "first_index":
                    dict_struct["first_index"] = values
                if key == "last_index":
                    dict_struct["last_index"] = values
            list_dict_struct.append(dict_struct)

    return list_dict_struct
