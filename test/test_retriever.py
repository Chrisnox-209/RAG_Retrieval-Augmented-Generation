from src.retriever import search
from src.utils import Color
from bm25s.tokenization import Tokenized
from typing import List
import sys
import bm25s


def build_index_test(corpus: list[str], documents_build) -> bm25s.BM25:
    corpus_tokens: List[List[str]] | Tokenized = bm25s.tokenize(corpus)
    retriever: bm25s.BM25 = bm25s.BM25(corpus=documents_build)
    retriever.index(corpus_tokens)

    return retriever


documents: list[str] = [
    "Le chat dort",
    "Le chien court",
    "Le chat mange"
]

documents_build: list[dict[str, str | int]] = [
    {"id": 0, "text": "Le chat dort"},
    {"id": -1, "text": "Le chien court"},
    {"id": 2, "text": "Le chat mange"}
]

chunk_info: list[dict[str, str | int]] = [
    {"path": "chat.txt", "first_index": 0, "last_index": 12},
    {"path": "chien.txt", "first_index": 0, "last_index": 14},
    {"path": "chat_mange.txt", "first_index": 0, "last_index": 13},
    ]


retriever: bm25s.BM25 = build_index_test(documents, documents_build)

print(retriever)

try:
    result: List[dict[str, str | int | float]] = search(chunk_info,
                                                        "chien", retriever, 1)
except Exception as error:
    print(f"{Color.RED.value}[ERROR]: "
          f"{Color.YELLOW.value}(search_index) "
          f"{Color.RST.value}{error}")
    sys.exit(1)

print(result)
