from typing import List
from src.models import ChunkModel
from bm25s.tokenization import Tokenized
import bm25s


def create_corpus(list_chunk: list[ChunkModel]) -> List[str]:
    corpus: list[str] = []
    for chunk in list_chunk:
        corpus.append(chunk.content)

    return corpus


def build_index(corpus: list[str]) -> bm25s.BM25:
    corpus_tokens: List[List[str]] | Tokenized = bm25s.tokenize(corpus)
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens)

    return retriever


def save_index(retriever: bm25s.BM25, corpus: list[str]) -> None:
    retriever.save("data/index/bm25", corpus=corpus)
