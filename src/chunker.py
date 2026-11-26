"""Split Python and text files into chunks with original character offsets."""

from langchain_core.documents.base import Document
from langchain_text_splitters import (MarkdownTextSplitter,
                                      PythonCodeTextSplitter)
from src.models import SourceFile, ChunkModel
from src.utils import Color
from tqdm import tqdm
from typing import Any


def chunk_markdown(chunk_size: int, chunk_overlap_percent: int,
                   data: str, path: str) -> list[ChunkModel]:
    """Split Markdown or text while preserving its character positions.

    Args:
        chunk_size: Maximum chunk length, from 1 to 2000 characters.
        chunk_overlap_percent: Desired overlap percentage, from 0 to 99.
        data: Full source text to split.
        path: Source file path to attach to each chunk.

    Returns:
        Chunks with inclusive start and exclusive end offsets, or an empty
        list when the input text is empty.

    Raises:
        ValueError: If the size, overlap, or resulting offsets are invalid.
    """

    if chunk_overlap_percent < 0 or chunk_overlap_percent >= 100:
        raise ValueError(f"{Color.MAGENTA.value}The chunk_overlap_percent must"
                         f" be between [{Color.CYAN.value}0"
                         f"{Color.MAGENTA.value}] and [{Color.CYAN.value}100"
                         f"{Color.MAGENTA.value}].{Color.RST.value}")
    if chunk_size < 1 or chunk_size > 2000:
        raise ValueError(f"{Color.MAGENTA.value}The chunk_size must be between"
                         f" [{Color.CYAN.value}1"
                         f"{Color.MAGENTA.value}] and [{Color.CYAN.value}2000"
                         f"{Color.MAGENTA.value}].{Color.RST.value}")

    if data == "":
        return []

    chunk_overlap = int(chunk_size * chunk_overlap_percent / 100)

    splitter: Any = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=False,
        keep_separator=True,
    )

    documents: list[Document] = splitter.create_documents([data])

    list_chunk: list[ChunkModel] = []

    for document in documents:
        start: int = document.metadata["start_index"]
        end: int = start + len(document.page_content)

        if start < 0 or data[start:end] != document.page_content:
            raise ValueError(f"{Color.MAGENTA.value}Chunk does not match"
                             f" the original text.{Color.RST.value}")

        if not 0 <= start < end <= len(data):
            raise ValueError(f"{Color.MAGENTA.value}Invalid chunk positions."
                             f"{Color.RST.value}")
        chunk = ChunkModel(
            file_path=path,
            content=data[start:end],
            first_character_index=start,
            last_character_index=end)

        list_chunk.append(chunk)

    return list_chunk


def chunk_python(chunk_size: int, chunk_overlap_percent: int,
                 data: str, path: str) -> list[ChunkModel]:
    """Split Python code using separators suited to its structure.

    Args:
        chunk_size: Maximum chunk length, from 1 to 2000 characters.
        chunk_overlap_percent: Desired overlap percentage, from 0 to 99.
        data: Full Python source text to split.
        path: Source file path to attach to each chunk.

    Returns:
        Chunks with inclusive start and exclusive end offsets, or an empty
        list when the input text is empty.

    Raises:
        ValueError: If the size, overlap, or resulting offsets are invalid.
    """

    if chunk_overlap_percent < 0 or chunk_overlap_percent >= 100:
        raise ValueError(f"{Color.MAGENTA.value}The chunk_overlap_percent must"
                         f" be between [{Color.CYAN.value}0"
                         f"{Color.MAGENTA.value}] and [{Color.CYAN.value}100"
                         f"{Color.MAGENTA.value}].{Color.RST.value}")
    if chunk_size < 1 or chunk_size > 2000:
        raise ValueError(f"{Color.MAGENTA.value}The chunk_size must be between"
                         f" [{Color.CYAN.value}1"
                         f"{Color.MAGENTA.value}] and [{Color.CYAN.value}2000"
                         f"{Color.MAGENTA.value}].{Color.RST.value}")
    if data == "":
        return []

    chunk_overlap = int(chunk_size * chunk_overlap_percent / 100)

    splitter: Any = PythonCodeTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=False,
        keep_separator=True,
    )
    documents: list[Document] = splitter.create_documents([data])

    list_chunk: list[ChunkModel] = []

    for document in documents:
        start: int = document.metadata["start_index"]
        end: int = start + len(document.page_content)

        if start < 0 or data[start:end] != document.page_content:
            raise ValueError(f"{Color.MAGENTA.value}Chunk does not match the "
                             f"original text.{Color.RST.value}")

        if not 0 <= start < end <= len(data):
            raise ValueError(f"{Color.MAGENTA.value}Invalid chunk positions."
                             f"{Color.RST.value}")
        chunk = ChunkModel(
            file_path=path,
            content=data[start:end],
            first_character_index=start,
            last_character_index=end)

        list_chunk.append(chunk)

    return list_chunk


def chunk_file(file_type: str,
               file_path: str,
               content: str,
               chunk_size: int,
               chunk_overlap_percent: int) -> list[ChunkModel]:
    """Select a chunking strategy according to the source file type.

    Args:
        file_type: Strategy to use, either "python" or "text".
        file_path: Source file path to attach to the chunks.
        content: Full source text to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap_percent: Desired overlap percentage between chunks.

    Returns:
        Chunks produced by the selected strategy, or an empty list for an
        unsupported file type.

    Raises:
        ValueError: If the selected strategy rejects the chunk parameters
            or the resulting offsets.
    """

    list_chunks: list[ChunkModel] = []
    result: list[ChunkModel] = []

    if file_type == "python":
        result = chunk_python(chunk_size,
                              chunk_overlap_percent,
                              content, file_path)
        list_chunks.extend(result)

    if file_type == "text":
        result = chunk_markdown(chunk_size,
                                chunk_overlap_percent,
                                content, file_path)
        list_chunks.extend(result)

    return list_chunks


def files_distribution(files: list[SourceFile], chunk_size: int,
                       chunk_overlap_percent: int) -> list[ChunkModel]:
    """Chunk all source files and display progress.

    Args:
        files: Parsed source files to process.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap_percent: Desired overlap percentage between chunks.

    Returns:
        Chunks from all files in their processing order.

    Raises:
        ValueError: If a chunking strategy rejects the parameters or offsets.
    """

    list_chunk: list[ChunkModel] = []
    for objet in tqdm(files, desc="Chunking", unit="file", colour="blue"):
        result: list[ChunkModel] = chunk_file(objet.file_type,
                                              objet.file_path,
                                              objet.content,
                                              chunk_size,
                                              chunk_overlap_percent)

        list_chunk.extend(result)

    return list_chunk
