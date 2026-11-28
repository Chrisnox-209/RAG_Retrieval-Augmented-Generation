"""Expose health, source search, and answer generation through FastAPI."""

from src.generator import build_context, load_model, generate_answer
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI
from src.models import MinimalSource
from src.retriever import load_index, search
from src.utils import build_minimal_sources


app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API",
    version="1.0.0"
)


class SearchRequest(BaseModel):
    """Validate the query and result count supplied to the API.

    Attributes:
        query: Search text containing at least one non-whitespace character.
        k: Number of sources to retrieve, with a minimum of one.
    """

    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1)

    @field_validator("query")
    @classmethod
    def check_query(cls, value: str) -> str:
        """Reject queries that contain only whitespace.

        Args:
            value: Query text received for validation.

        Returns:
            Original query text when it contains a non-whitespace character.

        Raises:
            ValueError: If the query is empty or contains only whitespace.
        """

        if not value.strip():
            raise ValueError("The query must not be blank.")
        return value


class AnswerResponse(BaseModel):
    """Group a generated answer with its retrieved source locations.

    Attributes:
        answer: Answer text generated from the retrieved context.
        sources: Source locations used to build the context.
    """

    answer: str
    sources: list[MinimalSource]


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API is responding.

    Returns:
        Dictionary with the status set to "ok".
    """

    return {"status": "ok"}


@app.post("/search")
def search_endpoint(request: SearchRequest) -> list[MinimalSource]:
    """Load the saved index and retrieve sources for one query.

    Args:
        request: Validated query and requested number of results.

    Returns:
        Source locations in BM25 ranking order.

    Raises:
        OSError: If an index or metadata file cannot be read.
        ValueError: If the saved index is invalid, k exceeds the corpus size,
            or a retrieved document ID is invalid.
    """

    file_info: str = "data/processed/info/chunk_info.json"
    directory_index: str = "data/processed/bm25"

    retriever, chunk_info = load_index(file_info, directory_index)

    results: list[dict[str, str | int | float]] = search(
        chunk_info, request.query, retriever, request.k
    )

    return build_minimal_sources(results)


@app.post("/answer")
def answer_endpoint(request: SearchRequest) -> AnswerResponse:
    """Retrieve sources and generate an answer with Qwen/Qwen3-0.6B.

    Load the saved index and the model for each request, then build the
    context from the retrieved source slices.

    Args:
        request: Validated question and number of sources to retrieve.

    Returns:
        Generated answer text and the source locations used for its context.

    Raises:
        OSError: If index metadata or source files cannot be read.
        ValueError: If retrieval parameters or sources are invalid, or the
            prompt and reserved answer exceed the model's context limit.
    """

    sources: list[MinimalSource] = search_endpoint(request)

    context: str = build_context(sources)

    tokenizer, modele = load_model("Qwen/Qwen3-0.6B")

    answer_llm: str = generate_answer(
        request.query, context, tokenizer, modele
    )

    return AnswerResponse(
        answer=answer_llm,
        sources=sources
    )
