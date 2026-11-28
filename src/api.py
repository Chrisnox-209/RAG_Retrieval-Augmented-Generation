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
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1)

    @field_validator("query")
    @classmethod
    def check_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("The query must not be blank.")
        return value


class AnswerResponse(BaseModel):
    answer: str
    sources: list[MinimalSource]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search")
def search_endpoint(request: SearchRequest) -> list[MinimalSource]:
    file_info: str = "data/processed/info/chunk_info.json"
    directory_index: str = "data/processed/bm25"

    retriever, chunk_info = load_index(file_info, directory_index)

    results: list[dict[str, str | int | float]] = search(
        chunk_info, request.query, retriever, request.k
    )

    return build_minimal_sources(results)


@app.post("/answer")
def answer_endpoint(request: SearchRequest) -> AnswerResponse:
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
