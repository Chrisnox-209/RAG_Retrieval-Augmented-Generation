"""Define Pydantic models for sources, questions, and pipeline results."""

from typing import Literal, List
from pydantic import BaseModel, Field
import uuid


class SourceFile(BaseModel):
    """Represent a source file loaded from the corpus."""

    file_path: str
    content: str
    file_type: Literal["python", "text"]


class ChunkModel(BaseModel):
    """Store a chunk and its character range in the original file.

    Attributes:
        file_path: Path to the source file relative to the project root.
        content: Text of the chunk.
        first_character_index: Inclusive start position in the source text.
        last_character_index: Exclusive end position in the source text.
    """

    file_path: str
    content: str
    first_character_index: int
    last_character_index: int


class MinimalSource(BaseModel):
    """Identify a source using its file path and character range.

    Attributes:
        file_path: Path to the source file relative to the project root.
        first_character_index: Inclusive start position in the source text.
        last_character_index: Exclusive end position in the source text.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Store a question and its identifier.

    Attributes:
        question_id: Question identifier, generated as a UUID if omitted.
        question: Natural-language question to answer.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Extend a question with its reference answer and supporting sources.

    Attributes:
        sources: Reference source locations for the question.
        answer: Reference answer text.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Group answered or unanswered questions into a dataset.

    Attributes:
        rag_questions: Questions to retrieve, answer, or use as references.
    """

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Store the ranked source locations retrieved for one question.

    Attributes:
        question_id: Identifier of the input question.
        question: Text of the input question.
        retrieved_sources: Source locations in retrieval order.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Extend one question's search results with a generated answer.

    Attributes:
        answer: Answer generated from the retrieved context.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Group search results in the JSON format used for evaluation.

    Attributes:
        search_results: Search results for each processed question.
        k: Number of sources requested per question.
    """

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Group search results and generated answers for JSON output.

    Attributes:
        search_results: Retrieved sources and answers for each question.
        k: Number of sources requested per question.
    """

    search_results: List[MinimalAnswer]
    k: int
