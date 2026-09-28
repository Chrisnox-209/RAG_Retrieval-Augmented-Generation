from typing import Literal
from pydantic import BaseModel


class SourceFile(BaseModel):
    """Represent a source file loaded from the corpus."""

    file_path: str
    content: str
    file_type: Literal["python", "text"]