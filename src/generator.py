"""Reconstruct source context and generate answers with a local model."""

from typing import Any

from transformers import (AutoTokenizer,
                          AutoModelForCausalLM, BatchEncoding,
                          PreTrainedTokenizerBase,
                          PreTrainedModel)
from src.models import MinimalSource
from torch import Tensor
from typing import cast
from src.utils import Color
from pathlib import Path


llm: str = "Qwen/Qwen3-0.6B"


def load_model(llm: str) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load a tokenizer and a causal language model.

    Use the local model cache when available and download missing files.
    Let Transformers choose the devices used to load the model.

    Args:
        llm: Hugging Face model identifier or local model directory.

    Returns:
        Tokenizer and model ready for answer generation.
    """

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(llm)
    modele: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        llm,
        device_map="auto"
    )
    return tokenizer, modele


def extract_source_text(source: MinimalSource) -> str:
    """Read the character range described by a source location.

    Args:
        source: File path and character range, with an inclusive start and
            exclusive end.

    Returns:
        Slice of the UTF-8 source text covered by the given offsets.

    Raises:
        ValueError: If the source path is missing, unreadable due to
            permissions, or points to a directory.
        OSError: If another filesystem error prevents reading the file.
        UnicodeDecodeError: If the source is not valid UTF-8 text.
    """

    path: Path = Path(source.file_path)
    first_id: int = source.first_character_index
    last_id: int = source.last_character_index

    try:
        with open(path, "r", encoding="utf-8") as content:
            data: str = content.read()
    except FileNotFoundError:
        raise ValueError(f"{Color.MAGENTA.value}The file ["
                         f"{Color.CYAN.value}{path.stem}{Color.MAGENTA.value}"
                         f"{Color.CYAN.value}] cannot be found."
                         f"{Color.RST.value}")
    except PermissionError:
        raise ValueError(f"{Color.MAGENTA.value}The file ["
                         f"{Color.CYAN.value}{path.stem}{Color.MAGENTA.value}"
                         f"{Color.CYAN.value}] permissions are not correct."
                         f"{Color.RST.value}")
    except IsADirectoryError:
        raise ValueError(f"{Color.MAGENTA.value}The file ["
                         f"{Color.CYAN.value}{path}{Color.MAGENTA.value}"
                         f"{Color.CYAN.value}] is a folder, not a file."
                         f"{Color.RST.value}")

    return data[first_id:last_id]


def build_context(sources: list[MinimalSource]) -> str:
    """Combine retrieved source slices into a numbered context.

    Args:
        sources: Source locations in the order to present to the model.

    Returns:
        Numbered excerpts separated by blank lines, or an empty string
        when no sources are provided.
    """

    str_source: str = ''

    for i, obj in enumerate(sources, start=1):
        piece_str: str = extract_source_text(obj)
        str_source += f"Source {i}:\n{piece_str}\n\n"

    return str_source


def generate_answer(
    query: str, context: str,
    tokenizer: PreTrainedTokenizerBase, modele: PreTrainedModel
) -> str:
    """Generate an answer from the question and retrieved context.

    Format a chat prompt and reserve 256 tokens for the answer. Count the
    prompt tokens before generation and stop if the model limit is exceeded.

    Args:
        query: Question to answer.
        context: Retrieved source text supplied as evidence.
        tokenizer: Tokenizer used to format and encode the prompt.
        modele: Loaded causal language model used for generation.

    Returns:
        Decoded answer text without the input prompt or special tokens.

    Raises:
        ValueError: If the prompt and reserved answer exceed the model's
            context limit.
    """

    pre_prompt: str = (
        "You are a technical assistant specialized in the vLLM codebase.\n"
        "Answer the question based on the provided context.\n"
        "Read all sources carefully before answering.\n"
        "Preserve exact function names, parameters, types and values.\n"
        "Provide a concise but complete answer in English.\n"
        "Do not guess or invent information.\n"
        "If only part of the answer is available, provide that part.\n"
        "If no relevant information is available, answer: "
        "'Insufficient information in the provided context.'"
    )

    prompt: list[dict[str, str]] = [
        {
            "role": "system",
            "content": pre_prompt
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}"
        }
    ]

    prompt_formate: str = cast(str, tokenizer.apply_chat_template(
        prompt, tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False))

    entrees: BatchEncoding[Any] = tokenizer(
        prompt_formate,
        return_tensors="pt",
        add_special_tokens=False,
        verbose=False
    )

    nb_tokens_prompt: int = entrees["input_ids"].shape[1]
    max_tokens: int = modele.config.max_position_embeddings
    max_tokens_answer: int = 256

    if nb_tokens_prompt + max_tokens_answer > max_tokens:
        raise ValueError(f"{Color.MAGENTA.value}The number of tokens ["
                         f"{Color.CYAN.value}"
                         f"{nb_tokens_prompt + max_tokens_answer}"
                         f"{Color.MAGENTA.value}] to be used is too large "
                         f"relative to the model's limit [{Color.CYAN.value}"
                         f"{max_tokens}{Color.MAGENTA.value}]"
                         f"{Color.RST.value}")

    entrees = entrees.to(modele.device)
    tokens_generates: Tensor = cast(Any, modele).generate(
        **entrees,
        max_new_tokens=max_tokens_answer,
        do_sample=False
    )
    tokens_answer: Tensor = tokens_generates[0][entrees["input_ids"].shape[1]:]
    answer_llm: str = cast(
        str, tokenizer.decode(tokens_answer, skip_special_tokens=True))

    return answer_llm
