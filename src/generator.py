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
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(llm)
    modele: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        llm,
        device_map="auto"
    )
    return tokenizer, modele


def extract_source_text(source: MinimalSource) -> str:
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
    str_source: str = ''

    for i, obj in enumerate(sources, start=1):
        piece_str: str = extract_source_text(obj)
        str_source += f"Source {i}:\n{piece_str}\n\n"

    return str_source


def generate_answer(
    query: str, context: str,
    tokenizer: PreTrainedTokenizerBase, modele: PreTrainedModel
) -> str:

    pre_prompt: str = ("You are an assistant answering questions about "
                       "the vLLM codebase. Answer using the provided context. "
                       "If the context is insufficient, say so.")

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
        add_special_tokens=False
    )
    entrees = entrees.to(modele.device)
    tokens_generates: Tensor = cast(Any, modele).generate(
        **entrees,
        max_new_tokens=256,
        do_sample=False
    )
    tokens_answer: Tensor = tokens_generates[0][entrees["input_ids"].shape[1]:]
    answer_llm: str = cast(
        str, tokenizer.decode(tokens_answer, skip_special_tokens=True))

    return answer_llm
