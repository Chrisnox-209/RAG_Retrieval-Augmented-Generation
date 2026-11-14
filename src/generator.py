from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          PreTrainedTokenizerBase)
from transformers.models.auto.modeling_auto import _BaseModelWithGenerate
from src.models import MinimalSource
from src.utils import Color
from pathlib import Path


llm: str = "Qwen/Qwen3-0.6B"

tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(llm)
modele: _BaseModelWithGenerate = AutoModelForCausalLM.from_pretrained(
    llm,
    device_map="auto"
)


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
