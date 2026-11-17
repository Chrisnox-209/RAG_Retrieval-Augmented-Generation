
from src.utils import Color
from src.evaluator import evaluate
from src.cli import (start_search,
                     start_indexation,
                     start_search_dataset,
                     answer_dataset,
                     answer)
from dotenv import load_dotenv
import fire
import sys

if __name__ == "__main__":

    try:
        load_dotenv()
        fire.Fire({
            "index": start_indexation,
            "search": start_search,
            "search_dataset": start_search_dataset,
            "answer": answer,
            "answer_dataset": answer_dataset,
            "evaluate": evaluate,
            })

    except Exception as error:
        print(f"{Color.RED.value}[ERROR]: "
              f"{Color.YELLOW.value}(unexpected) "
              f"{Color.MAGENTA.value}{error}"
              f"{Color.RST.value}")
        sys.exit(1)
