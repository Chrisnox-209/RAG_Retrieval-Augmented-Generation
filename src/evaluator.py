from src.models import StudentSearchResults, RagDataset
from src.utils import Color
from typing import Any
import json


def evaluate(student_search_results_path: str, dataset_path: str) -> None:

    try:
        with open(student_search_results_path, "r", encoding="utf-8") as c:
            json_content: dict[str, Any] = json.load(c)

    except FileNotFoundError:
        raise ValueError(f"{Color.MAGENTA.value}The file ["
                         f"{Color.CYAN.value}{student_search_results_path}"
                         f"{Color.MAGENTA.value}] cannot "
                         f"be found.{Color.RST.value}")

    except json.JSONDecodeError:
        raise ValueError(f"{Color.MAGENTA.value}The file ["
                         f"{Color.CYAN.value}{student_search_results_path}"
                         f"{Color.MAGENTA.value}] content "
                         f"is not valid JSON.{Color.RST.value}")

    student_results: StudentSearchResults = (
        StudentSearchResults.model_validate(json_content))

    try:
        with open(dataset_path, "r", encoding="utf-8") as c:
            json_dataset: dict[str, Any] = json.load(c)

    except FileNotFoundError:
        raise ValueError(f"{Color.MAGENTA.value}The file ["
                         f"{Color.CYAN.value}{dataset_path}"
                         f"{Color.MAGENTA.value}] cannot "
                         f"be found.{Color.RST.value}")

    except json.JSONDecodeError:
        raise ValueError(f"{Color.MAGENTA.value}The file ["
                         f"{Color.CYAN.value}{dataset_path}"
                         f"{Color.MAGENTA.value}] content "
                         f"is not valid JSON.{Color.RST.value}")

    dataset: RagDataset = (
        RagDataset.model_validate(json_dataset))

# faire une fonction pour trouver la question 
    for results in student_results.search_results:
        for data in dataset.rag_questions:
            if results.question_id == data.question_id:
                for src_result in results.retrieved_sources:
                    print(src_result)
            
        
    
