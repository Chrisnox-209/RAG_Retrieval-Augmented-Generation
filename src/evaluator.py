from src.models import (StudentSearchResults, RagDataset,
                        AnsweredQuestion, MinimalSource)
from src.utils import Color
from typing import Any, Literal
from pathlib import Path
import json


def calculate_iou(start_dataset: int, end_dataset: int,
                  start_result: int, end_result: int) -> float:

    intersection_start: int = max(start_dataset, start_result)
    intersection_end: int = min(end_dataset, end_result)
    intersection: int = intersection_end - intersection_start

    if intersection <= 0:
        return 0.0

    size_a: int = end_dataset - start_dataset
    size_b: int = end_result - start_result
    union: int = size_a + size_b - intersection

    iou: float = intersection / union
    return iou


def check_query(query_result: str,
                dataset: RagDataset) -> str | Literal[False]:
    for data in dataset.rag_questions:
        if query_result == data.question_id:
            return data.question_id
    return False


def check_path(question_id: str, results_sources: list[MinimalSource],
               dataset: RagDataset) -> tuple[int, int]:

    i: int = 0
    j: int = 0

    for data in dataset.rag_questions:
        if (data.question_id == question_id
           and isinstance(data, AnsweredQuestion)):

            for data_set in data.sources:

                for result in results_sources:

                    if result.file_path == data_set.file_path:
                        iou: float = calculate_iou(
                            data_set.first_character_index,
                            data_set.last_character_index,
                            result.first_character_index,
                            result.last_character_index,)

                        if iou >= 0.05:
                            i += 1
                            break

                j += 1

    return i, j


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

    for data in dataset.rag_questions:
        if not isinstance(data, AnsweredQuestion):
            raise ValueError(
                f"{Color.MAGENTA.value}The file ["
                f"{Color.CYAN.value}{dataset_path}"
                f"{Color.MAGENTA.value}] is not of the ["
                f"{Color.CYAN.value}AnsweredQuestion"
                f"{Color.MAGENTA.value}] type{Color.RST.value}")

    score: float = 0
    total = 0
    point: int
    j: int
    for results in student_results.search_results:
        id: str | Literal[False] = check_query(results.question_id, dataset)
        if id is not False:
            if results.question_id == id:
                point = 0
                j = 0
                point, j = check_path(results.question_id,
                                      results.retrieved_sources,
                                      dataset)

                if j > 0:
                    moy: float = point / j
                    score += moy
                    total += 1
                else:
                    moy = 0

    if total > 0:
        last_total: float = (score * 100) / total
    else:
        last_total = 0

    number_results: int = len(student_results.search_results)
    number_questions: int = len(dataset.rag_questions)
    number_skipped: int = number_results - total

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║                RAG EVALUATION                ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    print(
        f"  Results JSON           : {Path(student_search_results_path).name}")
    print(f"  Reference JSON         : {Path(dataset_path).name}")
    print()
    print(f"  Dataset questions      : {number_questions}")
    print(f"  Search results         : {number_results}")
    print(f"  Evaluated questions    : {total}")
    print(f"  Skipped questions      : {number_skipped}")
    print(f"  Top-K                  : {student_results.k}")

    print()
    print("────────────────────────────────────────────────")

    if total > 0:
        print(f"  Recall@{student_results.k:<2}              : "
              f"{last_total / 100:.3f} ({last_total:.2f}%)")
    else:
        print("  Recall                  : N/A")

    print("────────────────────────────────────────────────")
    print()
