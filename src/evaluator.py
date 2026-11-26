"""Compute local Recall@k from retrieved and reference source locations."""

from src.models import (StudentSearchResults, RagDataset,
                        AnsweredQuestion, MinimalSource)
from src.utils import Color
from typing import Any
from pathlib import Path
import json


def calculate_iou(start_dataset: int, end_dataset: int,
                  start_result: int, end_result: int) -> float:
    """Compute intersection over union for two character ranges.

    Args:
        start_dataset: Inclusive start of the reference range.
        end_dataset: Exclusive end of the reference range.
        start_result: Inclusive start of the retrieved range.
        end_result: Exclusive end of the retrieved range.

    Returns:
        Intersection length divided by union length, or zero when the
        ranges do not overlap.
    """

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


def check_path(dataset_sources: list[MinimalSource],
               results_sources: list[MinimalSource]) -> int:
    """Count the reference sources found in the retrieved results.

    Args:
        dataset_sources: Reference source locations for one question.
        results_sources: Retrieved sources to compare with the references.

    Returns:
        Number of reference sources matched by a result in the same file
        with an intersection over union of at least 0.05. Each reference
        source counts at most once.
    """

    point: int = 0

    for data_set in dataset_sources:
        for result in results_sources:
            if result.file_path != data_set.file_path:
                continue

            iou: float = calculate_iou(
                data_set.first_character_index,
                data_set.last_character_index,
                result.first_character_index,
                result.last_character_index,
            )

            if iou >= 0.05:
                point += 1
                break

    return point


def evaluate(student_search_results_path: str, dataset_path: str) -> None:
    """Print the average Recall@k over the reference questions.

    Match questions by identifier and compare only the first k retrieved
    sources. Missing results contribute zero recall. Skip reference questions
    with no sources.

    Args:
        student_search_results_path: Path to the retrieved results JSON.
        dataset_path: Path to the dataset with reference answers and sources.

    Raises:
        ValueError: If an input is missing, has invalid JSON or fields,
            contains unanswered reference questions, or uses a negative k.
        OSError: If an input file cannot be read for another filesystem reason.
    """

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
    k: int = student_results.k

    if k < 0:
        raise ValueError(f"{Color.MAGENTA.value}The value of k["
                         f"{Color.CYAN.value}{k}{Color.MAGENTA.value}] "
                         f"is not valid.{Color.RST.value}")

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

    results_by_question: dict[str, list[MinimalSource]] = {}
    for results in student_results.search_results:
        if results.question_id not in results_by_question:
            results_by_question[results.question_id] = (
                results.retrieved_sources[:k])

    score: float = 0
    total: int = 0
    number_missing: int = 0

    for data in dataset.rag_questions:
        if not isinstance(data, AnsweredQuestion):
            raise ValueError(
                f"{Color.MAGENTA.value}The file ["
                f"{Color.CYAN.value}{dataset_path}"
                f"{Color.MAGENTA.value}] is not of the ["
                f"{Color.CYAN.value}AnsweredQuestion"
                f"{Color.MAGENTA.value}] type{Color.RST.value}")

        number_sources: int = len(data.sources)
        if number_sources == 0:
            continue

        total += 1
        results_sources: list[MinimalSource] | None = (
            results_by_question.get(data.question_id))
        if results_sources is None:
            number_missing += 1
            continue

        point: int = check_path(data.sources, results_sources)
        moy: float = point / number_sources
        score += moy

    if total > 0:
        last_total: float = (score * 100) / total
    else:
        last_total = 0

    number_results: int = len(student_results.search_results)
    number_questions: int = len(dataset.rag_questions)
    number_skipped: int = number_questions - total

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
    print(f"  Missing results        : {number_missing}")
    print(f"  Skipped questions      : {number_skipped}")
    print(f"  Top-K                  : {k}")

    print()
    print("────────────────────────────────────────────────")

    if total > 0:
        print(f"  Recall@{k:<2}              : "
              f"{last_total / 100:.3f} ({last_total:.2f}%)")
    else:
        print("  Recall                  : N/A")

    print("────────────────────────────────────────────────")
    print()
