import ast
import json
import re
import time
from pathlib import Path
from typing import Any
import sys

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.evaluation.router_evaluation import DATASET_PATH, RESULTS_DIR, load_dataset
from backend.pipeline.query_pipeline import query_graph


NOT_ENOUGH_INFORMATION_PHRASES = [
    "do not have enough information",
    "don't have enough information",
    "not enough information",
    "cannot answer",
    "can't answer",
]


def normalize_text(text: str | None) -> str:
    if not text:
        return ""

    return " ".join(text.lower().replace("\n", " ").replace("\t", " ").split())


def check_expected_contains(
    answer: str,
    expected_contains: list[str] | None,
) -> float | None:
    if not expected_contains:
        return None

    answer_norm = normalize_text(answer)
    matches = 0

    for item in expected_contains:
        if normalize_text(item) in answer_norm:
            matches += 1

    return matches / len(expected_contains)


def answer_has_enough_information(answer: str) -> bool:
    answer_norm = normalize_text(answer)

    if not answer_norm:
        return False

    return not any(
        phrase in answer_norm for phrase in NOT_ENOUGH_INFORMATION_PHRASES
    )


def parse_context(context: str | None) -> dict[str, Any]:
    if not context:
        return {}

    try:
        parsed = ast.literal_eval(context)
    except (SyntaxError, ValueError):
        return {}

    if isinstance(parsed, dict):
        return parsed

    return {}


def context_has_rows(context: str | None) -> bool:
    parsed = parse_context(context)
    rows = parsed.get("rows")

    if isinstance(rows, list):
        return len(rows) > 0

    if not context:
        return False

    return bool(re.search(r"['\"]rows['\"]\s*:\s*\[(?!\])", context))


def context_present_for_route(result: dict[str, Any], route: str | None) -> bool:
    if route == "pdf":
        return bool(result.get("pdf_context"))

    if route == "sql":
        return context_has_rows(result.get("sql_context"))

    if route == "kg":
        return context_has_rows(result.get("kg_context"))

    if route == "hybrid":
        return any(
            [
                bool(result.get("pdf_context")),
                context_has_rows(result.get("sql_context")),
                context_has_rows(result.get("kg_context")),
            ]
        )

    return False


def evaluate_full_item(item: dict[str, Any]) -> dict[str, Any]:
    question_id = item.get("id")
    question = item.get("question")
    expected_route = item.get("expected_route")
    expected_contains = item.get("expected_contains")

    if expected_contains is None:
        expected_contains = item.get("expected_answer_contains")

    if not question:
        raise ValueError(f"Missing question in item: {item}")

    start_time = time.perf_counter()
    error = None

    try:
        graph_result = query_graph.invoke({"question": question})
    except Exception as exc:
        graph_result = {}
        error = str(exc)

    latency = time.perf_counter() - start_time

    actual_route = graph_result.get("route")
    answer = graph_result.get("answer", "")
    route_correct = None

    if expected_route:
        route_correct = actual_route == expected_route

    contains_score = check_expected_contains(answer, expected_contains)
    has_enough_information = answer_has_enough_information(answer)
    has_expected_context = context_present_for_route(graph_result, actual_route)

    return {
        "id": question_id,
        "question": question,
        "expected_route": expected_route,
        "actual_route": actual_route,
        "route_correct": route_correct,
        "question_type": item.get("question_type"),
        "difficulty": item.get("difficulty"),
        "latency_seconds": round(latency, 4),
        "error": error,
        "has_error": error is not None,
        "answer": answer,
        "answer_length": len(answer or ""),
        "answer_present": bool(answer),
        "answer_has_enough_information": has_enough_information,
        "expected_contains": expected_contains,
        "contains_score": contains_score,
        "has_expected_context": has_expected_context,
        "cache_hit": graph_result.get("cache_hit", False),
        "cache_type": graph_result.get("cache_type"),
        "retriever_cache_hit": graph_result.get("retriever_cache_hit", False),
        "retriever_cache_type": graph_result.get("retriever_cache_type"),
        "retriever_cache_hits": graph_result.get("retriever_cache_hits", {}),
        "pdf_context_present": bool(graph_result.get("pdf_context")),
        "sql_context_has_rows": context_has_rows(graph_result.get("sql_context")),
        "kg_context_has_rows": context_has_rows(graph_result.get("kg_context")),
    }


def summarize_results(df: pd.DataFrame) -> dict[str, Any]:
    total = len(df)
    error_rate = df["has_error"].mean() if total > 0 else 0

    route_accuracy = None
    route_df = df[df["route_correct"].notna()]

    if len(route_df) > 0:
        route_accuracy = route_df["route_correct"].mean()

    contains_score = None
    contains_df = df[df["contains_score"].notna()]

    if len(contains_df) > 0:
        contains_score = contains_df["contains_score"].mean()

    avg_latency = df["latency_seconds"].mean() if total > 0 else None
    p50_latency = df["latency_seconds"].quantile(0.50) if total > 0 else None
    p95_latency = df["latency_seconds"].quantile(0.95) if total > 0 else None

    return {
        "total_questions": total,
        "error_rate": round(float(error_rate), 4),
        "route_accuracy": round(float(route_accuracy), 4)
        if route_accuracy is not None
        else None,
        "answer_present_rate": round(float(df["answer_present"].mean()), 4)
        if total > 0
        else 0,
        "answer_enough_information_rate": round(
            float(df["answer_has_enough_information"].mean()), 4
        )
        if total > 0
        else 0,
        "context_present_rate": round(float(df["has_expected_context"].mean()), 4)
        if total > 0
        else 0,
        "avg_contains_score": round(float(contains_score), 4)
        if contains_score is not None
        else None,
        "cache_hit_rate": round(float(df["cache_hit"].mean()), 4)
        if total > 0
        else 0,
        "retriever_cache_hit_rate": round(float(df["retriever_cache_hit"].mean()), 4)
        if total > 0
        else 0,
        "avg_latency_seconds": round(float(avg_latency), 4)
        if avg_latency is not None
        else None,
        "p50_latency_seconds": round(float(p50_latency), 4)
        if p50_latency is not None
        else None,
        "p95_latency_seconds": round(float(p95_latency), 4)
        if p95_latency is not None
        else None,
    }


def run_full_evaluation(
    dataset_path: Path = DATASET_PATH,
    limit: int | None = None,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)

    if limit is not None:
        dataset = dataset[:limit]

    results = [evaluate_full_item(item) for item in dataset]
    summary = summarize_results(pd.DataFrame(results))

    return {
        "dataset_path": str(dataset_path),
        "summary": summary,
        "results": results,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n==============================")
    print("FULL E2E EVALUATION SUMMARY")
    print("==============================")
    print(f"Total questions:           {summary['total_questions']}")
    print(f"Error rate:                {summary['error_rate']}")
    print(f"Route accuracy:            {summary['route_accuracy']}")
    print(f"Answer present rate:       {summary['answer_present_rate']}")
    print(f"Enough information rate:   {summary['answer_enough_information_rate']}")
    print(f"Context present rate:      {summary['context_present_rate']}")
    print(f"Contains score:            {summary['avg_contains_score']}")
    print(f"Cache hit rate:            {summary['cache_hit_rate']}")
    print(f"Retriever cache hit rate:  {summary['retriever_cache_hit_rate']}")
    print(f"Average latency:           {summary['avg_latency_seconds']}s")
    print(f"p50 latency:               {summary['p50_latency_seconds']}s")
    print(f"p95 latency:               {summary['p95_latency_seconds']}s")
    print("==============================\n")


def main() -> None:
    dataset = load_dataset(DATASET_PATH)
    results = []

    print(f"Running full E2E evaluation on {len(dataset)} questions...")
    print(f"Dataset: {DATASET_PATH}")

    for item in dataset:
        result = evaluate_full_item(item)

        print(f"\n[{result['id']}] {result['question']}")

        if result["has_error"]:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Expected route:  {result['expected_route']}")
            print(f"  Actual route:    {result['actual_route']}")
            print(f"  Route correct:   {result['route_correct']}")
            print(f"  Context present: {result['has_expected_context']}")
            print(f"  Contains score:  {result['contains_score']}")
            print(f"  Answer length:   {result['answer_length']}")
            print(f"  Latency:         {result['latency_seconds']}s")

        results.append(result)

    df = pd.DataFrame(results)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    csv_path = RESULTS_DIR / f"full_results_{timestamp}.csv"
    json_path = RESULTS_DIR / f"full_results_{timestamp}.json"
    summary_path = RESULTS_DIR / f"full_summary_{timestamp}.json"

    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False, default=str)

    summary = summarize_results(df)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print_summary(summary)

    print(f"Saved CSV results to:  {csv_path}")
    print(f"Saved JSON results to: {json_path}")
    print(f"Saved summary to:      {summary_path}")


if __name__ == "__main__":
    main()
