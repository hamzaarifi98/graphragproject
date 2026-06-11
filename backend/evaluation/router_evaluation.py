import json
import os
import time
from pathlib import Path
from typing import Any
import sys

import pandas as pd
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add project root to path

from backend.services.langgraph.question_router import route_question




DATASET_PATH = Path(
    os.getenv(
        "GRAPH_RAG_EVAL_DATASET",
        "backend/evaluation/data/questions_router.json",
    )
)
RESULTS_DIR = Path(os.getenv("GRAPH_RAG_EVAL_RESULTS_DIR", "backend/evaluation/results"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Router dataset must be a list of question objects.")

    return data

def evaluate_router_item(item: dict[str, Any]) -> dict[str, Any]:
    question_id = item.get("id")
    question = item.get("question")
    expected_route = item.get("expected_route")

    if not question:
        raise ValueError(f"Missing question in item: {item}")

    start_time = time.perf_counter()
    error = None

    try:
        router_result = route_question({"question": question})
    except Exception as exc:
        router_result = {}
        error = str(exc)

    latency = time.perf_counter() - start_time
    actual_route = router_result.get("route")

    route_correct = None
    if expected_route:
        route_correct = actual_route == expected_route

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
    }


def summarize_results(df: pd.DataFrame) -> dict[str, Any]:
    total = len(df)
    error_rate = df["has_error"].mean() if total > 0 else 0

    route_accuracy = None
    route_df = df[df["route_correct"].notna()]

    if len(route_df) > 0:
        route_accuracy = route_df["route_correct"].mean()

    avg_latency = df["latency_seconds"].mean() if total > 0 else None
    p50_latency = df["latency_seconds"].quantile(0.50) if total > 0 else None
    p95_latency = df["latency_seconds"].quantile(0.95) if total > 0 else None

    return {
        "total_questions": total,
        "error_rate": round(float(error_rate), 4),
        "route_accuracy": round(float(route_accuracy), 4)
        if route_accuracy is not None
        else None,
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


def print_summary(summary: dict[str, Any]) -> None:
    print("\n==============================")
    print("ROUTER EVALUATION SUMMARY")
    print("==============================")
    print(f"Total questions:  {summary['total_questions']}")
    print(f"Error rate:       {summary['error_rate']}")
    print(f"Route accuracy:   {summary['route_accuracy']}")
    print(f"Average latency:  {summary['avg_latency_seconds']}s")
    print(f"p50 latency:      {summary['p50_latency_seconds']}s")
    print(f"p95 latency:      {summary['p95_latency_seconds']}s")
    print("==============================\n")


def main() -> None:
    dataset = load_dataset(DATASET_PATH)
    results = []

    print(f"Running router evaluation on {len(dataset)} questions...")
    print(f"Dataset: {DATASET_PATH}")

    for item in dataset:
        result = evaluate_router_item(item)

        print(f"\n[{result['id']}] {result['question']}")

        if result["has_error"]:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Expected route: {result['expected_route']}")
            print(f"  Actual route:   {result['actual_route']}")
            print(f"  Route correct:  {result['route_correct']}")
            print(f"  Latency:        {result['latency_seconds']}s")

        results.append(result)

    df = pd.DataFrame(results)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    csv_path = RESULTS_DIR / f"router_results_{timestamp}.csv"
    json_path = RESULTS_DIR / f"router_results_{timestamp}.json"
    summary_path = RESULTS_DIR / f"router_summary_{timestamp}.json"

    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    summary = summarize_results(df)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print_summary(summary)

    print(f"Saved CSV results to:  {csv_path}")
    print(f"Saved JSON results to: {json_path}")
    print(f"Saved summary to:      {summary_path}")


if __name__ == "__main__":
    main()
