import argparse
import ast
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "GraphRAG Full Evaluation")
os.environ.setdefault("LANGSMITH_PROJECT", os.environ["LANGCHAIN_PROJECT"])

from deepeval import evaluate
from deepeval.evaluate import DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

try:
    from langsmith import traceable
except (ImportError, ModuleNotFoundError):

    def traceable(*_args: Any, **_kwargs: Any):
        def decorator(function):
            return function

        return decorator

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from backend.evaluation.router_evaluation import RESULTS_DIR, load_dataset
from backend.pipeline.query_pipeline import query_graph


DATASET_PATH = Path(
    os.getenv(
        "GRAPH_RAG_FULL_EVAL_DATASET",
        "backend/evaluation/data/questions_full.json",
    )
)
RESULT_PREFIX = "full_deepeval"
DEEPEVAL_MODEL = os.getenv("DEEPEVAL_MODEL", "gpt-5.4-mini")
NOT_ENOUGH_INFORMATION_PHRASES = (
    "do not have enough information",
    "don't have enough information",
    "not enough information",
    "cannot answer",
    "can't answer",
)


@dataclass(frozen=True)
class EvaluatedCase:
    item: dict[str, Any]
    graph_result: dict[str, Any]
    latency_seconds: float
    error: str | None = None


def normalize_text(text: str | None) -> str:
    if not text:
        return ""

    return " ".join(text.lower().replace("\n", " ").replace("\t", " ").split())


def expected_output_for(item: dict[str, Any]) -> str:
    expected_output = item.get("expected_output")
    if expected_output:
        return str(expected_output)

    expected_contains = item.get("expected_contains") or item.get(
        "expected_answer_contains"
    )
    if expected_contains:
        return "Answer should include: " + ", ".join(expected_contains)

    return "Answer should be correct, grounded, and relevant."


def expected_contains_for(item: dict[str, Any]) -> list[str]:
    expected_contains = item.get("expected_contains") or item.get(
        "expected_answer_contains"
    )

    if isinstance(expected_contains, list):
        return [str(value) for value in expected_contains]

    return []


def contains_score(answer: str, expected_contains: list[str]) -> float | None:
    if not expected_contains:
        return None

    answer_norm = normalize_text(answer)
    matched = sum(
        1 for expected in expected_contains if normalize_text(expected) in answer_norm
    )
    return matched / len(expected_contains)


def answer_has_enough_information(answer: str) -> bool:
    answer_norm = normalize_text(answer)
    return bool(answer_norm) and not any(
        phrase in answer_norm for phrase in NOT_ENOUGH_INFORMATION_PHRASES
    )


def parse_context(context: str | None) -> dict[str, Any]:
    if not context:
        return {}

    try:
        parsed = ast.literal_eval(context)
    except (SyntaxError, ValueError):
        return {}

    return parsed if isinstance(parsed, dict) else {}


def context_has_rows(context: str | None) -> bool:
    parsed = parse_context(context)
    rows = parsed.get("rows")

    if isinstance(rows, list):
        return len(rows) > 0

    return bool(context and re.search(r"['\"]rows['\"]\s*:\s*\[(?!\])", context))


def context_present_for_route(result: dict[str, Any], route: str | None) -> bool:
    if route == "pdf":
        return bool(result.get("pdf_context"))

    if route == "sql":
        return context_has_rows(result.get("sql_context"))

    if route == "kg":
        return context_has_rows(result.get("kg_context"))

    if route == "hybrid":
        return any(
            (
                bool(result.get("pdf_context")),
                context_has_rows(result.get("sql_context")),
                context_has_rows(result.get("kg_context")),
            )
        )

    return False


def retrieval_context_from(result: dict[str, Any]) -> list[str]:
    contexts = [
        result.get("pdf_context"),
        result.get("sql_context"),
        result.get("kg_context"),
    ]
    return [str(context) for context in contexts if context]


@traceable(name="full_evaluation_case", run_type="chain")
def run_pipeline(item: dict[str, Any]) -> EvaluatedCase:
    question = item.get("question")
    if not question:
        raise ValueError(f"Missing question in item: {item}")

    start_time = time.perf_counter()
    try:
        graph_result = query_graph.invoke({"question": question})
        error = None
    except Exception as exc:
        graph_result = {}
        error = str(exc)

    return EvaluatedCase(
        item=item,
        graph_result=graph_result,
        latency_seconds=time.perf_counter() - start_time,
        error=error,
    )


def build_test_case(case: EvaluatedCase) -> LLMTestCase | None:
    if case.error:
        return None

    question = case.item["question"]
    answer = case.graph_result.get("answer") or ""
    retrieval_context = retrieval_context_from(case.graph_result)

    return LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected_output_for(case.item),
        retrieval_context=retrieval_context,
        context=retrieval_context,
        completion_time=case.latency_seconds,
        name=f"full-{case.item.get('id', question)}",
        tags=[
            str(case.item.get("expected_route", "unknown")),
            str(case.item.get("question_type", "unknown")),
            str(case.item.get("difficulty", "unknown")),
        ],
        metadata={
            "id": case.item.get("id"),
            "expected_route": case.item.get("expected_route"),
            "actual_route": case.graph_result.get("route"),
            "route_correct": route_correct(case),
            "expected_contains": expected_contains_for(case.item),
            "cache_hit": case.graph_result.get("cache_hit", False),
            "retriever_cache_hit": case.graph_result.get(
                "retriever_cache_hit",
                False,
            ),
        },
    )


def route_correct(case: EvaluatedCase) -> bool | None:
    expected_route = case.item.get("expected_route")
    if not expected_route:
        return None

    return case.graph_result.get("route") == expected_route


def result_row(case: EvaluatedCase) -> dict[str, Any]:
    item = case.item
    result = case.graph_result
    answer = result.get("answer") or ""
    expected_contains = expected_contains_for(item)
    actual_route = result.get("route")

    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "expected_route": item.get("expected_route"),
        "actual_route": actual_route,
        "route_correct": route_correct(case),
        "question_type": item.get("question_type"),
        "difficulty": item.get("difficulty"),
        "latency_seconds": round(case.latency_seconds, 4),
        "error": case.error,
        "has_error": case.error is not None,
        "answer": answer,
        "answer_length": len(answer),
        "answer_present": bool(answer),
        "answer_has_enough_information": answer_has_enough_information(answer),
        "expected_contains": expected_contains,
        "contains_score": contains_score(answer, expected_contains),
        "has_expected_context": context_present_for_route(result, actual_route),
        "cache_hit": result.get("cache_hit", False),
        "cache_type": result.get("cache_type"),
        "retriever_cache_hit": result.get("retriever_cache_hit", False),
        "retriever_cache_type": result.get("retriever_cache_type"),
        "retriever_cache_hits": result.get("retriever_cache_hits", {}),
        "pdf_context_present": bool(result.get("pdf_context")),
        "sql_context_has_rows": context_has_rows(result.get("sql_context")),
        "kg_context_has_rows": context_has_rows(result.get("kg_context")),
        "sql_template_hit": result.get("sql_template_hit", False),
        "sql_template_name": result.get("sql_template_name"),
        "kg_template_hit": result.get("kg_template_hit", False),
        "kg_template_name": result.get("kg_template_name"),
    }


def mean_bool(df: pd.DataFrame, column: str) -> float:
    if column not in df or df.empty:
        return 0.0

    return round(float(df[column].fillna(False).mean()), 4)


def mean_nullable(df: pd.DataFrame, column: str) -> float | None:
    if column not in df:
        return None

    values = df[df[column].notna()][column]
    if values.empty:
        return None

    return round(float(values.mean()), 4)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    total = len(df)

    if total == 0:
        return {"total_questions": 0}

    return {
        "total_questions": total,
        "error_rate": mean_bool(df, "has_error"),
        "route_accuracy": mean_nullable(df, "route_correct"),
        "answer_present_rate": mean_bool(df, "answer_present"),
        "answer_enough_information_rate": mean_bool(
            df,
            "answer_has_enough_information",
        ),
        "context_present_rate": mean_bool(df, "has_expected_context"),
        "avg_contains_score": mean_nullable(df, "contains_score"),
        "cache_hit_rate": mean_bool(df, "cache_hit"),
        "retriever_cache_hit_rate": mean_bool(df, "retriever_cache_hit"),
        "sql_template_hit_rate": mean_bool(df, "sql_template_hit"),
        "kg_template_hit_rate": mean_bool(df, "kg_template_hit"),
        "avg_latency_seconds": round(float(df["latency_seconds"].mean()), 4),
        "p50_latency_seconds": round(float(df["latency_seconds"].quantile(0.50)), 4),
        "p95_latency_seconds": round(float(df["latency_seconds"].quantile(0.95)), 4),
    }


def deepeval_metrics() -> list[Any]:
    return [
        AnswerRelevancyMetric(model=DEEPEVAL_MODEL, threshold=0.7),
        FaithfulnessMetric(model=DEEPEVAL_MODEL, threshold=0.7),
        ContextualRelevancyMetric(model=DEEPEVAL_MODEL, threshold=0.7),
    ]


def save_outputs(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    timestamp: str,
) -> dict[str, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / f"{RESULT_PREFIX}_results_{timestamp}.csv"
    json_path = RESULTS_DIR / f"{RESULT_PREFIX}_results_{timestamp}.json"
    summary_path = RESULTS_DIR / f"{RESULT_PREFIX}_summary_{timestamp}.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False, default=str)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return {
        "csv": csv_path,
        "json": json_path,
        "summary": summary_path,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n==============================")
    print("FULL DEEPEVAL SUMMARY")
    print("==============================")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("==============================\n")


def run_full_evaluation(
    dataset_path: Path = DATASET_PATH,
    limit: int | None = None,
    run_deepeval: bool = False,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    if limit is not None:
        dataset = dataset[:limit]

    cases = [run_pipeline(item) for item in dataset]
    rows = [result_row(case) for case in cases]
    test_cases = [
        test_case
        for case in cases
        if (test_case := build_test_case(case)) is not None
    ]

    if test_cases and run_deepeval:
        evaluate(
            test_cases=test_cases,
            metrics=deepeval_metrics(),
            display_config=DisplayConfig(
                file_type="md",
                file_output_dir=str(RESULTS_DIR),
                print_results=False,
            ),
        )

    return {
        "dataset_path": str(dataset_path),
        "summary": summarize(rows),
        "results": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Graph RAG Deepeval.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-deepeval",
        action="store_true",
        help="Only run deterministic route/context/latency checks.",
    )
    parser.add_argument(
        "--no-print-results",
        action="store_true",
        help="Suppress detailed Deepeval terminal output.",
    )
    parser.add_argument(
        "--disable-langsmith",
        action="store_true",
        help="Disable LangSmith tracing for local/offline runs.",
    )
    return parser.parse_args()


def configure_langsmith(disabled: bool) -> None:
    if disabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGCHAIN_PROJECT", "GraphRAG Full Evaluation")
    os.environ.setdefault("LANGSMITH_PROJECT", os.environ["LANGCHAIN_PROJECT"])


def main() -> None:
    args = parse_args()
    configure_langsmith(args.disable_langsmith)
    dataset = load_dataset(args.dataset)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    print(f"Running full evaluation on {len(dataset)} questions...")
    print(f"Dataset: {args.dataset}")
    print(f"Deepeval model: {DEEPEVAL_MODEL}")
    print(f"LangSmith tracing: {not args.disable_langsmith}")
    if not args.disable_langsmith:
        print(f"LangSmith project: {os.environ.get('LANGCHAIN_PROJECT')}")

    cases = [run_pipeline(item) for item in dataset]
    rows = [result_row(case) for case in cases]
    test_cases = [
        test_case
        for case in cases
        if (test_case := build_test_case(case)) is not None
    ]

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if test_cases and not args.skip_deepeval:
        evaluate(
            test_cases=test_cases,
            metrics=deepeval_metrics(),
            display_config=DisplayConfig(
                file_type="md",
                file_output_dir=str(RESULTS_DIR),
                print_results=not args.no_print_results,
            ),
        )

    summary = summarize(rows)
    paths = save_outputs(rows, summary, timestamp)

    print_summary(summary)
    print(f"Saved CSV results to:  {paths['csv']}")
    print(f"Saved JSON results to: {paths['json']}")
    print(f"Saved summary to:      {paths['summary']}")


if __name__ == "__main__":
    main()
