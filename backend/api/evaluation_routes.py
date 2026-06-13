from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Query
from starlette.concurrency import run_in_threadpool

from backend.evaluation.full_evaluation import (
    DATASET_PATH as FULL_DATASET_PATH,
    run_full_evaluation,
)
from backend.evaluation.router_evaluation import (
    DATASET_PATH,
    evaluate_router_item,
    load_dataset,
    summarize_results,
)


evaluation_router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def run_router_evaluation(
    dataset_path: Path = DATASET_PATH,
    limit: int | None = None,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)

    if limit is not None:
        dataset = dataset[:limit]

    results = [evaluate_router_item(item) for item in dataset]
    summary = summarize_results(pd.DataFrame(results))

    return {
        "dataset_path": str(dataset_path),
        "summary": summary,
        "results": results,
    }


@evaluation_router.get("/questions")
async def list_evaluation_questions(
    limit: int | None = Query(default=None, ge=1),
):
    dataset = await run_in_threadpool(load_dataset, DATASET_PATH)

    if limit is not None:
        dataset = dataset[:limit]

    return {
        "dataset_path": str(DATASET_PATH),
        "total": len(dataset),
        "questions": dataset,
    }


@evaluation_router.post("/run")
async def run_evaluation(
    limit: int | None = Query(default=None, ge=1),
):
    return await run_in_threadpool(run_router_evaluation, DATASET_PATH, limit)


@evaluation_router.post("/full/run")
async def run_full_e2e_evaluation(
    limit: int | None = Query(default=None, ge=1),
    run_deepeval: bool = Query(default=False),
    print_results: bool = Query(default=True),
):
    return await run_in_threadpool(
        run_full_evaluation,
        FULL_DATASET_PATH,
        limit,
        run_deepeval,
        print_results,
    )


@evaluation_router.get("/full/questions")
async def list_full_evaluation_questions(
    limit: int | None = Query(default=None, ge=1),
):
    dataset = await run_in_threadpool(load_dataset, FULL_DATASET_PATH)

    if limit is not None:
        dataset = dataset[:limit]

    return {
        "dataset_path": str(FULL_DATASET_PATH),
        "total": len(dataset),
        "questions": dataset,
    }
