import hashlib
import json
import os
from collections.abc import Iterator
from itertools import islice
from typing import Any

import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

SQL_CACHE_TTL_SECONDS = int(os.getenv("SQL_CACHE_TTL_SECONDS", "1800"))
NEO4J_CACHE_TTL_SECONDS = int(os.getenv("NEO4J_CACHE_TTL_SECONDS", "1800"))

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
)

CLEAR_BATCH_SIZE = 500


def batched(iterator: Iterator[str], size: int = CLEAR_BATCH_SIZE) -> Iterator[list[str]]:
    while batch := list(islice(iterator, size)):
        yield batch


def normalize_key(text: str) -> str:
    return " ".join(text.strip().lower().split())


def make_cache_key(prefix: str, value: str) -> str:
    normalized = normalize_key(value)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def get_cached_retriever_result(
    prefix: str,
    question: str,
) -> dict[str, Any] | None:
    cache_key = make_cache_key(prefix, question)
    cached = redis_client.get(cache_key)

    if not cached:
        return None

    return json.loads(cached)


def set_cached_retriever_result(
    prefix: str,
    question: str,
    result: dict[str, Any],
    ttl_seconds: int,
) -> None:
    cache_key = make_cache_key(prefix, question)

    redis_client.setex(
        cache_key,
        ttl_seconds,
        json.dumps(result, default=str),
    )


def clear_retriever_cache(prefix: str) -> None:
    for keys in batched(redis_client.scan_iter(f"{prefix}:*")):
        redis_client.delete(*keys)
