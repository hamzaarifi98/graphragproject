# backend/services/cache/querry_cache.py

import hashlib
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

import redis
from dotenv import load_dotenv

from backend.services.pdf_services.text_embedder import embed_text

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUERY_CACHE_VERSION = os.getenv("QUERY_CACHE_VERSION", "v1")
QUERY_CACHE_TTL_SECONDS = int(os.getenv("QUERY_CACHE_TTL_SECONDS", "600"))
QUERY_CACHE_SIMILARITY_THRESHOLD = float(
    os.getenv("QUERY_CACHE_SIMILARITY_THRESHOLD", "0.85")
)
QUERY_CACHE_KEYWORD_BOOST = float(os.getenv("QUERY_CACHE_KEYWORD_BOOST", "0.08"))
QUERY_CACHE_MAX_SCORE = 1.0

EXACT_PREFIX = f"query_cache:{QUERY_CACHE_VERSION}:exact"
SEMANTIC_PREFIX = f"query_cache:{QUERY_CACHE_VERSION}:semantic"
SEMANTIC_INDEX_KEY = f"query_cache:{QUERY_CACHE_VERSION}:semantic_keys"
MONTH_PATTERN = re.compile(
    r"\b("
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december"
    r")\b"
)
YEAR_PATTERN = re.compile(r"\b20\d{2}\b")
RELATIVE_TIME_PATTERNS = (
    "today",
    "yesterday",
    "this week",
    "last week",
    "this month",
    "last month",
    "this year",
    "last year",
)

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
)


@dataclass(frozen=True)
class SemanticMatch:
    entry: dict[str, Any]
    similarity: float
    score: float


def normalize_question(question: str) -> str:
    return " ".join(question.strip().lower().split())


def normalize_semantic_question(question: str) -> str:
    normalized = normalize_question(question)

    replacements = {
        "best seller": "seller sold most",
        "top seller": "seller sold most",
        "highest seller": "seller sold most",
        "sold most": "seller sold most",
        "most sold": "seller sold most",
        "most sales": "seller sold most",
    }

    for source, target in replacements.items():
        normalized = normalized.replace(source, target)

    return normalized


def infer_query_intent(question: str) -> str:
    normalized = normalize_semantic_question(question)
    terms = set(normalized.split())
    asks_revenue = bool(
        {"revenue", "income", "sales", "value", "amount", "money"} & terms
    )
    asks_best_seller = {"seller", "sold", "most"}.issubset(terms)

    if asks_best_seller and asks_revenue:
        return "best_seller_revenue"

    if asks_best_seller:
        return "best_seller_identity"

    return "general"


def extract_time_scope(question: str) -> tuple[str, ...]:
    normalized = normalize_question(question)
    terms = set(MONTH_PATTERN.findall(normalized))
    terms.update(YEAR_PATTERN.findall(normalized))

    for pattern in RELATIVE_TIME_PATTERNS:
        if pattern in normalized:
            terms.add(pattern)

    return tuple(sorted(terms))


def make_exact_cache_key(question: str) -> str:
    normalized = normalize_question(question)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{EXACT_PREFIX}:{digest}"


def cosine_similarity(first: list[float], second: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(first, second))
    first_norm = math.sqrt(sum(a * a for a in first))
    second_norm = math.sqrt(sum(b * b for b in second))

    if first_norm == 0 or second_norm == 0:
        return 0.0

    return dot_product / (first_norm * second_norm)


def keyword_boost(question: str, cached_question: str) -> float:
    question_terms = set(normalize_semantic_question(question).split())
    cached_terms = set(normalize_semantic_question(cached_question).split())

    if not question_terms or not cached_terms:
        return 0.0

    overlap = len(question_terms & cached_terms) / len(question_terms | cached_terms)

    if {"seller", "sold", "most"}.issubset(question_terms | cached_terms):
        return min(QUERY_CACHE_KEYWORD_BOOST, overlap)

    return 0.0


def find_best_semantic_match(question: str) -> SemanticMatch | None:
    semantic_question = normalize_semantic_question(question)
    question_embedding = embed_text(semantic_question)
    question_intent = infer_query_intent(question)
    question_time_scope = extract_time_scope(question)
    semantic_keys = list(redis_client.smembers(SEMANTIC_INDEX_KEY))

    if not semantic_keys:
        return None

    cached_values = redis_client.mget(semantic_keys)
    best_entry: dict[str, Any] | None = None
    best_similarity = 0.0
    best_cosine_score = 0.0
    stale_keys = []

    for semantic_key, cached_json in zip(semantic_keys, cached_values):
        if not cached_json:
            stale_keys.append(semantic_key)
            continue

        cached_entry = json.loads(cached_json)

        if cached_entry.get("intent", "general") != question_intent:
            continue

        cached_time_scope = tuple(cached_entry.get("time_scope", ()))
        if cached_time_scope != question_time_scope:
            continue

        cosine_score = cosine_similarity(
            question_embedding,
            cached_entry["question_embedding"],
        )
        match_score = cosine_score + keyword_boost(question, cached_entry["question"])
        match_score = min(match_score, QUERY_CACHE_MAX_SCORE)

        if match_score > best_similarity:
            best_similarity = match_score
            best_entry = cached_entry
            best_cosine_score = cosine_score

    if stale_keys:
        redis_client.srem(SEMANTIC_INDEX_KEY, *stale_keys)

    if best_entry is None:
        return None

    return SemanticMatch(
        entry=best_entry,
        similarity=best_cosine_score,
        score=best_similarity,
    )


def get_cached_answer(question: str) -> dict[str, Any] | None:
    exact_key = make_exact_cache_key(question)
    exact_cached = redis_client.get(exact_key)

    if exact_cached:
        result = json.loads(exact_cached)
        return {
            **result,
            "cache_hit": True,
            "cache_type": "exact",
            "cache_similarity": 1.0,
            "cached_question": question,
        }

    best_match = find_best_semantic_match(question)

    if best_match is None:
        return None

    if best_match.score < QUERY_CACHE_SIMILARITY_THRESHOLD:
        return None

    return {
        **best_match.entry["result"],
        "question": question,
        "cache_hit": True,
        "cache_type": "semantic",
        "cache_similarity": round(best_match.similarity, 4),
        "cache_score": round(best_match.score, 4),
        "cached_question": best_match.entry["question"],
    }


def set_cached_answer(question: str, result: dict[str, Any]) -> None:
    exact_key = make_exact_cache_key(question)
    semantic_question = normalize_semantic_question(question)
    question_embedding = embed_text(semantic_question)

    semantic_key = f"{SEMANTIC_PREFIX}:{uuid.uuid4().hex}"

    semantic_entry = {
        "question": question,
        "normalized_question": normalize_question(question),
        "semantic_question": semantic_question,
        "intent": infer_query_intent(question),
        "time_scope": extract_time_scope(question),
        "question_embedding": question_embedding,
        "result": result,
        "created_at": int(time.time()),
    }

    pipe = redis_client.pipeline()
    pipe.setex(exact_key, QUERY_CACHE_TTL_SECONDS, json.dumps(result))
    pipe.setex(
        semantic_key,
        QUERY_CACHE_TTL_SECONDS,
        json.dumps(semantic_entry),
    )
    pipe.sadd(SEMANTIC_INDEX_KEY, semantic_key)
    pipe.execute()


def clear_query_cache() -> None:
    semantic_keys = list(redis_client.smembers(SEMANTIC_INDEX_KEY))

    if semantic_keys:
        redis_client.delete(*semantic_keys)

    redis_client.delete(SEMANTIC_INDEX_KEY)

    exact_keys = list(redis_client.scan_iter(f"{EXACT_PREFIX}:*"))

    if exact_keys:
        redis_client.delete(*exact_keys)


def get_best_semantic_match(question: str) -> dict[str, Any] | None:
    question_intent = infer_query_intent(question)
    best_match = find_best_semantic_match(question)

    if best_match is None:
        return None

    return {
        "question": question,
        "semantic_question": normalize_semantic_question(question),
        "intent": question_intent,
        "cached_question": best_match.entry["question"],
        "cached_semantic_question": best_match.entry.get("semantic_question"),
        "cached_intent": best_match.entry.get("intent"),
        "similarity": round(best_match.similarity, 4),
        "score": round(best_match.score, 4),
        "threshold": QUERY_CACHE_SIMILARITY_THRESHOLD,
        "would_hit": best_match.score >= QUERY_CACHE_SIMILARITY_THRESHOLD,
    }
