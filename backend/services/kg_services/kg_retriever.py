from neo4j.exceptions import ClientError

from backend.schemas.router import QueryState
from backend.services.cache.retriever_cache import (
    NEO4J_CACHE_TTL_SECONDS,
    get_cached_retriever_result,
    set_cached_retriever_result,
)
from backend.services.kg_services.llm_query import generate_cypher, validate_cypher
from backend.services.kg_services.kg_query import run_cypher


def ask_neo4j_with_cypher(question: str) -> dict:
    cached = get_cached_retriever_result("neo4j_retriever_cache", question)

    if cached:
        return {
            **cached,
            "cache_hit": True,
            "cache_type": "neo4j",
        }

    cypher = generate_cypher(question)
    validate_cypher(cypher)

    try:
        rows = run_cypher(cypher)
    except ClientError as exc:
        first_error = str(exc)
        try:
            cypher = generate_cypher(question, previous_cypher=cypher, error=first_error)
            validate_cypher(cypher)
            rows = run_cypher(cypher)
        except (ClientError, ValueError) as retry_exc:
            return {
                "cypher": cypher,
                "rows": [],
                "error": f"Neo4j query failed after retry: {retry_exc}",
                "cache_hit": False,
                "cache_type": "neo4j",
            }

    result = {
        "cypher": cypher,
        "rows": rows,
    }

    set_cached_retriever_result(
        "neo4j_retriever_cache",
        question,
        result,
        NEO4J_CACHE_TTL_SECONDS,
    )

    return {
        **result,
        "cache_hit": False,
        "cache_type": "neo4j",
    }


def kg_retriever(state: QueryState) -> QueryState:
    kg_result = ask_neo4j_with_cypher(state["question"])
    retriever_cache_hits = {
        **state.get("retriever_cache_hits", {}),
        "neo4j": kg_result["cache_hit"],
    }
    retriever_cache_types = [
        cache_type
        for cache_type, cache_hit in retriever_cache_hits.items()
        if cache_hit
    ]

    return {
        **state,
        "kg_context": str(kg_result),
        "retriever_cache_hit": any(retriever_cache_hits.values()),
        "retriever_cache_type": ", ".join(retriever_cache_types) or None,
        "retriever_cache_hits": retriever_cache_hits,
    }
