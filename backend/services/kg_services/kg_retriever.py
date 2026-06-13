from neo4j.exceptions import ClientError

from backend.schemas.router import QueryState
from backend.services.cache.retriever_cache import (
    NEO4J_CACHE_TTL_SECONDS,
    get_cached_retriever_result,
    set_cached_retriever_result,
)
from backend.services.kg_services.llm_query import generate_cypher_result, validate_cypher
from backend.services.kg_services.kg_query import run_cypher


def ask_neo4j_with_cypher(question: str) -> dict:
    cached = get_cached_retriever_result("neo4j_retriever_cache", question)

    if cached:
        return {
            **cached,
            "cache_hit": True,
            "cache_type": "neo4j",
        }

    generated = generate_cypher_result(question)
    cypher = generated.cypher
    validate_cypher(cypher)

    try:
        rows = run_cypher(cypher)
    except ClientError as exc:
        first_error = str(exc)
        try:
            generated = generate_cypher_result(
                question,
                previous_cypher=cypher,
                error=first_error,
            )
            cypher = generated.cypher
            validate_cypher(cypher)
            rows = run_cypher(cypher)
        except (ClientError, ValueError) as retry_exc:
            return {
                "cypher": cypher,
                "cypher_source": generated.source,
                "kg_template_hit": generated.source == "template",
                "kg_template_name": generated.template_name,
                "kg_template_similarity": generated.template_similarity,
                "rows": [],
                "error": f"Neo4j query failed after retry: {retry_exc}",
                "cache_hit": False,
                "cache_type": "neo4j",
            }

    result = {
        "cypher": cypher,
        "cypher_source": generated.source,
        "kg_template_hit": generated.source == "template",
        "kg_template_name": generated.template_name,
        "kg_template_similarity": generated.template_similarity,
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
        "cypher_source": kg_result.get("cypher_source"),
        "kg_template_hit": kg_result.get("kg_template_hit", False),
        "kg_template_name": kg_result.get("kg_template_name"),
        "kg_template_similarity": kg_result.get("kg_template_similarity"),
        "retriever_cache_hit": any(retriever_cache_hits.values()),
        "retriever_cache_type": ", ".join(retriever_cache_types) or None,
        "retriever_cache_hits": retriever_cache_hits,
    }
