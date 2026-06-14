from neo4j.exceptions import ClientError

from backend.schemas.router import QueryState
from backend.services.cache.retriever_cache import (
    NEO4J_RETRIEVER_CACHE_PREFIX,
    NEO4J_CACHE_TTL_SECONDS,
    get_cached_retriever_result,
    set_cached_retriever_result,
)
from backend.services.kg_services.llm_query import generate_cypher_result, validate_cypher
from backend.services.kg_services.kg_query import run_cypher
from backend.services.langgraph.retriever_state import merge_retriever_cache_hit

MAX_KG_CONTEXT_ROWS = 50


def get_graph_readiness_error() -> str | None:
    counts = run_cypher("""
        MATCH (o:Order)
        WITH count(o) AS orders
        OPTIONAL MATCH (:Order)-[review_rel:HAS_REVIEW]->(:Review)
        WITH orders, count(review_rel) AS review_relationships
        OPTIONAL MATCH (:Order)-[seller_rel:SOLD_BY]->(:Seller)
        WITH
            orders,
            review_relationships,
            count(seller_rel) AS seller_relationships
        OPTIONAL MATCH (dated_order:Order)
        WHERE dated_order.delivered_customer_date IS NOT NULL
          AND dated_order.estimated_delivery_date IS NOT NULL
        RETURN
            orders,
            review_relationships,
            seller_relationships,
            count(dated_order) AS dated_orders
    """)

    summary = counts[0] if counts else {}

    if summary.get("orders", 0) == 0:
        return "Neo4j graph is empty. Run POST /structured/ingest, then GET /kg/build."

    if summary.get("review_relationships", 0) == 0:
        return "Neo4j graph has no HAS_REVIEW relationships. Rebuild it with GET /kg/build."

    if summary.get("seller_relationships", 0) == 0:
        return "Neo4j graph has no SOLD_BY relationships. Rebuild it with GET /kg/build."

    if summary.get("dated_orders", 0) == 0:
        return (
            "Neo4j Order nodes do not have delivery date properties. "
            "Rebuild the graph with GET /kg/build."
        )

    return None


def ask_neo4j_with_cypher(question: str) -> dict:
    cached = get_cached_retriever_result(NEO4J_RETRIEVER_CACHE_PREFIX, question)

    if cached:
        return {
            **normalize_kg_result(cached),
            "cache_hit": True,
            "cache_type": "neo4j",
        }

    try:
        readiness_error = get_graph_readiness_error()
    except ClientError as exc:
        return {
            "cypher": None,
            "cypher_source": None,
            "kg_template_hit": False,
            "kg_template_name": None,
            "kg_template_similarity": None,
            "rows": [],
            "error": f"Could not inspect Neo4j graph: {exc}",
            "cache_hit": False,
            "cache_type": "neo4j",
        }

    if readiness_error:
        return {
            "cypher": None,
            "cypher_source": None,
            "kg_template_hit": False,
            "kg_template_name": None,
            "kg_template_similarity": None,
            "rows": [],
            "error": readiness_error,
            "cache_hit": False,
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

    result = normalize_kg_result({
        "cypher": cypher,
        "cypher_source": generated.source,
        "kg_template_hit": generated.source == "template",
        "kg_template_name": generated.template_name,
        "kg_template_similarity": generated.template_similarity,
        "rows": rows,
    })

    set_cached_retriever_result(
        NEO4J_RETRIEVER_CACHE_PREFIX,
        question,
        result,
        NEO4J_CACHE_TTL_SECONDS,
    )

    return {
        **result,
        "cache_hit": False,
        "cache_type": "neo4j",
    }


def normalize_kg_result(result: dict) -> dict:
    rows = result.get("rows", [])
    if isinstance(rows, list):
        rows = rows[:MAX_KG_CONTEXT_ROWS]

    return {
        **result,
        "rows": rows,
        "row_count": (
            len(result.get("rows", []))
            if isinstance(result.get("rows"), list)
            else None
        ),
        "rows_truncated": (
            isinstance(result.get("rows"), list)
            and len(result.get("rows", [])) > MAX_KG_CONTEXT_ROWS
        ),
    }


def kg_retriever(state: QueryState) -> QueryState:
    kg_result = ask_neo4j_with_cypher(state["question"])

    return {
        **state,
        **merge_retriever_cache_hit(state, "neo4j", kg_result["cache_hit"]),
        "kg_context": str(kg_result),
        "cypher": kg_result.get("cypher"),
        "cypher_source": kg_result.get("cypher_source"),
        "kg_template_hit": kg_result.get("kg_template_hit", False),
        "kg_template_name": kg_result.get("kg_template_name"),
        "kg_template_similarity": kg_result.get("kg_template_similarity"),
    }
