from backend.schemas.router import QueryState
from backend.services.cache.query_cache import get_cached_answer, set_cached_answer

CACHED_RESULT_FIELDS = (
    "question",
    "route",
    "answer",
    "cypher",
    "cypher_source",
    "kg_template_hit",
    "kg_template_name",
    "kg_template_similarity",
    "sql",
    "sql_source",
    "sql_template_hit",
    "sql_template_name",
    "sql_template_similarity",
)


def cache_lookup_node(state: QueryState) -> QueryState:
    cached = get_cached_answer(state["question"])

    if not cached:
        return {
            **state,
            "cache_hit": False,
        }

    return {
        **state,
        **cached,
        "cache_hit": True,
    }


def choose_after_cache(state: QueryState) -> str:
    if state.get("cache_hit"):
        return "cached"

    return "miss"


def cache_save_node(state: QueryState) -> QueryState:
    result = {
        field: state.get(field)
        for field in CACHED_RESULT_FIELDS
        if state.get(field) is not None
    }

    set_cached_answer(state["question"], result)

    return {
        **state,
        "cache_hit": False,
        "cache_type": None,
    }
