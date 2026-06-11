# backend/services/langgraph/cache_nodes.py

from backend.schemas.router import QueryState
from backend.services.cache.querry_cache import get_cached_answer, set_cached_answer


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
        "question": state.get("question"),
        "route": state.get("route"),
        "answer": state.get("answer"),
    }

    set_cached_answer(state["question"], result)

    return {
        **state,
        "cache_hit": False,
        "cache_type": None,
    }