from backend.schemas.router import QueryState


def merge_retriever_cache_hit(
    state: QueryState,
    retriever_name: str,
    cache_hit: bool,
) -> dict:
    retriever_cache_hits = {
        **state.get("retriever_cache_hits", {}),
        retriever_name: cache_hit,
    }
    retriever_cache_types = [
        cache_type
        for cache_type, cache_hit_value in retriever_cache_hits.items()
        if cache_hit_value
    ]

    return {
        "retriever_cache_hit": any(retriever_cache_hits.values()),
        "retriever_cache_type": ", ".join(retriever_cache_types) or None,
        "retriever_cache_hits": retriever_cache_hits,
    }
