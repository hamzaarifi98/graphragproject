# backend/schemas/router.py

from typing import Literal
from typing_extensions import NotRequired, Required, TypedDict


class QueryState(TypedDict):
    question: Required[str]
    route: NotRequired[Literal["pdf", "sql", "kg", "hybrid"]]

    pdf_context: NotRequired[str]
    sql_context: NotRequired[str]
    kg_context: NotRequired[str]
    answer: NotRequired[str]

    cypher_source: NotRequired[str]
    kg_template_hit: NotRequired[bool]
    kg_template_name: NotRequired[str]
    kg_template_similarity: NotRequired[float]
    sql_source: NotRequired[str]
    sql_template_hit: NotRequired[bool]
    sql_template_name: NotRequired[str]
    sql_template_similarity: NotRequired[float]

    cache_hit: NotRequired[bool]
    cache_type: NotRequired[str]
    cache_similarity: NotRequired[float]
    cached_question: NotRequired[str]

    retriever_cache_hit: NotRequired[bool]
    retriever_cache_type: NotRequired[str]
    retriever_cache_hits: NotRequired[dict[str, bool]]
