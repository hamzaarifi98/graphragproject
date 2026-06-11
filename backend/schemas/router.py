from typing import Literal, NotRequired, Required, TypedDict



# backend/schemas/router.py

from typing import Literal, NotRequired, Required, TypedDict


class QueryState(TypedDict):
    question: Required[str]
    route: NotRequired[Literal["pdf", "sql", "kg", "hybrid"]]

    pdf_context: NotRequired[str]
    sql_context: NotRequired[str]
    kg_context: NotRequired[str]
    answer: NotRequired[str]

    cache_hit: NotRequired[bool]
    cache_type: NotRequired[str]
    cache_similarity: NotRequired[float]
    cached_question: NotRequired[str]
