from typing import Literal, NotRequired, Required, TypedDict



class QueryState(TypedDict):
    question: Required[str]
    route: NotRequired[Literal["pdf", "sql", "hybrid"]]
    pdf_context: NotRequired[str]
    sql_context: NotRequired[str]
    answer: NotRequired[str]
