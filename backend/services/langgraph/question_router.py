from backend.core.llm import get_chat_model
from backend.schemas.router import QueryState
from backend.services.csv_services.sql_query_templates import find_sql_query_template
from backend.services.kg_services.kg_query_templates import find_kg_query_template

client = get_chat_model()

SQL_ROUTE_TERMS = {
    "revenue",
    "sales",
    "amount",
    "value",
    "money",
    "average order",
    "average review",
    "most sold product",
    "best seller revenue",
}
KG_ROUTE_TERMS = {
    "delayed deliver",
    "delayed order",
    "late deliver",
    "late order",
    "bad review",
    "good review",
    "review score",
    "graph",
    "relationship",
    "connected",
}
VALID_ROUTES = {"pdf", "sql", "kg", "hybrid"}
ROUTER_PROMPT = """
You route questions for a GraphRAG assistant.

Return only one route name:
pdf
sql
kg
hybrid

Routing rules:
- Use kg for graph relationship questions, delayed deliveries, reviews tied to sellers/orders/customers, and connected-entity questions.
- Use sql for structured aggregate questions about revenue, sales, order value, payments, counts, averages, and table-style metrics.
- Use pdf for policy/document questions.
- Use hybrid only when the question clearly needs both documents and structured/graph data.
"""


def normalize_question(question: str) -> str:
    return " ".join(question.strip().lower().split())


def infer_route(question: str) -> str | None:
    normalized = normalize_question(question)

    if any(term in normalized for term in SQL_ROUTE_TERMS):
        return "sql"

    if any(term in normalized for term in KG_ROUTE_TERMS):
        return "kg"

    return None


def infer_template_route(question: str) -> str | None:
    sql_match = find_sql_query_template(question)
    kg_match = find_kg_query_template(question)

    if sql_match and kg_match:
        return "sql" if sql_match.similarity >= kg_match.similarity else "kg"

    if sql_match:
        return "sql"

    if kg_match:
        return "kg"

    return None


def route_question(state: QueryState) -> QueryState:
    question = state["question"]
    template_route = infer_template_route(question)
    if template_route:
        return {
            **state,
            "route": template_route,
        }

    inferred_route = infer_route(question)

    if inferred_route:
        return {
            **state,
            "route": inferred_route,
        }

    response = client.invoke(
        [
            {
                "role": "system",
                "content": ROUTER_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ]
    )

    route = response.content.strip().lower()

    if route not in VALID_ROUTES:
        route = "pdf"

    return {
        **state,
        "route": route,
    }


def choose_route(state: QueryState) -> str:
    return state["route"]
