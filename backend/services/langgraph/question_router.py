from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from backend.schemas.router import QueryState
from backend.services.csv_services.sql_query_templates import find_sql_query_template
from backend.services.kg_services.kg_query_templates import find_kg_query_template

load_dotenv()

client = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

SQL_ROUTE_TERMS = {
    "revenue",
    "sales",
    "amount",
    "value",
    "money",
    "average order",
    "most sold product",
    "best seller revenue",
}
KG_ROUTE_TERMS = {
    "delayed deliver",
    "bad review",
    "graph",
    "relationship",
    "connected",
}
VALID_ROUTES = {"pdf", "sql", "kg", "hybrid"}


def normalize_question(question: str) -> str:
    return " ".join(question.strip().lower().split())


def infer_route(question: str) -> str | None:
    normalized = normalize_question(question)

    if any(term in normalized for term in KG_ROUTE_TERMS):
        return "kg"

    if any(term in normalized for term in SQL_ROUTE_TERMS):
        return "sql"

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
                "content": """
You are a router for a RAG invoice assistant.

If question is : [
        Who had most delayed deliveries in the last month?
        Who had most bad reviews in the last month?
        Who sold the most in the last month?
] or something similar you rout to kg because it needs graph 
relationships to answer. 

If question is : [
what is the total revenue for last month?
what is the average order value for last month?
whats the most sold products in last month
tell me total revenue of best seller
] then when route to sql.

Questions asking for revenue, sales amount, value, or money should route to sql.

Return only one word:

pdf 
sql 
kg
hybrid 


"""

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
