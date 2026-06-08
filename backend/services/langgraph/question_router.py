from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from backend.schemas.router import QueryState

load_dotenv()




client = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


def route_question(state: QueryState) -> QueryState:
    question = state["question"]

    response = client.invoke([
        {
            "role": "system",
            "content": """
You are a router for a RAG invoice assistant.

Return only one word:

pdf = if the question asks about policies, rules, documents, return policy, refund policy
sql = if the question asks about invoices, orders, customers, prices, totals, dates
hybrid = if it needs both structured data and document policy
"""
        },
        {
            "role": "user",
            "content": question
        }
    ])

    route = response.content.strip().lower()

    if route not in ["pdf", "sql", "hybrid"]:
        route = "pdf"

    return {
        **state,
        "route": route
    }


def choose_route(state: QueryState) -> str:
    return state["route"]
