from langgraph.graph import StateGraph, START, END

from backend.schemas.router import QueryState
from backend.services.langgraph.question_router import route_question, choose_route
from backend.services.pdf_services.pdf_retriever import pdf_retriever
from backend.services.csv_services.csv_retriever import postgres_retriever
from backend.services.langgraph.answer_node import answer_node


def both_retrievers(state: QueryState) -> QueryState:
    state = pdf_retriever(state)
    return postgres_retriever(state)


graph = StateGraph(QueryState)

graph.add_node("route_question", route_question)
graph.add_node("pdf_retriever", pdf_retriever)
graph.add_node("postgres_retriever", postgres_retriever)
graph.add_node("both_retrievers", both_retrievers)
graph.add_node("answer_node", answer_node)

graph.add_edge(START, "route_question")

graph.add_conditional_edges(
    "route_question",
    choose_route,
    {
        "pdf": "pdf_retriever",
        "sql": "postgres_retriever",
        "hybrid": "both_retrievers",
    },
)

graph.add_edge("pdf_retriever", "answer_node")
graph.add_edge("postgres_retriever", "answer_node")
graph.add_edge("both_retrievers", "answer_node")
graph.add_edge("answer_node", END)

query_graph = graph.compile()
