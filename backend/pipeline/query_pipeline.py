
from langgraph.graph import StateGraph, START, END

from backend.schemas.router import QueryState
from backend.services.langgraph.question_router import route_question, choose_route
from backend.services.langgraph.cache_nodes import (
    cache_lookup_node,
    choose_after_cache,
    cache_save_node,
)
from backend.services.pdf_services.pdf_retriever import pdf_retriever
from backend.services.csv_services.csv_retriever import postgres_retriever
from backend.services.kg_services.kg_retriever import kg_retriever
from backend.services.langgraph.answer_node import answer_node


def hybrid_retrievers(state: QueryState) -> QueryState:
    state = pdf_retriever(state)
    state = postgres_retriever(state)
    return kg_retriever(state)


graph = StateGraph(QueryState)

graph.add_node("cache_lookup", cache_lookup_node)
graph.add_node("route_question", route_question)
graph.add_node("pdf_retriever", pdf_retriever)
graph.add_node("postgres_retriever", postgres_retriever)
graph.add_node("kg_retriever", kg_retriever)
graph.add_node("hybrid_retrievers", hybrid_retrievers)
graph.add_node("answer_node", answer_node)
graph.add_node("cache_save", cache_save_node)

graph.add_edge(START, "cache_lookup")

graph.add_conditional_edges(
    "cache_lookup",
    choose_after_cache,
    {
        "cached": END,
        "miss": "route_question",
    },
)

graph.add_conditional_edges(
    "route_question",
    choose_route,
    {
        "pdf": "pdf_retriever",
        "sql": "postgres_retriever",
        "kg": "kg_retriever",
        "hybrid": "hybrid_retrievers",
    },
)

graph.add_edge("pdf_retriever", "answer_node")
graph.add_edge("postgres_retriever", "answer_node")
graph.add_edge("kg_retriever", "answer_node")
graph.add_edge("hybrid_retrievers", "answer_node")
graph.add_edge("answer_node", "cache_save")
graph.add_edge("cache_save", END)

query_graph = graph.compile()