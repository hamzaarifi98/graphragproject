from backend.core.llm import get_chat_model
from backend.schemas.router import QueryState

client = get_chat_model("gpt-4o-mini")

ANSWER_SYSTEM_PROMPT = """
You are a helpful GraphRAG assistant.

Answer the user using only the provided context.

Rules:
- If the context is enough, answer clearly and directly.
- If a SQL or Neo4j context contains an error, explain that error and the needed ingest/build step.
- If the context is missing or not enough and there is no retriever error, say that you do not have enough information.
- Do not invent facts.
- If SQL/table context is provided, use it for structured data questions.
- Also know that last date is from 2018 so when asked last month you can say last month from 2018 and use that date for filtering.
- If PDF context is provided, use it for policy/document questions.
- If Neo4j graph context is provided, use it for relationship and connected-entity questions.
"""


def answer_node(state: QueryState) -> QueryState:
    question = state["question"]
    pdf_context = state.get("pdf_context", "")
    sql_context = state.get("sql_context", "")
    kg_context = state.get("kg_context", "")

    response = client.invoke(
        [
            {
                "role": "system",
                "content": ANSWER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
Question:
{question}

PDF context:
{pdf_context}

SQL/Postgres context:
{sql_context}

Neo4j graph context:
{kg_context}
""",
            },
        ]
    )

    return {
        **state,
        "answer": response.content,
    }
