from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from backend.schemas.router import QueryState

load_dotenv()

client = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


def answer_node(state: QueryState) -> QueryState:
    question = state["question"]
    pdf_context = state.get("pdf_context", "")
    sql_context = state.get("sql_context", "")

    response = client.invoke(
        [
            {
                "role": "system",
                "content": """
You are a helpful RAG invoice assistant.

Answer the user using only the provided context.

Rules:
- If the context is enough, answer clearly and directly.
- If the context is missing or not enough, say that you do not have enough information.
- Do not invent facts.
- If SQL/table context is provided, use it for structured data questions.
- If PDF context is provided, use it for policy/document questions.
""",
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
""",
            },
        ]
    )

    return {
        **state,
        "answer": response.content,
    }