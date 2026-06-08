from sqlalchemy import text

from backend.core.postgre import engine
from backend.schemas.router import QueryState
from backend.services.pdf_services.text_embedder import embed_text
from backend.services.pdf_services.vector_store import format_embedding_for_pgvector


def retrieve_pdf_chunks(question: str, top_k: int = 5) -> list[dict]:
    question_embedding = embed_text(question)
    embedding_text = format_embedding_for_pgvector(question_embedding)

    with engine.connect() as connection:
        rows = connection.execute(
            text("""
                SELECT
                    id,
                    source_name,
                    chunk_index,
                    content,
                    embedding <-> CAST(:embedding AS vector) AS distance
                FROM document_chunks
                WHERE source_type = 'pdf'
                ORDER BY embedding <-> CAST(:embedding AS vector)
                LIMIT :top_k
            """),
            {
                "embedding": embedding_text,
                "top_k": top_k,
            },
        ).mappings()

        return [dict(row) for row in rows]
    


def pdf_retriever(state: QueryState) -> QueryState:
    chunks = retrieve_pdf_chunks(state["question"])

    context = "\n\n".join(
        chunk["content"] for chunk in chunks
    )

    return {
        **state,
        "pdf_context": context,
    }