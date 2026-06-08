from sqlalchemy import text

from backend.core.pgvector import create_vector_table
from backend.core.postgre import engine


def format_embedding_for_pgvector(embedding: list[float]) -> str:
    return "[" + ",".join(str(value) for value in embedding) + "]"


def save_pdf_chunks(
    source_name: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> int:
    create_vector_table()

    with engine.begin() as connection:
        connection.execute(
            text("DELETE FROM document_chunks WHERE source_name = :source_name"),
            {"source_name": source_name},
        )

        for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            connection.execute(
                text("""
                    INSERT INTO document_chunks (
                        source_name,
                        source_type,
                        chunk_index,
                        content,
                        embedding
                    )
                    VALUES (
                        :source_name,
                        :source_type,
                        :chunk_index,
                        :content,
                        CAST(:embedding AS vector)
                    )
                """),
                {
                    "source_name": source_name,
                    "source_type": "pdf",
                    "chunk_index": chunk_index,
                    "content": chunk,
                    "embedding": format_embedding_for_pgvector(embedding),
                },
            )

    return len(chunks)


def create_olist_schema() -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS olist"))


def list_olist_tables() -> list[dict[str, int | str]]:
    with engine.connect() as connection:
        rows = connection.execute(
            text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'olist'
                ORDER BY table_name
            """)
        ).mappings()

        tables = []
        for row in rows:
            table_name = row["table_name"]
            count = connection.execute(
                text(f'SELECT COUNT(*) AS row_count FROM olist."{table_name}"')
            ).scalar_one()
            tables.append({"schema": "olist", "table_name": table_name, "row_count": count})

    return tables


def get_olist_rows(table_name: str, limit: int = 50) -> list[dict]:
    with engine.connect() as connection:
        rows = connection.execute(
            text(f'SELECT * FROM olist."{table_name}" LIMIT :limit'),
            {"limit": limit},
        ).mappings()

        return [dict(row) for row in rows]


def get_pdf_chunks(pdf_name: str) -> list[dict]:
    with engine.connect() as connection:
        rows = connection.execute(
            text("""
                SELECT id, source_name, chunk_index, content, created_at
                FROM document_chunks
                WHERE source_type = 'pdf'
                  AND source_name = :pdf_name
                ORDER BY chunk_index
            """),
            {"pdf_name": pdf_name},
        ).mappings()

        return [dict(row) for row in rows]
