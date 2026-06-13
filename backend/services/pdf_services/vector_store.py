import re

from sqlalchemy import text

from backend.core.pgvector import create_vector_table
from backend.core.postgre import engine

PDF_CHUNK_INSERT_QUERY = text("""
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
""")
VALID_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def format_embedding_for_pgvector(embedding: list[float]) -> str:
    return "[" + ",".join(format(value, ".17g") for value in embedding) + "]"


def quote_identifier(identifier: str) -> str:
    if not VALID_IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier}")

    return f'"{identifier}"'


def save_pdf_chunks(
    source_name: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> int:
    if len(chunks) != len(embeddings):
        raise ValueError(
            "Chunks and embeddings must have the same length. "
            f"Got {len(chunks)} chunks and {len(embeddings)} embeddings."
        )

    create_vector_table()
    chunk_rows = [
        {
            "source_name": source_name,
            "source_type": "pdf",
            "chunk_index": chunk_index,
            "content": chunk,
            "embedding": format_embedding_for_pgvector(embedding),
        }
        for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    with engine.begin() as connection:
        connection.execute(
            text("DELETE FROM document_chunks WHERE source_name = :source_name"),
            {"source_name": source_name},
        )

        if chunk_rows:
            connection.execute(PDF_CHUNK_INSERT_QUERY, chunk_rows)

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
            safe_table_name = quote_identifier(table_name)
            count = connection.execute(
                text(f"SELECT COUNT(*) AS row_count FROM olist.{safe_table_name}")
            ).scalar_one()
            tables.append(
                {
                    "schema": "olist",
                    "table_name": table_name,
                    "row_count": count,
                }
            )

    return tables


def get_olist_rows(table_name: str, limit: int = 50) -> list[dict]:
    safe_table_name = quote_identifier(table_name)

    with engine.connect() as connection:
        rows = connection.execute(
            text(f"SELECT * FROM olist.{safe_table_name} LIMIT :limit"),
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
