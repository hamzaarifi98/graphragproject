from pathlib import Path
import sys

from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.constants.constants import EMBEDDING_DIMENSION
from backend.core.postgre import engine


def create_vector_table() -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        connection.execute(text(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id BIGSERIAL PRIMARY KEY,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR({EMBEDDING_DIMENSION}),
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """))

        current_dimension = connection.execute(text("""
            SELECT atttypmod - 4
            FROM pg_attribute
            WHERE attrelid = 'document_chunks'::regclass
              AND attname = 'embedding'
              AND NOT attisdropped
        """)).scalar_one()

        if current_dimension != EMBEDDING_DIMENSION:
            connection.execute(text("TRUNCATE TABLE document_chunks"))
            connection.execute(text(f"""
                ALTER TABLE document_chunks
                ALTER COLUMN embedding TYPE VECTOR({EMBEDDING_DIMENSION})
            """))
