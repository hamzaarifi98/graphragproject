import re

from sqlalchemy import text

from backend.constants.constants import (
    RERANK_CANDIDATE_MULTIPLIER,
    RERANK_LEXICAL_WEIGHT,
    RERANK_VECTOR_WEIGHT,
    TOP_K,
)
from backend.core.postgre import engine
from backend.schemas.router import QueryState
from backend.services.pdf_services.vector_store import format_embedding_for_pgvector

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
MIN_TOKEN_LENGTH = 3
PDF_CANDIDATE_QUERY = text("""
    SELECT
        id,
        source_name,
        chunk_index,
        content,
        embedding <-> CAST(:embedding AS vector) AS distance
    FROM document_chunks
    WHERE source_type = 'pdf'
    ORDER BY embedding <-> CAST(:embedding AS vector)
    LIMIT :candidate_limit
""")


def tokenize(text_content: str) -> set[str]:
    tokens = TOKEN_PATTERN.findall(text_content.lower())
    return {token for token in tokens if len(token) >= MIN_TOKEN_LENGTH}


def calculate_candidate_limit(top_k: int) -> int:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    return max(top_k, top_k * RERANK_CANDIDATE_MULTIPLIER)


def lexical_overlap_score(question_tokens: set[str], content: str) -> float:
    if not question_tokens:
        return 0.0

    content_tokens = tokenize(content)
    return len(question_tokens & content_tokens) / len(question_tokens)


def vector_similarity_score(distance: float | None) -> float:
    return 1 / (1 + float(distance or 0))


def score_chunk(question_tokens: set[str], chunk: dict) -> dict:
    lexical_score = lexical_overlap_score(
        question_tokens,
        chunk.get("content", ""),
    )
    vector_score = vector_similarity_score(chunk.get("distance"))
    rerank_score = (
        RERANK_LEXICAL_WEIGHT * lexical_score
        + RERANK_VECTOR_WEIGHT * vector_score
    )

    return {
        **chunk,
        "lexical_score": lexical_score,
        "vector_score": vector_score,
        "rerank_score": rerank_score,
    }


def rerank_pdf_chunks(question: str, chunks: list[dict], top_k: int) -> list[dict]:
    if not chunks:
        return []

    question_tokens = tokenize(question)

    if not question_tokens:
        return chunks[:top_k]

    reranked_chunks = [score_chunk(question_tokens, chunk) for chunk in chunks]

    return sorted(
        reranked_chunks,
        key=lambda chunk: chunk["rerank_score"],
        reverse=True,
    )[:top_k]


def fetch_pdf_candidates(question: str, candidate_limit: int) -> list[dict]:
    from backend.services.pdf_services.text_embedder import embed_text

    question_embedding = embed_text(question)
    embedding_text = format_embedding_for_pgvector(question_embedding)

    with engine.connect() as connection:
        rows = connection.execute(
            PDF_CANDIDATE_QUERY,
            {
                "embedding": embedding_text,
                "candidate_limit": candidate_limit,
            },
        ).mappings()

        return [dict(row) for row in rows]


def retrieve_pdf_chunks(question: str, top_k: int = TOP_K) -> list[dict]:
    if not question.strip():
        return []

    chunks = fetch_pdf_candidates(
        question=question,
        candidate_limit=calculate_candidate_limit(top_k),
    )

    return rerank_pdf_chunks(question, chunks, top_k)


def pdf_retriever(state: QueryState) -> QueryState:
    chunks = retrieve_pdf_chunks(state["question"], top_k=TOP_K)
    context = "\n\n".join(chunk["content"] for chunk in chunks)

    return {
        **state,
        "pdf_context": context,
    }
