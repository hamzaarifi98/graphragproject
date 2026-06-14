import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_community.storage import RedisStore
from langchain_openai import OpenAIEmbeddings

try:
    from langchain.embeddings import CacheBackedEmbeddings
except (ImportError, ModuleNotFoundError):
    try:
        from langchain_classic.embeddings.cache import CacheBackedEmbeddings
    except (ImportError, ModuleNotFoundError):
        from langchain.embeddings import CacheBackedEmbeddings

from backend.constants.constants import (
    OPENAI_EMBEDDING_MODEL,
)

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
EMBEDDING_CACHE_NAMESPACE = f"embeddings:openai:{OPENAI_EMBEDDING_MODEL}"


def create_base_embeddings() -> Any:
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_embeddings() -> Any:
    redis_store = RedisStore(
        redis_url=REDIS_URL,
        namespace=EMBEDDING_CACHE_NAMESPACE,
    )

    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=create_base_embeddings(),
        document_embedding_cache=redis_store,
        namespace=EMBEDDING_CACHE_NAMESPACE,
    )


def embed_text(text_content: str) -> list[float]:
    return get_embeddings().embed_query(text_content)


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    return get_embeddings().embed_documents(texts)
