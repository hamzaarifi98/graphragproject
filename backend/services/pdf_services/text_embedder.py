import os

from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import EncoderBackedStore
from langchain_community.storage import RedisStore
from langchain_core.utils.iter import batch_iterate
from langchain_openai import OpenAIEmbeddings

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

base_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

redis_store = RedisStore(
    redis_url=REDIS_URL,
    namespace=f"embeddings:{EMBEDDING_MODEL}",
)

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=base_embeddings,
    document_embedding_cache=redis_store,
    namespace=EMBEDDING_MODEL,
)


def embed_text(text_content: str) -> list[float]:
    return cached_embeddings.embed_query(text_content)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return cached_embeddings.embed_documents(texts)