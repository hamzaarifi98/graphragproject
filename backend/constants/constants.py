import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]

PROJECT_NAME = "Internal RAG Assistant"

PDF_DATA_DIR = str(ROOT_DIR / "data/raw/unstructured")
STRUCTURED_DATA_DIR = str(ROOT_DIR / "data/raw/structured")
PROCESSED_DATA_DIR = str(ROOT_DIR / "data/processed")
ARTIFACTS_DIR = str(ROOT_DIR / "artifacts")

VECTOR_STORE_DIR = str(ROOT_DIR / "artifacts/vector_store")
DATABASE_URL = f"sqlite:///{ROOT_DIR / 'artifacts/internal_data.db'}"
COLLECTION_NAME = "internal_documents"

LOGS_DIR = str(ROOT_DIR / "artifacts/logs")
REPORTS_DIR = str(ROOT_DIR / "artifacts/reports")

CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
TOP_K = 4
RERANK_CANDIDATE_MULTIPLIER = 4
RERANK_LEXICAL_WEIGHT = 0.6
RERANK_VECTOR_WEIGHT = 0.4

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-nano")
OPEN_SOURCE_EMBEDDING_MODEL = os.getenv(
    "OPEN_SOURCE_EMBEDDING_MODEL",
    "qwen3-embedding:0.6b",
)
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
DEFAULT_OPEN_SOURCE_EMBEDDING_DIMENSION = (
    "1024"
    if OPEN_SOURCE_EMBEDDING_MODEL.startswith("qwen3-embedding:")
    else "384"
)
DEFAULT_EMBEDDING_DIMENSION = (
    DEFAULT_OPEN_SOURCE_EMBEDDING_DIMENSION
    if EMBEDDING_PROVIDER == "open_source"
    else "1536"
)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", DEFAULT_EMBEDDING_DIMENSION))
