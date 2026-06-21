from pathlib import Path

from backend.config import settings

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

OPENAI_EMBEDDING_MODEL = settings.OPENAI_EMBEDDING_MODEL
OPENAI_CHAT_MODEL = settings.OPENAI_CHAT_MODEL
EMBEDDING_DIMENSION = settings.EMBEDDING_DIMENSION
