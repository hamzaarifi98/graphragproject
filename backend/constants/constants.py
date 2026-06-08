from pathlib import Path


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

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5


