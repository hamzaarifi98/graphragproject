from pathlib import Path
import sys

PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.constants.constants import PDF_DATA_DIR
from backend.services.pdf_services.pdf_loader import extract_pdf_text, get_pdf_paths
from backend.services.pdf_services.text_embedder import embed_documents
from backend.services.pdf_services.text_splitter import split_text
from backend.services.pdf_services.vector_store import save_pdf_chunks


def ingest_pdf(pdf_path: Path) -> dict[str, int | str]:
    text_content = extract_pdf_text(pdf_path)
    chunks = split_text(text_content)
    embeddings = embed_documents(chunks)
    chunks_loaded = save_pdf_chunks(pdf_path.name, chunks, embeddings)

    result = {
        "source_name": pdf_path.name,
        "source_type": "pdf",
        "chunks_loaded": chunks_loaded,
    }

    print(f"Loaded {chunks_loaded} chunks from {pdf_path.name}")
    return result


def ingest_pdf_directory(
    pdf_dir: str | Path = PDF_DATA_DIR,
    recursive: bool = False,
) -> list[dict[str, int | str]]:
    results = []

    for pdf_path in get_pdf_paths(pdf_dir, recursive=recursive):
        results.append(ingest_pdf(pdf_path))

    return results


def main() -> list[dict[str, int | str]]:
    return ingest_pdf_directory()


if __name__ == "__main__":
    main()
