from pathlib import Path

from dotenv import load_dotenv

from backend.constants.constants import PDF_DATA_DIR
from backend.services.pdf_services.pdf_loader import extract_pdf_text
from backend.services.pdf_services.text_embedder import embed_text
from backend.services.pdf_services.text_splitter import split_text
from backend.services.pdf_services.vector_store import save_pdf_chunks
from backend.services.pdf_services.text_embedder import embed_documents

load_dotenv()


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


def main() -> list[dict[str, int | str]]:
    pdf_dir = Path(PDF_DATA_DIR)
    results = []

    for pdf_path in pdf_dir.glob("*.pdf"):
        results.append(ingest_pdf(pdf_path))

    return results


if __name__ == "__main__":
    main()
