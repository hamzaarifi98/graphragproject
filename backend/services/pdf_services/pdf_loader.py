from pathlib import Path

from pypdf import PdfReader

PDF_SUFFIX = ".pdf"


def extract_pdf_text(pdf_path: str | Path) -> str:
    path = Path(pdf_path).expanduser()
    reader = PdfReader(str(path))
    pages = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)

    return "\n".join(pages)


def get_pdf_paths(pdf_path: str | Path, recursive: bool = False) -> list[Path]:
    path = Path(pdf_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"PDF path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() != PDF_SUFFIX:
            raise ValueError(f"Expected a PDF file, got: {path}")
        return [path]

    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(
        pdf
        for pdf in path.glob(pattern)
        if pdf.is_file() and pdf.suffix.lower() == PDF_SUFFIX
    )


def load_pdf_dataset(
    pdf_path: str | Path,
    recursive: bool = False,
) -> list[dict[str, str]]:
    return [
        {
            "source_name": path.name,
            "source_path": str(path),
            "text_content": extract_pdf_text(path),
        }
        for path in get_pdf_paths(pdf_path, recursive=recursive)
    ]
