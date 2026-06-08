from backend.services.csv_services import csv_to_neon
from backend.services.pdf_services import pdf_to_neon


def main() -> dict[str, list[dict[str, int | str]]]:
    return {
        "structured": csv_to_neon.main(),
        "pdfs": pdf_to_neon.main(),
    }


if __name__ == "__main__":
    main()
