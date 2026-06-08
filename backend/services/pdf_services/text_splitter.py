from backend.constants.constants import CHUNK_SIZE, CHUNK_OVERLAP


def split_text(text_content: str) -> list[str]:
    chunks = []
    start = 0

    while start < len(text_content):
        end = start + CHUNK_SIZE
        chunk = text_content[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - CHUNK_OVERLAP

    return chunks

