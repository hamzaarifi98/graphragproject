from functools import lru_cache

from backend.constants.constants import CHUNK_OVERLAP, CHUNK_SIZE

TEXT_SEPARATORS = [
    "\n\n",
    "\n",
    " ",
    ".",
    ",",
    "\u200b",
    "\uff0c",
    "\u3001",
    "\uff0e",
    "\u3002",
    "",
]


@lru_cache(maxsize=1)
def get_text_splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=TEXT_SEPARATORS,
    )


def split_text(text_content: str) -> list[str]:
    if not text_content:
        return []

    return [
        chunk.strip()
        for chunk in get_text_splitter().split_text(text_content)
        if chunk.strip()
    ]
