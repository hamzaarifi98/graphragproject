from langchain_openai import ChatOpenAI

from backend.constants.constants import OPENAI_CHAT_MODEL


def get_chat_model(model: str | None = None, temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or OPENAI_CHAT_MODEL,
        temperature=temperature,
    )
