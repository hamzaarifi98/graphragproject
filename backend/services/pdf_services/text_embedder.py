from openai import OpenAI


client = OpenAI()


def embed_text(text_content: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text_content,
    )

    return response.data[0].embedding
