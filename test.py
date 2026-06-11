import time

from backend.services.pdf_services.text_embedder import embed_text

question = "What is the refund policy?"

start = time.perf_counter()
embedding_1 = embed_text(question)
first_duration = time.perf_counter() - start

start = time.perf_counter()
embedding_2 = embed_text(question)
second_duration = time.perf_counter() - start

print("First embedding length:", len(embedding_1))
print("Second embedding length:", len(embedding_2))
print("Same embedding:", embedding_1 == embedding_2)
print("First call seconds:", round(first_duration, 3))
print("Second call seconds:", round(second_duration, 3))