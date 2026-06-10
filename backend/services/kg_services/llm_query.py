import re
from openai import OpenAI

from backend.services.kg_services.kg_query import run_cypher

client = OpenAI()


GRAPH_SCHEMA = """
Node labels:
- Customer(id)
- Order(id, status, purchase_timestamp, approved_at, delivered_customer_date, estimated_delivery_date)
- Product(id)
- Seller(id)
- Payment(id, type, installments, value)
- Review(id, score, title, message, created_at)
- Category(name)

Relationships:
- (:Customer)-[:PLACED]->(:Order)
- (:Order)-[:CONTAINS {item_id, price, freight_value}]->(:Product)
- (:Order)-[:SOLD_BY]->(:Seller)
- (:Product)-[:IN_CATEGORY]->(:Category)
- (:Order)-[:HAS_PAYMENT]->(:Payment)
- (:Order)-[:HAS_REVIEW]->(:Review)
"""


BLOCKED_WORDS = [
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "LOAD",
    "CALL",
]


def generate_cypher(question: str) -> str:
    prompt = f"""
You are a Neo4j Cypher expert.

Use this graph schema:

{GRAPH_SCHEMA}

Write one read-only Cypher query for this user question:
{question}

Rules:
- Only return Cypher.
- Do not explain.
- Only use MATCH, OPTIONAL MATCH, WHERE, RETURN, ORDER BY, LIMIT, WITH.
- Do not write CREATE, MERGE, DELETE, SET, REMOVE, DROP, LOAD, or CALL.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You generate safe read-only Neo4j Cypher queries."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    cypher = response.choices[0].message.content.strip()
    cypher = cypher.replace("```cypher", "").replace("```", "").strip()
    return cypher


def validate_cypher(cypher: str) -> None:
    upper = cypher.upper()

    if not re.match(r"^\s*(MATCH|OPTIONAL MATCH)\b", upper):
        raise ValueError("Only read-only MATCH queries are allowed.")

    for word in BLOCKED_WORDS:
        if re.search(rf"\b{word}\b", upper):
            raise ValueError(f"Blocked unsafe Cypher keyword: {word}")


def summarize_answer(question: str, rows: list[dict]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You explain Neo4j query results in simple business language.",
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nRows: {rows[:50]}\n\nAnswer clearly and briefly.",
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def ask_graph(question: str) -> dict:
    cypher = generate_cypher(question)
    validate_cypher(cypher)

    rows = run_cypher(cypher)
    answer = summarize_answer(question, rows)

    return {
        "question": question,
        "cypher": cypher,
        "rows": rows,
        "answer": answer,
    }


if __name__ == "__main__":
    result = ask_graph("What are the top 10 product categories by number of orders?")
    print("Cypher:")
    print(result["cypher"])
    print()
    print("Answer:")
    print(result["answer"])