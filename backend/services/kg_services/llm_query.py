import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from neo4j.exceptions import ClientError

from backend.services.kg_services.kg_query import run_cypher

load_dotenv()

client = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0,
)


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


TEMPORAL_PROPERTIES = [
    "purchase_timestamp",
    "approved_at",
    "delivered_customer_date",
    "estimated_delivery_date",
    "created_at",
]

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


def generate_cypher(question: str, previous_cypher: str | None = None, error: str | None = None) -> str:
    repair_context = ""
    if previous_cypher and error:
        repair_context = f"""
The previous Cypher failed.

Previous Cypher:
{previous_cypher}

Neo4j error:
{error}

Return a corrected query.
"""

    prompt = f"""
You are a Neo4j Cypher expert.

Use this graph schema:

{GRAPH_SCHEMA}

Write one read-only Cypher query for this user question:
{question}

{repair_context}

Rules:
- Only return Cypher.
- Do not explain.
- Remember that data is from 2018, so "last month" means last month from 2018 and should be calculated accordingly.
- Only use MATCH, OPTIONAL MATCH, WHERE, RETURN, ORDER BY, LIMIT, WITH.
- Do not write CREATE, MERGE, DELETE, SET, REMOVE, DROP, LOAD, or CALL.
- Order date and review date properties are stored as strings like "2017-10-10 21:25:13".
- When comparing or calculating durations with date/time properties, convert them first:
  datetime(replace(o.delivered_customer_date, " ", "T"))
- If its asked whats the top sold product category in last month you check the product category.  
- Never pass raw string date properties directly into duration.between().
- For delayed delivery, compare datetime(replace(o.delivered_customer_date, " ", "T")) > datetime(replace(o.estimated_delivery_date, " ", "T")).
- For relative windows, use Neo4j temporal arithmetic like datetime() - duration({{months: 1}}).
- Do not use epochMillis, toMillis(), milliseconds, or numeric subtraction for date filtering.
"""

    response = client.invoke(
        [
            {"role": "system", "content": "You generate safe read-only Neo4j Cypher queries."},
            {"role": "user", "content": prompt},
        ]
    )

    cypher = response.content.strip()
    cypher = cypher.replace("```cypher", "").replace("```", "").strip()
    return normalize_temporal_cypher(cypher)


def normalize_temporal_cypher(cypher: str) -> str:
    cypher = normalize_relative_datetime_cypher(cypher)

    for prop in TEMPORAL_PROPERTIES:
        pattern = rf"(?<!replace\()(?<!datetime\()\b([a-zA-Z_][a-zA-Z0-9_]*)\.{prop}\b"
        cypher = re.sub(
            pattern,
            lambda match: _temporal_reference(match.group(0)),
            cypher,
        )

    return cypher


def normalize_relative_datetime_cypher(cypher: str) -> str:
    replacements = [
        (
            r"datetime\(\)\.epochMillis\s*-\s*duration\(['\"]P(\d+)M['\"]\)\.toMillis\(\)",
            r"datetime() - duration({months: \1})",
        ),
        (
            r"datetime\(\)\.epochMillis\s*-\s*duration\(['\"]P(\d+)D['\"]\)\.toMillis\(\)",
            r"datetime() - duration({days: \1})",
        ),
        (
            r"datetime\(\)\.epochMillis\s*-\s*duration\(['\"]P(\d+)Y['\"]\)\.toMillis\(\)",
            r"datetime() - duration({years: \1})",
        ),
    ]

    for pattern, replacement in replacements:
        cypher = re.sub(pattern, replacement, cypher, flags=re.IGNORECASE)

    return cypher


def _temporal_reference(reference: str) -> str:
    if f"replace({reference}" in reference or f"datetime({reference}" in reference:
        return reference

    return f"datetime(replace({reference}, ' ', 'T'))"


def validate_cypher(cypher: str) -> None:
    upper = cypher.upper()

    if not re.match(r"^\s*(MATCH|OPTIONAL MATCH)\b", upper):
        raise ValueError("Only read-only MATCH queries are allowed.")

    for word in BLOCKED_WORDS:
        if re.search(rf"\b{word}\b", upper):
            raise ValueError(f"Blocked unsafe Cypher keyword: {word}")

    blocked_temporal_patterns = ["EPOCHMILLIS", "TOMILLIS", "MILLISECOND"]
    for pattern in blocked_temporal_patterns:
        if pattern in upper:
            raise ValueError(f"Blocked invalid temporal expression: {pattern}")


def summarize_answer(question: str, rows: list[dict]) -> str:
    response = client.invoke(
        [
            {
                "role": "system",
                "content": "You explain Neo4j query results in simple business language.",
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nRows: {rows[:50]}\n\nAnswer clearly and briefly.",
            },
        ]
    )

    return response.content.strip()


def ask_graph(question: str) -> dict:
    cypher = generate_cypher(question)
    validate_cypher(cypher)

    try:
        rows = run_cypher(cypher)
    except ClientError as exc:
        first_error = str(exc)
        try:
            cypher = generate_cypher(question, previous_cypher=cypher, error=first_error)
            validate_cypher(cypher)
            rows = run_cypher(cypher)
        except (ClientError, ValueError) as retry_exc:
            return {
                "question": question,
                "cypher": cypher,
                "rows": [],
                "answer": "I could not run a valid Neo4j query for that question.",
                "error": f"Neo4j query failed after retry: {retry_exc}",
            }

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
