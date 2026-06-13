import re
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from neo4j.exceptions import ClientError

from backend.services.kg_services.kg_query import run_cypher
from backend.services.kg_services.kg_query_templates import find_kg_query_template

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


TEMPORAL_PROPERTIES = (
    "purchase_timestamp",
    "approved_at",
    "delivered_customer_date",
    "estimated_delivery_date",
    "created_at",
)

BLOCKED_WORDS = (
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "LOAD",
    "CALL",
)
BLOCKED_TEMPORAL_PATTERNS = ("EPOCHMILLIS", "TOMILLIS", "MILLISECOND")
READ_ONLY_CYPHER_PATTERN = re.compile(r"^\s*(MATCH|OPTIONAL MATCH)\b", re.IGNORECASE)
CODE_FENCE_PATTERN = re.compile(r"```(?:cypher)?|```", re.IGNORECASE)
BLOCKED_WORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in BLOCKED_WORDS) + r")\b",
    re.IGNORECASE,
)
RELATIVE_DATETIME_REPLACEMENTS = (
    (
        re.compile(
            r"datetime\(\)\.epochMillis\s*-\s*"
            r"duration\(['\"]P(\d+)M['\"]\)\.toMillis\(\)",
            re.IGNORECASE,
        ),
        r"datetime() - duration({months: \1})",
    ),
    (
        re.compile(
            r"datetime\(\)\.epochMillis\s*-\s*"
            r"duration\(['\"]P(\d+)D['\"]\)\.toMillis\(\)",
            re.IGNORECASE,
        ),
        r"datetime() - duration({days: \1})",
    ),
    (
        re.compile(
            r"datetime\(\)\.epochMillis\s*-\s*"
            r"duration\(['\"]P(\d+)Y['\"]\)\.toMillis\(\)",
            re.IGNORECASE,
        ),
        r"datetime() - duration({years: \1})",
    ),
)
TEMPORAL_REFERENCE_PATTERNS = tuple(
    re.compile(
        rf"(?<!replace\()(?<!datetime\()\b([a-zA-Z_][a-zA-Z0-9_]*)\.{prop}\b"
    )
    for prop in TEMPORAL_PROPERTIES
)


@dataclass(frozen=True)
class GeneratedCypher:
    cypher: str
    source: str
    template_name: str | None = None
    template_similarity: float | None = None


def generate_cypher(
    question: str,
    previous_cypher: str | None = None,
    error: str | None = None,
) -> str:
    return generate_cypher_result(question, previous_cypher, error).cypher


def generate_cypher_result(
    question: str,
    previous_cypher: str | None = None,
    error: str | None = None,
) -> GeneratedCypher:
    template_match = None
    if not previous_cypher and not error:
        template_match = find_kg_query_template(question)

    if template_match:
        return GeneratedCypher(
            cypher=template_match.template.cypher.strip(),
            source="template",
            template_name=template_match.template.name,
            template_similarity=round(template_match.similarity, 4),
        )

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
            {
                "role": "system",
                "content": "You generate safe read-only Neo4j Cypher queries.",
            },
            {"role": "user", "content": prompt},
        ]
    )

    cypher = response.content.strip()
    cypher = CODE_FENCE_PATTERN.sub("", cypher).strip()
    return GeneratedCypher(
        cypher=normalize_temporal_cypher(cypher),
        source="llm_repair" if previous_cypher and error else "llm",
    )


def normalize_temporal_cypher(cypher: str) -> str:
    cypher = normalize_relative_datetime_cypher(cypher)

    for pattern in TEMPORAL_REFERENCE_PATTERNS:
        cypher = pattern.sub(lambda match: _temporal_reference(match.group(0)), cypher)

    return cypher


def normalize_relative_datetime_cypher(cypher: str) -> str:
    for pattern, replacement in RELATIVE_DATETIME_REPLACEMENTS:
        cypher = pattern.sub(replacement, cypher)

    return cypher


def _temporal_reference(reference: str) -> str:
    if f"replace({reference}" in reference or f"datetime({reference}" in reference:
        return reference

    return f"datetime(replace({reference}, ' ', 'T'))"


def validate_cypher(cypher: str) -> None:
    upper = cypher.upper()

    if not READ_ONLY_CYPHER_PATTERN.match(cypher):
        raise ValueError("Only read-only MATCH queries are allowed.")

    blocked_word_match = BLOCKED_WORD_PATTERN.search(cypher)
    if blocked_word_match:
        raise ValueError(
            f"Blocked unsafe Cypher keyword: {blocked_word_match.group(1).upper()}"
        )

    for pattern in BLOCKED_TEMPORAL_PATTERNS:
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
                "content": (
                    f"Question: {question}\n\n"
                    f"Rows: {rows[:50]}\n\n"
                    "Answer clearly and briefly."
                ),
            },
        ]
    )

    return response.content.strip()


def ask_graph(question: str) -> dict:
    generated = generate_cypher_result(question)
    cypher = generated.cypher
    validate_cypher(cypher)

    try:
        rows = run_cypher(cypher)
    except ClientError as exc:
        first_error = str(exc)
        try:
            generated = generate_cypher_result(
                question,
                previous_cypher=cypher,
                error=first_error,
            )
            cypher = generated.cypher
            validate_cypher(cypher)
            rows = run_cypher(cypher)
        except (ClientError, ValueError) as retry_exc:
            return {
                "question": question,
                "cypher": cypher,
                "cypher_source": generated.source,
                "kg_template_hit": generated.source == "template",
                "kg_template_name": generated.template_name,
                "kg_template_similarity": generated.template_similarity,
                "rows": [],
                "answer": "I could not run a valid Neo4j query for that question.",
                "error": f"Neo4j query failed after retry: {retry_exc}",
            }

    answer = summarize_answer(question, rows)

    return {
        "question": question,
        "cypher": cypher,
        "cypher_source": generated.source,
        "kg_template_hit": generated.source == "template",
        "kg_template_name": generated.template_name,
        "kg_template_similarity": generated.template_similarity,
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
