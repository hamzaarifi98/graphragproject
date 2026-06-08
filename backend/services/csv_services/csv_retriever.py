import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sqlalchemy import text

from backend.core.postgre import engine
from backend.schemas.router import QueryState

load_dotenv()

client = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


ALLOWED_TABLES = [
    "customers",
    "geolocation",
    "order_items",
    "order_payments",
    "order_reviews",
    "orders",
    "products",
    "sellers",
    "product_category_translation",
]

ALLOWED_TABLE_SET = set(ALLOWED_TABLES)


def get_database_schema() -> str:
    with engine.connect() as connection:
        rows = connection.execute(
            text("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'olist'
                ORDER BY table_name, ordinal_position
            """)
        ).mappings()

        schema_lines = []
        for row in rows:
            schema_lines.append(
                f"olist.{row['table_name']}.{row['column_name']} ({row['data_type']})"
            )

    return "\n".join(schema_lines)


def generate_sql(question: str) -> str:
    schema = get_database_schema()

    response = client.invoke(
        [
            {
                "role": "system",
                "content": f"""
You are a PostgreSQL SQL generator.

Generate one safe SELECT query for the user's question.

Rules:
- Use only the olist schema.
- Use only these tables: {", ".join(ALLOWED_TABLES)}
- Return only SQL, no explanation.
- Do not use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE.
- Always add LIMIT 50 unless the query uses aggregation and returns a small result.
- Prefer clear column aliases.

Database schema:
{schema}
""",
            },
            {
                "role": "user",
                "content": question,
            },
        ]
    )

    sql = response.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()

    return sql


def validate_sql(sql: str) -> None:
    normalized = sql.lower().strip()

    if not normalized.startswith("select"):
        raise ValueError("Only SELECT queries are allowed")

    blocked_keywords = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "truncate",
        "grant",
        "revoke",
    ]

    for keyword in blocked_keywords:
        if re.search(rf"\b{keyword}\b", normalized):
            raise ValueError(f"Blocked SQL keyword: {keyword}")

    if "olist." not in normalized:
        raise ValueError("Query must use the olist schema")

    referenced_tables = set(
        re.findall(r"\bolist\.([a-z_][a-z0-9_]*)\b", normalized)
    )
    unknown_tables = referenced_tables - ALLOWED_TABLE_SET

    if unknown_tables:
        raise ValueError(
            f"Query references tables that are not allowed: {', '.join(sorted(unknown_tables))}"
        )


def run_sql(sql: str) -> list[dict]:
    validate_sql(sql)

    with engine.connect() as connection:
        rows = connection.execute(text(sql)).mappings()
        return [dict(row) for row in rows]


def ask_postgres_with_sql(question: str) -> dict:
    sql = generate_sql(question)
    rows = run_sql(sql)

    return {
        "sql": sql,
        "rows": rows,
    }


def postgres_retriever(state: QueryState) -> QueryState:
    sql_result = ask_postgres_with_sql(state["question"])

    return {
        **state,
        "sql_context": str(sql_result),
    }
