import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sqlalchemy import text

from backend.core.postgre import engine
from backend.schemas.router import QueryState
from backend.services.cache.retriever_cache import (
    SQL_CACHE_TTL_SECONDS,
    get_cached_retriever_result,
    set_cached_retriever_result,
)

load_dotenv()

client = ChatOpenAI(
    model="gpt-5.4",
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


def generate_known_sql(question: str) -> str | None:
    normalized = " ".join(question.strip().lower().split())

    asks_best_seller = (
        "best seller" in normalized
        or "top seller" in normalized
        or "sold most" in normalized
        or "most sold" in normalized
    )
    asks_revenue = any(
        term in normalized
        for term in ["revenue", "sales", "value", "amount", "money"]
    )

    if asks_best_seller and asks_revenue:
        return """
            SELECT
                oi.seller_id,
                COUNT(DISTINCT oi.order_id) AS order_count,
                ROUND(SUM(oi.price)::numeric, 2) AS total_revenue
            FROM olist.order_items AS oi
            GROUP BY oi.seller_id
            ORDER BY total_revenue DESC
            LIMIT 1
        """

    if asks_best_seller:
        return """
            SELECT
                oi.seller_id,
                COUNT(DISTINCT oi.order_id) AS order_count,
                ROUND(SUM(oi.price)::numeric, 2) AS total_revenue
            FROM olist.order_items AS oi
            GROUP BY oi.seller_id
            ORDER BY order_count DESC
            LIMIT 1
        """

    return None


def generate_sql(question: str) -> str:
    known_sql = generate_known_sql(question)

    if known_sql:
        return known_sql.strip()

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
- In PostgreSQL, use ROUND(value::numeric, 2) when rounding averages or double precision values.
- Know that the date is from 2018 so when asked last month you can say last month from 2018 and use that date for filtering.

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


def fix_postgres_round(sql: str) -> str:
    return re.sub(
        r"ROUND\((AVG\([^)]+\))\s*,\s*(\d+)\)",
        r"ROUND(\1::numeric, \2)",
        sql,
        flags=re.IGNORECASE,
    )


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
    sql = fix_postgres_round(sql)
    validate_sql(sql)

    with engine.connect() as connection:
        rows = connection.execute(text(sql)).mappings()
        return [dict(row) for row in rows]


def ask_postgres_with_sql(question: str) -> dict:
    cached = get_cached_retriever_result("sql_retriever_cache", question)

    if cached:
        return {
            **cached,
            "cache_hit": True,
            "cache_type": "sql",
        }

    sql = generate_sql(question)
    rows = run_sql(sql)

    result = {
        "sql": sql,
        "rows": rows,
    }

    set_cached_retriever_result(
        "sql_retriever_cache",
        question,
        result,
        SQL_CACHE_TTL_SECONDS,
    )

    return {
        **result,
        "cache_hit": False,
        "cache_type": "sql",
    }


def postgres_retriever(state: QueryState) -> QueryState:
    sql_result = ask_postgres_with_sql(state["question"])
    retriever_cache_hits = {
        **state.get("retriever_cache_hits", {}),
        "sql": sql_result["cache_hit"],
    }
    retriever_cache_types = [
        cache_type
        for cache_type, cache_hit in retriever_cache_hits.items()
        if cache_hit
    ]

    return {
        **state,
        "sql_context": str(sql_result),
        "retriever_cache_hit": any(retriever_cache_hits.values()),
        "retriever_cache_type": ", ".join(retriever_cache_types) or None,
        "retriever_cache_hits": retriever_cache_hits,
    }
