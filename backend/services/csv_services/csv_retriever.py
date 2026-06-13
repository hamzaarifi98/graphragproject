import re
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sqlalchemy import text

from backend.constants.constants import OPENAI_CHAT_MODEL
from backend.core.postgre import engine
from backend.schemas.router import QueryState
from backend.services.cache.retriever_cache import (
    SQL_CACHE_TTL_SECONDS,
    get_cached_retriever_result,
    set_cached_retriever_result,
)
from backend.services.csv_services.sql_query_templates import find_sql_query_template

load_dotenv()


client = ChatOpenAI(
    model=OPENAI_CHAT_MODEL,
    temperature=0,
)


ALLOWED_TABLES = (
    "customers",
    "geolocation",
    "order_items",
    "order_payments",
    "order_reviews",
    "orders",
    "products",
    "sellers",
    "product_category_translation",
)

ALLOWED_TABLE_SET = set(ALLOWED_TABLES)
BLOCKED_SQL_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
)
SQL_FENCE_PATTERN = re.compile(r"```(?:sql)?|```", re.IGNORECASE)
AVG_ROUND_PATTERN = re.compile(
    r"ROUND\((AVG\([^)]+\))\s*,\s*(\d+)\)",
    re.IGNORECASE,
)
TABLE_REFERENCE_PATTERN = re.compile(r"\bolist\.([a-z_][a-z0-9_]*)\b")


@dataclass(frozen=True)
class GeneratedSql:
    sql: str
    source: str
    template_name: str | None = None
    template_similarity: float | None = None


@lru_cache(maxsize=1)
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
    return generate_sql_result(question).sql


def generate_sql_result(question: str) -> GeneratedSql:
    template_match = find_sql_query_template(question)

    if template_match:
        return GeneratedSql(
            sql=template_match.template.sql.strip(),
            source="template",
            template_name=template_match.template.name,
            template_similarity=round(template_match.similarity, 4),
        )

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
    sql = SQL_FENCE_PATTERN.sub("", sql).strip()

    return GeneratedSql(sql=sql, source="llm")


def fix_postgres_round(sql: str) -> str:
    return AVG_ROUND_PATTERN.sub(r"ROUND(\1::numeric, \2)", sql)


def validate_sql(sql: str) -> None:
    normalized = sql.lower().strip()

    if not normalized.startswith("select"):
        raise ValueError("Only SELECT queries are allowed")

    for keyword in BLOCKED_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", normalized):
            raise ValueError(f"Blocked SQL keyword: {keyword}")

    if "olist." not in normalized:
        raise ValueError("Query must use the olist schema")

    referenced_tables = set(TABLE_REFERENCE_PATTERN.findall(normalized))
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

    generated = generate_sql_result(question)
    sql = generated.sql
    rows = run_sql(sql)

    result = {
        "sql": sql,
        "sql_source": generated.source,
        "sql_template_hit": generated.source == "template",
        "sql_template_name": generated.template_name,
        "sql_template_similarity": generated.template_similarity,
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
        "sql_source": sql_result.get("sql_source"),
        "sql_template_hit": sql_result.get("sql_template_hit", False),
        "sql_template_name": sql_result.get("sql_template_name"),
        "sql_template_similarity": sql_result.get("sql_template_similarity"),
        "retriever_cache_hit": any(retriever_cache_hits.values()),
        "retriever_cache_type": ", ".join(retriever_cache_types) or None,
        "retriever_cache_hits": retriever_cache_hits,
    }



def main():
    return ask_postgres_with_sql()

if __name__ == "__main__":
    main()
