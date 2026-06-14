import re
from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from backend.core.llm import get_chat_model
from backend.core.postgre import engine
from backend.schemas.router import QueryState
from backend.services.cache.retriever_cache import (
    SQL_RETRIEVER_CACHE_PREFIX,
    SQL_CACHE_TTL_SECONDS,
    get_cached_retriever_result,
    set_cached_retriever_result,
)
from backend.services.csv_services.sql_query_templates import find_sql_query_template
from backend.services.langgraph.retriever_state import merge_retriever_cache_hit

client = get_chat_model(model='gpt-5.4')


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
MAX_SQL_CONTEXT_ROWS = 50


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


def get_missing_olist_tables() -> list[str]:
    with engine.connect() as connection:
        rows = connection.execute(
            text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'olist'
            """)
        ).scalars()

        existing_tables = set(rows)

    return sorted(ALLOWED_TABLE_SET - existing_tables)


def ask_postgres_with_sql(question: str) -> dict:
    cached = get_cached_retriever_result(SQL_RETRIEVER_CACHE_PREFIX, question)

    if cached:
        return {
            **normalize_sql_result(cached),
            "cache_hit": True,
            "cache_type": "sql",
        }

    try:
        missing_tables = get_missing_olist_tables()
    except SQLAlchemyError as exc:
        return {
            "sql": None,
            "sql_source": None,
            "sql_template_hit": False,
            "sql_template_name": None,
            "sql_template_similarity": None,
            "rows": [],
            "error": f"Could not inspect structured tables: {exc}",
            "cache_hit": False,
            "cache_type": "sql",
        }

    if missing_tables:
        return {
            "sql": None,
            "sql_source": None,
            "sql_template_hit": False,
            "sql_template_name": None,
            "sql_template_similarity": None,
            "rows": [],
            "error": (
                "Structured data is not loaded. Run POST /structured/ingest "
                f"or POST /ingest first. Missing tables: {', '.join(missing_tables)}"
            ),
            "cache_hit": False,
            "cache_type": "sql",
        }

    generated = generate_sql_result(question)
    sql = generated.sql
    try:
        rows = run_sql(sql)
    except SQLAlchemyError as exc:
        return {
            "sql": sql,
            "sql_source": generated.source,
            "sql_template_hit": generated.source == "template",
            "sql_template_name": generated.template_name,
            "sql_template_similarity": generated.template_similarity,
            "rows": [],
            "error": f"Postgres query failed: {exc}",
            "cache_hit": False,
            "cache_type": "sql",
        }

    result = normalize_sql_result({
        "sql": sql,
        "sql_source": generated.source,
        "sql_template_hit": generated.source == "template",
        "sql_template_name": generated.template_name,
        "sql_template_similarity": generated.template_similarity,
        "rows": rows,
    })

    set_cached_retriever_result(
        SQL_RETRIEVER_CACHE_PREFIX,
        question,
        result,
        SQL_CACHE_TTL_SECONDS,
    )

    return {
        **result,
        "cache_hit": False,
        "cache_type": "sql",
    }


def normalize_sql_result(result: dict) -> dict:
    rows = result.get("rows", [])
    if isinstance(rows, list):
        rows = rows[:MAX_SQL_CONTEXT_ROWS]

    return {
        **result,
        "rows": rows,
        "row_count": (
            len(result.get("rows", []))
            if isinstance(result.get("rows"), list)
            else None
        ),
        "rows_truncated": (
            isinstance(result.get("rows"), list)
            and len(result.get("rows", [])) > MAX_SQL_CONTEXT_ROWS
        ),
    }


def postgres_retriever(state: QueryState) -> QueryState:
    sql_result = ask_postgres_with_sql(state["question"])

    return {
        **state,
        **merge_retriever_cache_hit(state, "sql", sql_result["cache_hit"]),
        "sql_context": str(sql_result),
        "sql": sql_result.get("sql"),
        "sql_source": sql_result.get("sql_source"),
        "sql_template_hit": sql_result.get("sql_template_hit", False),
        "sql_template_name": sql_result.get("sql_template_name"),
        "sql_template_similarity": sql_result.get("sql_template_similarity"),
    }
