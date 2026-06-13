import math
import os
from dataclasses import dataclass
from functools import lru_cache
from textwrap import dedent

from dotenv import load_dotenv

from backend.services.pdf_services.text_embedder import embed_text

load_dotenv()

SQL_QUERY_TEMPLATE_THRESHOLD = float(
    os.getenv("SQL_QUERY_TEMPLATE_THRESHOLD", "0.88")
)
SQL_QUERY_TEMPLATE_KEYWORD_BOOST = float(
    os.getenv("SQL_QUERY_TEMPLATE_KEYWORD_BOOST", "0.08")
)
MAX_TEMPLATE_SCORE = 1.0

QUESTION_REPLACEMENTS = {
    "best seller": "seller sold most",
    "top seller": "seller sold most",
    "highest seller": "seller sold most",
    "sold most": "seller sold most",
    "most sold": "seller sold most",
    "most sales": "seller sold most",
    "sales": "revenue",
    "money": "revenue",
    "amount": "revenue",
    "value": "revenue",
    "shipping time": "delivery time",
    "delivered orders": "shipped orders",
}
BOOST_INTENTS = (
    frozenset(("seller", "sold", "most")),
    frozenset(("category", "revenue")),
)


def sql_template(sql: str) -> str:
    return dedent(sql).strip()


SELLER_SALES_SQL = sql_template("""
    SELECT
        oi.seller_id,
        COUNT(DISTINCT oi.order_id) AS order_count,
        ROUND(SUM(oi.price)::numeric, 2) AS total_revenue
    FROM olist.order_items AS oi
    GROUP BY oi.seller_id
    ORDER BY {order_column} DESC
    LIMIT 1
""")

HIGHEST_CATEGORY_REVENUE_SQL = sql_template("""
    SELECT
        COALESCE(pct.product_category_name_english, p.product_category_name) AS category,
        ROUND(SUM(oi.price)::numeric, 2) AS total_revenue
    FROM olist.order_items AS oi
    JOIN olist.products AS p
        ON p.product_id = oi.product_id
    LEFT JOIN olist.product_category_translation AS pct
        ON pct.product_category_name = p.product_category_name
    GROUP BY category
    ORDER BY total_revenue DESC
    LIMIT 1
""")

AVERAGE_DELIVERY_DAYS_SQL = sql_template("""
    SELECT
        ROUND(
            AVG(
                EXTRACT(
                    EPOCH FROM (
                        o.order_delivered_customer_date::timestamp
                        - o.order_purchase_timestamp::timestamp
                    )
                ) / 86400
            )::numeric,
            2
        ) AS average_delivery_days
    FROM olist.orders AS o
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
      AND o.order_purchase_timestamp IS NOT NULL
""")


@dataclass(frozen=True)
class SqlQueryTemplate:
    name: str
    question: str
    sql: str


@dataclass(frozen=True)
class SqlQueryTemplateMatch:
    template: SqlQueryTemplate
    similarity: float


@dataclass(frozen=True)
class TemplateEmbedding:
    template: SqlQueryTemplate
    embedding: list[float]


SQL_QUERY_TEMPLATES: tuple[SqlQueryTemplate, ...] = (
    SqlQueryTemplate(
        name="seller_with_most_orders",
        question="Who sold the most last month?",
        sql=SELLER_SALES_SQL.format(order_column="order_count"),
    ),
    SqlQueryTemplate(
        name="best_seller_by_revenue",
        question="Tell me total revenue of best seller",
        sql=SELLER_SALES_SQL.format(order_column="total_revenue"),
    ),
    SqlQueryTemplate(
        name="product_category_with_highest_revenue",
        question="Which product category has the highest revenue?",
        sql=HIGHEST_CATEGORY_REVENUE_SQL,
    ),
    SqlQueryTemplate(
        name="average_delivery_time_for_shipped_orders",
        question="What is the average delivery time for shipped orders?",
        sql=AVERAGE_DELIVERY_DAYS_SQL,
    ),
)


def normalize_template_question(question: str) -> str:
    normalized = " ".join(question.strip().lower().split())

    for source, target in QUESTION_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)

    return normalized


def cosine_similarity(first: list[float], second: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(first, second))
    first_norm = math.sqrt(sum(a * a for a in first))
    second_norm = math.sqrt(sum(b * b for b in second))

    if first_norm == 0 or second_norm == 0:
        return 0.0

    return dot_product / (first_norm * second_norm)


@lru_cache(maxsize=1)
def get_sql_template_embeddings() -> tuple[TemplateEmbedding, ...]:
    return tuple(
        TemplateEmbedding(
            template=template,
            embedding=embed_text(normalize_template_question(template.question)),
        )
        for template in SQL_QUERY_TEMPLATES
    )


def template_score(
    question_embedding: list[float],
    question_terms: set[str],
    template_embedding: TemplateEmbedding,
) -> float:
    cosine_score = cosine_similarity(question_embedding, template_embedding.embedding)
    boost = keyword_boost(question_terms, template_embedding.template.question)
    return min(MAX_TEMPLATE_SCORE, cosine_score + boost)


def keyword_boost(question_terms: set[str], template_question: str) -> float:
    template_terms = set(normalize_template_question(template_question).split())

    if not question_terms or not template_terms:
        return 0.0

    has_shared_intent = any(
        intent.issubset(question_terms) and intent.issubset(template_terms)
        for intent in BOOST_INTENTS
    )
    if not has_shared_intent:
        return 0.0

    overlap = len(question_terms & template_terms) / len(question_terms | template_terms)
    return min(SQL_QUERY_TEMPLATE_KEYWORD_BOOST, overlap)


def find_sql_query_template(question: str) -> SqlQueryTemplateMatch | None:
    normalized_question = normalize_template_question(question)
    question_embedding = embed_text(normalized_question)
    question_terms = set(normalized_question.split())
    template_embeddings = get_sql_template_embeddings()

    if not template_embeddings:
        return None

    best_template_embedding, best_similarity = max(
        (
            (
                template_embedding,
                template_score(
                    question_embedding,
                    question_terms,
                    template_embedding,
                ),
            )
            for template_embedding in template_embeddings
        ),
        key=lambda candidate: candidate[1],
    )

    if best_similarity < SQL_QUERY_TEMPLATE_THRESHOLD:
        return None

    return SqlQueryTemplateMatch(
        template=best_template_embedding.template,
        similarity=best_similarity,
    )
