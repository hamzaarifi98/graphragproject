import math
import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

from backend.services.pdf_services.text_embedder import embed_text

load_dotenv()

KG_QUERY_TEMPLATE_THRESHOLD = float(
    os.getenv("KG_QUERY_TEMPLATE_THRESHOLD", "0.88")
)


@dataclass(frozen=True)
class KgQueryTemplate:
    name: str
    question: str
    cypher: str


@dataclass(frozen=True)
class KgQueryTemplateMatch:
    template: KgQueryTemplate
    similarity: float


KG_QUERY_TEMPLATES: tuple[KgQueryTemplate, ...] = (
    KgQueryTemplate(
        name="seller_with_most_delayed_orders",
        question="Who had most delayed deliveries last month?",
        cypher="""
            MATCH (o:Order)-[:SOLD_BY]->(s:Seller)
            WHERE o.delivered_customer_date IS NOT NULL
              AND o.estimated_delivery_date IS NOT NULL
              AND datetime(replace(o.delivered_customer_date, ' ', 'T'))
                  > datetime(replace(o.estimated_delivery_date, ' ', 'T'))
            RETURN
                s.id AS seller_id,
                count(o) AS delayed_orders
            ORDER BY delayed_orders DESC
            LIMIT 1
        """,
    ),
    KgQueryTemplate(
        name="seller_with_most_bad_reviews",
        question="Who had most bad reviews last month?",
        cypher="""
            MATCH (o:Order)-[:SOLD_BY]->(s:Seller)
            MATCH (o)-[:HAS_REVIEW]->(r:Review)
            WHERE r.score <= 2
            RETURN
                s.id AS seller_id,
                count(r) AS bad_reviews
            ORDER BY bad_reviews DESC
            LIMIT 1
        """,
    ),
    KgQueryTemplate(
        name="customers_from_sellers_with_delayed_orders",
        question="Which customers bought products from a seller with delayed orders?",
        cypher="""
            MATCH (c:Customer)-[:PLACED]->(o:Order)-[:SOLD_BY]->(s:Seller)
            WHERE o.delivered_customer_date IS NOT NULL
              AND o.estimated_delivery_date IS NOT NULL
              AND datetime(replace(o.delivered_customer_date, ' ', 'T'))
                  > datetime(replace(o.estimated_delivery_date, ' ', 'T'))
            RETURN DISTINCT
                c.id AS customer_id,
                s.id AS seller_id,
                o.id AS order_id
            LIMIT 100
        """,
    ),
    KgQueryTemplate(
        name="orders_connected_to_sellers_with_bad_reviews",
        question="Which orders connect customers to sellers with bad reviews?",
        cypher="""
            MATCH (c:Customer)-[:PLACED]->(o:Order)-[:SOLD_BY]->(s:Seller)
            MATCH (o)-[:HAS_REVIEW]->(r:Review)
            WHERE r.score <= 2
            RETURN
                o.id AS order_id,
                c.id AS customer_id,
                s.id AS seller_id,
                r.score AS review_score
            ORDER BY review_score ASC
            LIMIT 100
        """,
    ),
    KgQueryTemplate(
        name="categories_connected_to_late_deliveries",
        question="Which product categories are connected to late deliveries?",
        cypher="""
            MATCH (o:Order)-[:CONTAINS]->(:Product)-[:IN_CATEGORY]->(c:Category)
            WHERE o.delivered_customer_date IS NOT NULL
              AND o.estimated_delivery_date IS NOT NULL
              AND datetime(replace(o.delivered_customer_date, ' ', 'T'))
                  > datetime(replace(o.estimated_delivery_date, ' ', 'T'))
            RETURN DISTINCT c.name AS category
            ORDER BY category
        """,
    ),
    KgQueryTemplate(
        name="sellers_linked_to_delays_and_low_reviews",
        question="Which sellers are linked to both delays and low review scores?",
        cypher="""
            MATCH (o:Order)-[:SOLD_BY]->(s:Seller)
            MATCH (o)-[:HAS_REVIEW]->(r:Review)
            WHERE r.score <= 3
              AND o.delivered_customer_date IS NOT NULL
              AND o.estimated_delivery_date IS NOT NULL
              AND datetime(replace(o.delivered_customer_date, ' ', 'T'))
                  > datetime(replace(o.estimated_delivery_date, ' ', 'T'))
            RETURN
                s.id AS seller_id,
                count(DISTINCT o) AS delayed_low_review_orders,
                avg(r.score) AS average_review_score
            ORDER BY delayed_low_review_orders DESC, average_review_score ASC
            LIMIT 100
        """,
    ),
    KgQueryTemplate(
        name="top_product_categories_by_orders",
        question="What are the top product categories by number of orders?",
        cypher="""
            MATCH (:Order)-[:CONTAINS]->(:Product)-[:IN_CATEGORY]->(c:Category)
            RETURN c.name AS category, count(*) AS orders
            ORDER BY orders DESC
            LIMIT 10
        """,
    ),
)


def normalize_template_question(question: str) -> str:
    normalized = " ".join(question.strip().lower().split())

    replacements = {
        "late deliveries": "delayed deliveries",
        "late delivery": "delayed delivery",
        "low review scores": "bad reviews",
        "low reviews": "bad reviews",
        "poor reviews": "bad reviews",
        "negative reviews": "bad reviews",
        "linked to": "connected to",
    }

    for source, target in replacements.items():
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
def get_kg_template_embeddings() -> tuple[tuple[KgQueryTemplate, list[float]], ...]:
    return tuple(
        (
            template,
            embed_text(normalize_template_question(template.question)),
        )
        for template in KG_QUERY_TEMPLATES
    )


def find_kg_query_template(question: str) -> KgQueryTemplateMatch | None:
    question_embedding = embed_text(normalize_template_question(question))
    best_template: KgQueryTemplate | None = None
    best_similarity = 0.0

    for template, template_embedding in get_kg_template_embeddings():
        similarity = cosine_similarity(question_embedding, template_embedding)

        if similarity > best_similarity:
            best_similarity = similarity
            best_template = template

    if best_template is None or best_similarity < KG_QUERY_TEMPLATE_THRESHOLD:
        return None

    return KgQueryTemplateMatch(
        template=best_template,
        similarity=best_similarity,
    )


def get_template_cypher(question: str) -> str | None:
    template_match = find_kg_query_template(question)

    if template_match is None:
        return None

    return template_match.template.cypher.strip()
