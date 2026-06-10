from sqlalchemy import text
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.neo4j import neo4j_driver
from backend.core.postgre import engine
from dotenv import load_dotenv
load_dotenv()


def fetch_rows(sql):
    with engine.connect() as pg:
        return list(pg.execute(sql).mappings())


def chunks(rows, size=1000):
    for index in range(0, len(rows), size):
        yield rows[index:index + size]


def run_batched(query, rows, batch_size=1000):
    with neo4j_driver.session() as neo:
        for batch in chunks(rows, batch_size):
            neo.run(query, rows=batch)




def create_constraints() -> None:
    queries = [
        "CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (c:Customer) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT order_id IF NOT EXISTS FOR (o:Order) REQUIRE o.id IS UNIQUE",
        "CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT seller_id IF NOT EXISTS FOR (s:Seller) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT payment_id IF NOT EXISTS FOR (p:Payment) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT review_id IF NOT EXISTS FOR (r:Review) REQUIRE r.id IS UNIQUE",
        "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
    ]

    with neo4j_driver.session() as session:
        for query in queries:
            session.run(query)


def build_customers_and_orders() -> int:
    sql = text("""
        SELECT
            order_id,
            customer_id,
            order_status,
            order_purchase_timestamp,
            order_approved_at,
            order_delivered_customer_date,
            order_estimated_delivery_date
        FROM olist.orders
    """)

    count = 0
    rows = [
        {
            "customer_id": row["customer_id"],
            "order_id": row["order_id"],
            "status": row["order_status"],
            "purchase_timestamp": str(row["order_purchase_timestamp"]),
            "approved_at": str(row["order_approved_at"]) if row["order_approved_at"] else None,
            "delivered_customer_date": str(row["order_delivered_customer_date"]) if row["order_delivered_customer_date"] else None,
            "estimated_delivery_date": str(row["order_estimated_delivery_date"]) if row["order_estimated_delivery_date"] else None,
        }
        for row in fetch_rows(sql)
    ]

    run_batched(
        """
        UNWIND $rows AS row
        MERGE (c:Customer {id: row.customer_id})
        MERGE (o:Order {id: row.order_id})
        SET o.status = row.status,
            o.purchase_timestamp = row.purchase_timestamp,
            o.approved_at = row.approved_at,
            o.delivered_customer_date = row.delivered_customer_date,
            o.estimated_delivery_date = row.estimated_delivery_date
        MERGE (c)-[:PLACED]->(o)
        """,
        rows,
    )
    count = len(rows)

    return count


def build_products_sellers_and_items() -> int:
    sql = text("""
        SELECT
            oi.order_id,
            oi.product_id,
            oi.seller_id,
            oi.order_item_id,
            oi.price,
            oi.freight_value,
            p.product_category_name,
            t.product_category_name_english
        FROM olist.order_items oi
        LEFT JOIN olist.products p
            ON oi.product_id = p.product_id
        LEFT JOIN olist.product_category_translation t
            ON p.product_category_name = t.product_category_name
    """)

    count = 0
    rows = [
        {
            "order_id": row["order_id"],
            "product_id": row["product_id"],
            "seller_id": row["seller_id"],
            "item_id": row["order_item_id"],
            "price": float(row["price"]) if row["price"] is not None else None,
            "freight_value": float(row["freight_value"]) if row["freight_value"] is not None else None,
            "category": row["product_category_name_english"] or row["product_category_name"],
        }
        for row in fetch_rows(sql)
    ]

    run_batched(
        """
        UNWIND $rows AS row
        MERGE (o:Order {id: row.order_id})
        MERGE (p:Product {id: row.product_id})
        MERGE (s:Seller {id: row.seller_id})
        MERGE (o)-[contains:CONTAINS {item_id: row.item_id}]->(p)
        SET contains.price = row.price,
            contains.freight_value = row.freight_value
        MERGE (o)-[:SOLD_BY]->(s)
        FOREACH (_ IN CASE WHEN row.category IS NULL THEN [] ELSE [1] END |
            MERGE (c:Category {name: row.category})
            MERGE (p)-[:IN_CATEGORY]->(c)
        )
        """,
        rows,
    )
    count = len(rows)

    return count


def build_payments() -> int:
    sql = text("""
        SELECT
            order_id,
            payment_sequential,
            payment_type,
            payment_installments,
            payment_value
        FROM olist.order_payments
    """)

    count = 0
    rows = [
        {
            "order_id": row["order_id"],
            "payment_id": f"{row['order_id']}:{row['payment_sequential']}",
            "payment_type": row["payment_type"],
            "installments": row["payment_installments"],
            "value": float(row["payment_value"]) if row["payment_value"] is not None else None,
        }
        for row in fetch_rows(sql)
    ]

    run_batched(
        """
        UNWIND $rows AS row
        MERGE (o:Order {id: row.order_id})
        MERGE (p:Payment {id: row.payment_id})
        SET p.type = row.payment_type,
            p.installments = row.installments,
            p.value = row.value
        MERGE (o)-[:HAS_PAYMENT]->(p)
        """,
        rows,
    )
    count = len(rows)

    return count


def build_reviews() -> int:
    sql = text("""
        SELECT
            review_id,
            order_id,
            review_score,
            review_comment_title,
            review_comment_message,
            review_creation_date
        FROM olist.order_reviews
    """)

    count = 0
    rows = [
        {
            "order_id": row["order_id"],
            "review_id": row["review_id"],
            "score": row["review_score"],
            "title": row["review_comment_title"],
            "message": row["review_comment_message"],
            "created_at": str(row["review_creation_date"]) if row["review_creation_date"] else None,
        }
        for row in fetch_rows(sql)
    ]

    run_batched(
        """
        UNWIND $rows AS row
        MERGE (o:Order {id: row.order_id})
        MERGE (r:Review {id: row.review_id})
        SET r.score = row.score,
            r.title = row.title,
            r.message = row.message,
            r.created_at = row.created_at
        MERGE (o)-[:HAS_REVIEW]->(r)
        """,
        rows,
    )
    count = len(rows)

    return count


def build_knowledge_graph() -> dict[str, int]:
    create_constraints()

    return {
        "orders": build_customers_and_orders(),
        "items": build_products_sellers_and_items(),
        "payments": build_payments(),
        "reviews": build_reviews(),
    }


if __name__ == "__main__":
    result = build_knowledge_graph()
    print(result)
