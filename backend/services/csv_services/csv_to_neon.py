from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from backend.constants.constants import STRUCTURED_DATA_DIR
from backend.core.postgre import engine
from backend.services.csv_services.csv_loader import load_csv
from backend.services.pdf_services.vector_store import create_olist_schema


CSV_TABLES = {
    "olist_customers_dataset.csv": "customers",
    "olist_geolocation_dataset.csv": "geolocation",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_orders_dataset.csv": "orders",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "product_category_name_translation.csv": "product_category_translation",
}


def ingest_csvs() -> list[dict[str, int | str]]:
    create_olist_schema()
    structured_data_dir = Path(STRUCTURED_DATA_DIR)
    results = []

    for file_name, table_name in CSV_TABLES.items():
        csv_path = structured_data_dir / file_name
        df = load_csv(csv_path)

        df.to_sql(
            name=table_name,
            con=engine,
            schema="olist",
            if_exists="replace",
            index=False,
            chunksize=5000,
            method="multi",
        )

        results.append(
            {
                "file_name": file_name,
                "schema": "olist",
                "table_name": table_name,
                "rows_loaded": len(df),
            }
        )
        print(f"Loaded {len(df)} rows into olist.{table_name}")

    return results


def main() -> list[dict[str, int | str]]:
    return ingest_csvs()


if __name__ == "__main__":
    main()
