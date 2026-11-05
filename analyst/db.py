"""Load the Olist CSVs into a local DuckDB file and run read-only queries against it."""

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "olist.duckdb"

# table name -> source CSV
TABLES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# Business-friendly view that most questions can start from: one row per order item,
# already joined to order, customer, product (English category), seller and review.
SALES_VIEW = """
CREATE OR REPLACE VIEW sales AS
SELECT
    oi.order_id,
    oi.order_item_id,
    o.order_status,
    o.order_purchase_timestamp,
    CAST(date_trunc('month', o.order_purchase_timestamp) AS DATE) AS order_month,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    date_diff('day', o.order_purchase_timestamp, o.order_delivered_customer_date) AS delivery_days,
    o.order_delivered_customer_date > o.order_estimated_delivery_date AS delivered_late,
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    oi.product_id,
    COALESCE(t.product_category_name_english, p.product_category_name) AS category,
    oi.seller_id,
    s.seller_city,
    s.seller_state,
    oi.price,
    oi.freight_value,
    oi.price + oi.freight_value AS item_total,
    r.review_score
FROM order_items oi
JOIN orders o USING (order_id)
JOIN customers c USING (customer_id)
LEFT JOIN products p USING (product_id)
LEFT JOIN category_translation t USING (product_category_name)
LEFT JOIN sellers s USING (seller_id)
LEFT JOIN (
    SELECT order_id, avg(review_score) AS review_score FROM reviews GROUP BY order_id
) r USING (order_id)
"""


def build(force: bool = False) -> Path:
    """Create data/olist.duckdb from the raw CSVs (skipped if it already exists)."""
    if DB_PATH.exists() and not force:
        return DB_PATH
    tmp = DB_PATH.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    for table, filename in TABLES.items():
        con.execute(
            f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?, header=true)",
            [str(RAW_DIR / filename)],
        )
    con.execute(SALES_VIEW)
    con.close()
    tmp.replace(DB_PATH)
    return DB_PATH


def connect() -> duckdb.DuckDBPyConnection:
    build()
    return duckdb.connect(str(DB_PATH), read_only=True)


def schema_text(con: duckdb.DuckDBPyConnection) -> str:
    """Compact description of every table/view and its columns, for the system prompt."""
    rows = con.execute(
        """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'main'
        ORDER BY table_name, ordinal_position
        """
    ).fetchall()
    tables: dict[str, list[str]] = {}
    for table, column, dtype in rows:
        tables.setdefault(table, []).append(f"{column} {dtype}")
    return "\n".join(f"{t}({', '.join(cols)})" for t, cols in tables.items())


class QueryError(Exception):
    pass


def run_query(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pd.DataFrame:
    """Run a single read-only SELECT/WITH statement."""
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error as e:
        raise QueryError(str(e)) from e
    if len(statements) != 1:
        raise QueryError("Send exactly one SQL statement.")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise QueryError("Only SELECT queries are allowed.")
    try:
        return con.execute(sql, params or []).df()
    except duckdb.Error as e:
        raise QueryError(str(e)) from e


if __name__ == "__main__":
    print(f"Built {build(force=True)}")
