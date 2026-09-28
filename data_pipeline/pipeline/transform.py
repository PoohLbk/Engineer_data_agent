"""Silver (ทำความสะอาด + แยกแถวเสีย) และ Gold (ตารางพร้อมให้ agent ถาม)"""
from . import config


def _status_list() -> str:
    return ",".join(f"'{s}'" for s in config.VALID_STATUSES)


def build_silver(conn):
    conn.executescript(f"""
    DROP VIEW  IF EXISTS _stage;
    DROP TABLE IF EXISTS silver_orders;
    DROP TABLE IF EXISTS silver_rejects;

    CREATE TEMP VIEW _stage AS
    SELECT
        rowid AS _rid,
        TRIM(order_id)    AS order_id,
        TRIM(customer_id) AS customer_id,
        TRIM(product_id)  AS product_id,
        date(TRIM(order_date)) AS order_date,
        CAST(TRIM(quantity)   AS INTEGER) AS quantity,
        CAST(TRIM(unit_price) AS REAL)    AS unit_price,
        UPPER(SUBSTR(TRIM(order_status), 1, 1)) || LOWER(SUBSTR(TRIM(order_status), 2)) AS order_status,
        _batch_id, _ingested_at,
        CASE
            WHEN order_id IS NULL OR TRIM(order_id) = ''                THEN 'missing_order_id'
            WHEN date(TRIM(order_date)) IS NULL                         THEN 'invalid_date'
            WHEN COALESCE(CAST(TRIM(quantity)   AS INTEGER), 0) <= 0    THEN 'invalid_quantity'
            WHEN COALESCE(CAST(TRIM(unit_price) AS REAL), 0)    <= 0    THEN 'invalid_price'
            WHEN UPPER(SUBSTR(TRIM(order_status), 1, 1)) || LOWER(SUBSTR(TRIM(order_status), 2))
                 NOT IN ({_status_list()})                              THEN 'invalid_status'
        END AS reject_reason
    FROM bronze_orders;

    -- order_id ซ้ำ: เก็บแถวที่โหลดล่าสุด
    CREATE TABLE silver_orders AS
    SELECT order_id, customer_id, product_id, order_date, quantity, unit_price,
           order_status, _batch_id, _ingested_at
    FROM (
        SELECT *, ROW_NUMBER() OVER (
                   PARTITION BY order_id ORDER BY _ingested_at DESC, _rid DESC) AS rn
        FROM _stage WHERE reject_reason IS NULL
    ) WHERE rn = 1;

    CREATE TABLE silver_rejects AS
    SELECT _rid, order_id, reject_reason, _batch_id, _ingested_at
    FROM _stage WHERE reject_reason IS NOT NULL;
    """)


def build_gold(conn):
    conn.executescript("""
    DROP TABLE IF EXISTS gold_fact_orders;
    DROP TABLE IF EXISTS gold_order_status_summary;
    DROP TABLE IF EXISTS gold_daily_sales;
    DROP TABLE IF EXISTS gold_top_products;

    CREATE TABLE gold_fact_orders AS
    SELECT order_id, customer_id, product_id, order_date, order_status,
           quantity, unit_price, ROUND(quantity * unit_price, 2) AS revenue
    FROM silver_orders;

    CREATE TABLE gold_order_status_summary AS
    SELECT order_status,
           COUNT(*) AS total_orders,
           ROUND(SUM(revenue), 2) AS revenue,
           ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM gold_fact_orders), 2) AS pct_of_orders
    FROM gold_fact_orders GROUP BY order_status;

    -- ยอดขายนับเฉพาะออเดอร์ที่ Completed
    CREATE TABLE gold_daily_sales AS
    SELECT order_date, COUNT(*) AS orders, ROUND(SUM(revenue), 2) AS revenue
    FROM gold_fact_orders WHERE order_status = 'Completed'
    GROUP BY order_date;

    CREATE TABLE gold_top_products AS
    SELECT product_id, SUM(quantity) AS units_sold, ROUND(SUM(revenue), 2) AS revenue
    FROM gold_fact_orders WHERE order_status = 'Completed'
    GROUP BY product_id;
    """)


def run_transform(conn) -> dict:
    build_silver(conn)
    build_gold(conn)
    conn.commit()
    count = lambda t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    return {t: count(t) for t in ("bronze_orders", "silver_orders", "silver_rejects",
                                  "gold_fact_orders", "gold_order_status_summary")}
