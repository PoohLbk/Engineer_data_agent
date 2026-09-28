"""Bronze layer: โหลดไฟล์ดิบเข้าฐานข้อมูลแบบ idempotent (รันซ้ำแล้วข้อมูลไม่ซ้ำ)"""
import csv
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = ["order_id", "customer_id", "product_id", "order_date",
            "order_status", "quantity", "unit_price"]


def ensure_tables(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS bronze_orders (
        order_id TEXT, customer_id TEXT, product_id TEXT, order_date TEXT,
        order_status TEXT, quantity TEXT, unit_price TEXT,
        _batch_id TEXT, _source_file TEXT, _ingested_at TEXT
    );
    CREATE TABLE IF NOT EXISTS ingestion_log (
        file_hash TEXT PRIMARY KEY, source_file TEXT, batch_id TEXT,
        row_count INTEGER, ingested_at TEXT
    );
    """)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_file(conn, path) -> int:
    """คืนจำนวนแถวที่โหลดใหม่ (0 = เคยโหลดไฟล์นี้แล้ว จึงข้าม)"""
    path = Path(path)
    ensure_tables(conn)
    digest = file_hash(path)
    if conn.execute("SELECT 1 FROM ingestion_log WHERE file_hash = ?", (digest,)).fetchone():
        return 0

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    batch_id = uuid.uuid4().hex[:12]
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: ขาดคอลัมน์ {sorted(missing)}")
        rows = [tuple((r.get(c) or "") for c in REQUIRED) + (batch_id, path.name, now)
                for r in reader]

    with conn:  # transaction เดียว: ล้มแล้ว rollback ทั้งก้อน
        conn.executemany("INSERT INTO bronze_orders VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        conn.execute("INSERT INTO ingestion_log VALUES (?,?,?,?,?)",
                     (digest, path.name, batch_id, len(rows), now))
    return len(rows)


def ingest_dir(conn, raw_dir) -> dict:
    result = {}
    for path in sorted(Path(raw_dir).glob("*.csv")):
        result[path.name] = ingest_file(conn, path)
    return result
