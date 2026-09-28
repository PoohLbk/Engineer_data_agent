import sqlite3
from pathlib import Path

from . import config


def connect(db_path=None, read_only: bool = False) -> sqlite3.Connection:
    """read_only=True ใช้สำหรับ AI agent: เปิดไฟล์แบบอ่านอย่างเดียวจริง ๆ"""
    path = Path(db_path or config.DB_PATH)
    if read_only:
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only = ON")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def describe_gold_schema(conn) -> str:
    """สร้างข้อความ schema จากฐานข้อมูลจริง เอาไปใส่ prompt ของ LLM แทนการพิมพ์เอง"""
    lines = []
    for table in config.GOLD_TABLES:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        lines.append(f"{table}({', '.join(c['name'] + ' ' + c['type'] for c in cols)})")
    return "\n".join(lines)
