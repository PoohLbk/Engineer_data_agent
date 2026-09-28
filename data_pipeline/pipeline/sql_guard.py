"""Guardrail สำหรับ SQL ที่ LLM สร้าง: ตรวจก่อนรัน + รันแบบ read-only

หมายเหตุ: นี่เป็นการตรวจด้วย regex ซึ่งพอสำหรับโปรเจกต์ตัวอย่าง
ถ้าขึ้น production ควรใช้ parser จริง (เช่น sqlglot) แทน และใช้ DB user ที่มีสิทธิ์ SELECT อย่างเดียว
"""
import re

from . import config, db

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|"
    r"truncate|grant|revoke|exec|load_extension)\b", re.I)


class UnsafeSQLError(ValueError):
    pass


def _strip(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)   # block comment
    sql = re.sub(r"--[^\n]*", " ", sql)                # line comment
    return sql.strip().rstrip(";").strip()


def validate_sql(sql: str, allowed_tables=config.GOLD_TABLES, max_rows: int = 1000) -> str:
    """คืน SQL ที่ปลอดภัยพร้อม LIMIT หรือ raise UnsafeSQLError"""
    cleaned = _strip(sql)
    no_literals = re.sub(r"'(?:[^']|'')*'", "''", cleaned)   # ตัดข้อความใน quote ออกก่อนตรวจ

    if not cleaned:
        raise UnsafeSQLError("SQL ว่างเปล่า")
    if ";" in no_literals:
        raise UnsafeSQLError("อนุญาตเพียงคำสั่งเดียว")
    if not re.match(r"(?i)^(select|with)\b", no_literals):
        raise UnsafeSQLError("อนุญาตเฉพาะ SELECT")
    if FORBIDDEN.search(no_literals):
        raise UnsafeSQLError("พบคำสั่งที่ไม่อนุญาต")

    ctes = set(re.findall(r"(?i)(?:\bwith|,)\s*(\w+)\s+as\s*\(", no_literals))
    tables = set(re.findall(r"(?i)\b(?:from|join)\s+([A-Za-z_][\w.]*)", no_literals))
    not_allowed = {t for t in tables if t not in ctes and t not in allowed_tables}
    if not_allowed:
        raise UnsafeSQLError(f"ตารางที่ไม่อนุญาต: {sorted(not_allowed)}")

    if not re.search(r"(?i)\blimit\s+\d+\s*$", no_literals):
        cleaned = f"{cleaned}\nLIMIT {max_rows}"
    return cleaned


def run_safe_query(sql: str, db_path=None, max_rows: int = 1000):
    """ตรวจ + รันแบบ read-only คืน (ชื่อคอลัมน์, แถว, SQL ที่รันจริง)"""
    safe_sql = validate_sql(sql, max_rows=max_rows)
    conn = db.connect(db_path, read_only=True)
    try:
        cur = conn.execute(safe_sql)
        cols = [d[0] for d in cur.description]
        return cols, [tuple(r) for r in cur.fetchall()], safe_sql
    finally:
        conn.close()
