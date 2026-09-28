"""Data quality checks: บันทึกผลลง dq_results และล้ม pipeline เมื่อ check ระดับ critical ไม่ผ่าน"""
from datetime import datetime, timezone

from . import config


class DataQualityError(RuntimeError):
    pass


def _scalar(conn, sql):
    return conn.execute(sql).fetchone()[0]


def collect_checks(conn):
    """คืน list ของ (ชื่อ check, ระดับ, ผ่านไหม, รายละเอียด)"""
    checks = []

    n_silver = _scalar(conn, "SELECT COUNT(*) FROM silver_orders")
    checks.append(("silver_not_empty", "critical", n_silver > 0, f"rows={n_silver}"))

    dup = _scalar(conn, "SELECT COUNT(*) - COUNT(DISTINCT order_id) FROM silver_orders")
    checks.append(("silver_order_id_unique", "critical", dup == 0, f"duplicates={dup}"))

    statuses = ",".join(f"'{s}'" for s in config.VALID_STATUSES)
    bad = _scalar(conn, f"SELECT COUNT(*) FROM silver_orders WHERE order_status NOT IN ({statuses})")
    checks.append(("silver_status_accepted_values", "critical", bad == 0, f"invalid={bad}"))

    n_bronze = _scalar(conn, "SELECT COUNT(*) FROM bronze_orders")
    n_rej = _scalar(conn, "SELECT COUNT(*) FROM silver_rejects")
    ratio = (n_rej / n_bronze) if n_bronze else 0.0
    checks.append(("reject_ratio_within_limit", "critical",
                   ratio <= config.MAX_REJECT_RATIO,
                   f"ratio={ratio:.2%} limit={config.MAX_REJECT_RATIO:.0%}"))

    gold_total = _scalar(conn, "SELECT COALESCE(SUM(total_orders), 0) FROM gold_order_status_summary")
    checks.append(("gold_reconciles_with_silver", "critical",
                   gold_total == n_silver, f"gold={gold_total} silver={n_silver}"))

    latest = _scalar(conn, "SELECT MAX(_ingested_at) FROM bronze_orders")
    if latest:
        age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(latest)).total_seconds() / 3600
        checks.append(("bronze_freshness", "warning",
                       age_h <= config.FRESHNESS_HOURS, f"age_hours={age_h:.1f}"))
    else:
        checks.append(("bronze_freshness", "warning", False, "no data"))
    return checks


def run_checks(conn, run_id: str) -> list:
    conn.execute("""CREATE TABLE IF NOT EXISTS dq_results (
        run_id TEXT, check_name TEXT, severity TEXT, passed INTEGER,
        detail TEXT, checked_at TEXT)""")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    checks = collect_checks(conn)
    with conn:
        conn.executemany("INSERT INTO dq_results VALUES (?,?,?,?,?,?)",
                         [(run_id, n, s, int(ok), d, now) for n, s, ok, d in checks])
    failed = [c for c in checks if c[1] == "critical" and not c[2]]
    if failed:
        raise DataQualityError("; ".join(f"{n} ({d})" for n, _, _, d in failed))
    return checks
