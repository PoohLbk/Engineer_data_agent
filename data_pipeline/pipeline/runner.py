"""ตัวสั่งรัน pipeline: ingest -> transform -> quality พร้อม retry และ run log"""
import argparse
import json
import time
import uuid
from datetime import datetime, timezone

from . import config, db, ingest, quality, transform


def retry(times=3, base_delay=1.0, exceptions=(OSError,), sleep=time.sleep):
    """retry แบบ exponential backoff (ใช้กับงานที่อาจล้มชั่วคราว เช่น I/O)"""
    def deco(fn):
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions:
                    if attempt == times:
                        raise
                    sleep(base_delay * 2 ** (attempt - 1))
        return wrapper
    return deco


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _record(conn, run_id, step, status, detail, started):
    conn.execute("""CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_id TEXT, step TEXT, status TEXT, detail TEXT,
        started_at TEXT, finished_at TEXT)""")
    with conn:
        conn.execute("INSERT INTO pipeline_runs VALUES (?,?,?,?,?,?)",
                     (run_id, step, status, detail, started, _now()))


def _run_step(conn, run_id, step, fn):
    started = _now()
    try:
        result = fn()
    except Exception as exc:
        _record(conn, run_id, step, "failed", str(exc), started)
        raise
    _record(conn, run_id, step, "success", json.dumps(result, default=str), started)
    return result


def run_pipeline(db_path=None, raw_dir=None) -> dict:
    run_id = uuid.uuid4().hex[:12]
    conn = db.connect(db_path)
    try:
        load = retry(times=3, base_delay=1.0)(ingest.ingest_dir)
        summary = {"run_id": run_id}
        summary["ingest"] = _run_step(conn, run_id, "ingest",
                                      lambda: load(conn, raw_dir or config.RAW_DIR))
        summary["transform"] = _run_step(conn, run_id, "transform",
                                         lambda: transform.run_transform(conn))
        checks = _run_step(conn, run_id, "quality",
                           lambda: quality.run_checks(conn, run_id))
        summary["quality"] = [{"check": n, "passed": ok} for n, _, ok, _ in checks]
        return summary
    finally:
        conn.close()


if __name__ == "__main__":
    argparse.ArgumentParser(description="Run the data pipeline").parse_args()
    print(json.dumps(run_pipeline(), indent=2, ensure_ascii=False))
