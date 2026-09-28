import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import db, ingest, llm_retry, quality, runner, sample_data, sql_guard


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raw = self.tmp / "raw"
        self.db = self.tmp / "wh.db"
        sample_data.generate(self.raw / "orders.csv", n=1000)

    def test_full_run_and_idempotency(self):
        first = runner.run_pipeline(self.db, self.raw)
        self.assertGreater(first["ingest"]["orders.csv"], 1000)   # รวมแถวซ้ำ
        second = runner.run_pipeline(self.db, self.raw)
        self.assertEqual(second["ingest"]["orders.csv"], 0)       # ไฟล์เดิมต้องถูกข้าม
        conn = db.connect(self.db)
        bronze = conn.execute("SELECT COUNT(*) FROM bronze_orders").fetchone()[0]
        self.assertEqual(bronze, first["ingest"]["orders.csv"])
        dup = conn.execute("SELECT COUNT(*)-COUNT(DISTINCT order_id) FROM silver_orders").fetchone()[0]
        self.assertEqual(dup, 0)
        self.assertGreater(conn.execute("SELECT COUNT(*) FROM silver_rejects").fetchone()[0], 0)
        runs = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
        self.assertEqual(runs, 6)                                  # 2 รอบ x 3 ขั้น

    def test_missing_column_raises(self):
        bad = self.tmp / "bad.csv"
        bad.write_text("order_id,customer_id\n1,2\n")
        conn = db.connect(self.db)
        with self.assertRaises(ValueError):
            ingest.ingest_file(conn, bad)

    def test_quality_fails_when_too_many_rejects(self):
        dirty = self.tmp / "dirty_raw"
        sample_data.generate(dirty / "d.csv", n=500, dirty_ratio=0.5)
        with self.assertRaises(quality.DataQualityError):
            runner.run_pipeline(self.tmp / "d.db", dirty)
        conn = db.connect(self.tmp / "d.db")
        failed = conn.execute("SELECT COUNT(*) FROM pipeline_runs WHERE status='failed'").fetchone()[0]
        self.assertEqual(failed, 1)                                # บันทึกความล้มเหลวไว้

    def test_gold_matches_status_shape(self):
        runner.run_pipeline(self.db, self.raw)
        conn = db.connect(self.db)
        statuses = {r[0] for r in conn.execute("SELECT order_status FROM gold_order_status_summary")}
        self.assertTrue(statuses <= {"Completed", "Returned", "Cancelled", "Pending"})
        self.assertIn("gold_fact_orders", db.describe_gold_schema(conn))

    def test_agent_can_query_gold_but_not_write(self):
        runner.run_pipeline(self.db, self.raw)
        cols, rows, sql = sql_guard.run_safe_query(
            "SELECT order_status, total_orders FROM gold_order_status_summary", self.db)
        self.assertEqual(cols, ["order_status", "total_orders"])
        self.assertIn("LIMIT", sql)
        ro = db.connect(self.db, read_only=True)
        with self.assertRaises(sqlite3.OperationalError):
            ro.execute("DELETE FROM gold_fact_orders")             # ชั้น read-only ของ DB เอง


class SQLGuardTests(unittest.TestCase):
    def test_blocks_bad_sql(self):
        bad = [
            "DROP TABLE gold_fact_orders",
            "SELECT * FROM gold_fact_orders; DELETE FROM gold_fact_orders",
            "SELECT * FROM bronze_orders",
            "SELECT * FROM sqlite_master",
            "WITH x AS (SELECT 1) UPDATE gold_fact_orders SET revenue = 0",
            "",
        ]
        for sql in bad:
            with self.subTest(sql=sql), self.assertRaises(sql_guard.UnsafeSQLError):
                sql_guard.validate_sql(sql)

    def test_allows_good_sql(self):
        ok = sql_guard.validate_sql(
            "-- comment\nWITH t AS (SELECT * FROM gold_daily_sales) "
            "SELECT order_date FROM t WHERE order_date > '2026-01-01' -- update")
        self.assertTrue(ok.rstrip().endswith("LIMIT 1000"))
        self.assertEqual(sql_guard.validate_sql("SELECT 1 FROM gold_top_products LIMIT 5").count("LIMIT"), 1)

    def test_keyword_inside_string_is_fine(self):
        sql_guard.validate_sql("SELECT * FROM gold_fact_orders WHERE order_status = 'update'")


class LLMRetryTests(unittest.TestCase):
    def test_retries_then_falls_back(self):
        calls = {"gemini": 0}

        def gemini(_):
            calls["gemini"] += 1
            raise RuntimeError("503 UNAVAILABLE")

        name, text = llm_retry.call_with_fallback(
            [("gemini", gemini), ("local", lambda p: "ok")], "hi", sleep=lambda s: None)
        self.assertEqual((name, text), ("local", "ok"))
        self.assertEqual(calls["gemini"], 3)

    def test_permanent_error_not_retried(self):
        calls = {"n": 0}

        def bad(_):
            calls["n"] += 1
            raise ValueError("invalid api key")

        with self.assertRaises(llm_retry.LLMUnavailable):
            llm_retry.call_with_fallback([("g", bad)], "hi", sleep=lambda s: None)
        self.assertEqual(calls["n"], 1)

    def test_runner_retry_backoff(self):
        delays, state = [], {"n": 0}

        @runner.retry(times=3, base_delay=1, sleep=delays.append)
        def flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise OSError("disk busy")
            return "done"

        self.assertEqual(flaky(), "done")
        self.assertEqual(delays, [1, 2])


if __name__ == "__main__":
    unittest.main()
