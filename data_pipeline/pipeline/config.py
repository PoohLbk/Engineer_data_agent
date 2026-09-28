"""ค่าตั้งต้นของ pipeline (แก้ผ่าน environment variable ได้)"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("WAREHOUSE_DB", ROOT / "warehouse.db"))
RAW_DIR = Path(os.getenv("RAW_DIR", ROOT / "data" / "raw"))

VALID_STATUSES = ("Completed", "Returned", "Cancelled", "Pending")
MAX_REJECT_RATIO = 0.05     # ถ้าแถวที่ถูกปฏิเสธเกิน 5% ให้ pipeline ล้ม
FRESHNESS_HOURS = 26        # ข้อมูลล่าสุดต้องไม่เก่ากว่านี้ (เตือน ไม่ล้ม)

# ตารางที่ AI agent ได้รับอนุญาตให้อ่าน (เฉพาะ Gold)
GOLD_TABLES = (
    "gold_fact_orders",
    "gold_order_status_summary",
    "gold_daily_sales",
    "gold_top_products",
)
