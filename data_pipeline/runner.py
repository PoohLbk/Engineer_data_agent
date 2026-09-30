import os
import sqlite3
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# กำหนดเส้นทางโฟลเดอร์และฐานข้อมูลภายในโฟลเดอร์ data_pipeline
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = BASE_DIR / "warehouse.db"

def setup_directories():
    """สร้างโฟลเดอร์สำหรับเก็บข้อมูลดิบ"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_sample_data():
    """ดาวน์โหลดข้อมูลตัวอย่างจาก GitHub Releases ของคุณมาเก็บไว้ที่ raw/ เพื่อให้มีข้อมูลรันได้ทันที"""
    base_url = "https://github.com/PoohLbk/Engineer_data_agent/releases/download/v1.0"
    files = {
        "customer_master.csv": f"{base_url}/customer_master.csv",
        "ecommerce_sales_customer_analytics_150k.csv": f"{base_url}/ecommerce_sales_customer_analytics_150k.csv",
        "order_items.csv": f"{base_url}/order_items.csv",
        "product_catalog.csv": f"{base_url}/product_catalog.csv"
    }
    
    for filename, url in files.items():
        file_path = RAW_DIR / filename
        if not file_path.exists():
            try:
                print(กำลังดาวน์โหลด {filename}...)
                res = requests.get(url, timeout=30)
                if res.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(res.content)
            except Exception as e:
                print(f"ไม่สามารถดาวน์โหลด {filename}: {e}")

def run_pipeline(db_path=None) -> dict:
    """ฟังก์ชันหลักสำหรับรัน ETL Pipeline ทั้งหมด"""
    target_db = db_path or str(DB_PATH)
    setup_directories()
    
    # 1. เชื่อมต่อ DuckDB
    con = duckdb.connect(database=target_db)
    
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary = {"run_id": run_id, "status": "success", "tables_created": []}
    
    try:
        # 2. Bronze Layer: สร้างตารางและโหลดข้อมูล CSV เข้า DuckDB โดยตรง
        # (แก้ไขปัญหา no such table: bronze_orders โดยการสร้างจากไฟล์ CSV ใน raw_dir)
        csv_files = list(RAW_DIR.glob("*.csv"))
        
        if not csv_files:
            # ถ้ายังไม่มีไฟล์ในเครื่อง ให้ลองดึงผ่าน URL ตรงเข้า View/Table ทันที
            base_url = "https://github.com/PoohLbk/Engineer_data_agent/releases/download/v1.0"
            con.execute(f"CREATE OR REPLACE TABLE bronze_orders AS SELECT * FROM '{base_url}/order_items.csv'")
            con.execute(f"CREATE OR REPLACE TABLE customer_master AS SELECT * FROM '{base_url}/customer_master.csv'")
            con.execute(f"CREATE OR REPLACE TABLE product_catalog AS SELECT * FROM '{base_url}/product_catalog.csv'")
            con.execute(f"CREATE OR REPLACE TABLE ecommerce_sales AS SELECT * FROM '{base_url}/ecommerce_sales_customer_analytics_150k.csv'")
        else:
            # ถ้ามีไฟล์ในโฟลเดอร์ raw/ ให้อ่านเข้ามาสร้างเป็น Table
            for file_path in csv_files:
                table_name = "bronze_" + file_path.stem.lower().replace(" ", "_")
                con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")
                summary["tables_created"].append(table_name)
                
        # สร้างตาราง log สำหรับเก็บบันทึกการรัน
        con.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT, status TEXT, executed_at TEXT
            )
        """)
        now_str = datetime.now(timezone.utc).isoformat()
        con.execute("INSERT INTO pipeline_runs VALUES (?, ?, ?)", (run_id, "success", now_str))
        
        print("✅ Pipeline รันสำเร็จและสร้างตารางใน DuckDB เรียบร้อยแล้ว")
        
    except Exception as e:
        summary["status"] = "failed"
        summary["error"] = str(e)
        raise e
    finally:
        con.close()
        
    return summary

if __name__ == "__main__":
    import requests
    download_sample_data()
    run_pipeline()
