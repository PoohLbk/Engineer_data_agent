import os
import requests
import duckdb
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = BASE_DIR / "warehouse.db"

def setup_directories():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_sample_data():
    base_url = "https://github.com/PoohLbk/Engineer_data_agent/releases/download/v1.0"
    files = {
        "customer_master.csv": f"{base_url}/customer_master.csv",
        "dataset_statistics.csv": f"{base_url}/dataset_statistics.csv",
        "ecommerce_sales_customer_analytics_150k.csv": f"{base_url}/ecommerce_sales_customer_analytics_150k.csv",
        "order_items.csv": f"{base_url}/order_items.csv",
        "product_catalog.csv": f"{base_url}/product_catalog.csv"
    }
    
    for filename, url in files.items():
        file_path = RAW_DIR / filename
        if not file_path.exists():
            try:
                print(f"กำลังดาวน์โหลด {filename} ลงสู่โฟลเดอร์ data/raw...")
                res = requests.get(url, timeout=30)
                if res.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(res.content)
            except Exception as e:
                print(f"ไม่สามารถดาวน์โหลด {filename}: {e}")

def run_pipeline(db_path=None) -> dict:
    target_db = db_path or str(DB_PATH)
    
    setup_directories()
    download_sample_data()
    
    con = duckdb.connect(database=target_db)
    
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary = {"run_id": run_id, "status": "success", "tables_created": []}
    
    try:
        csv_files = list(RAW_DIR.glob("*.csv"))
        
        if csv_files:
            for file_path in csv_files:
                table_name = "bronze_" + file_path.stem.lower().replace(" ", "_")
                con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")
                summary["tables_created"].append(table_name)
        
        if (RAW_DIR / "order_items.csv").exists():
            con.execute(f"CREATE OR REPLACE TABLE bronze_orders AS SELECT * FROM read_csv_auto('{RAW_DIR / 'order_items.csv'}')")
        
        con.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT, status TEXT, executed_at TEXT
            )
        """)
        now_str = datetime.now(timezone.utc).isoformat()
        con.execute("INSERT INTO pipeline_runs VALUES (?, ?, ?)", (run_id, "success", now_str))
        
        print("✅ Pipeline รันสำเร็จและสร้างตารางทั้งหมดใน DuckDB เรียบร้อยแล้ว")
        
    except Exception as e:
        summary["status"] = "failed"
        summary["error"] = str(e)
        raise e
    finally:
        con.close()
        
    return summary

if __name__ == "__main__":
    run_pipeline()
