import os
import duckdb

class EnterpriseDataAgent:
    def __init__(self):
        # 1. เชื่อมต่อ DuckDB ใน Memory
        self.con = duckdb.connect(database=':memory:')
        
        # 2. รวบรวม Direct Link ของทั้ง 5 ไฟล์จาก GitHub Release v1.0
        base_url = "https://github.com/PoohLbk/Engineer_data_agent/releases/download/v1.0"
        
        files_to_load = {
            "customer_master": f"{base_url}/customer_master.csv",
            "dataset_statistics": f"{base_url}/dataset_statistics.csv",
            "ecommerce_sales": f"{base_url}/ecommerce_sales_customer_analytics_150k.csv",
            "order_items": f"{base_url}/order_items.csv",
            "product_catalog": f"{base_url}/product_catalog.csv"
        }
        
        # 3. วนลูปอ่านข้อมูลทุกไฟล์เข้า DuckDB แยกตามชื่อตาราง
        for table_name, url in files_to_load.items():
            try:
                self.con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM '{url}'")
            except Exception as e:
                print(f"Error loading {table_name}: {e}")

        # โหลดไฟล์ local เพิ่มเติม (ถ้ามี)
        self._load_local_data()
