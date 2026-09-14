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
    def execute_with_self_correction(self, user_query):
        """
        ฟังก์ชันสำหรับแปลงภาษาธรรมชาติเป็น SQL และรันคำสั่งบน DuckDB
        """
        logs = []
        logs.append(f"Received query: {user_query}")
        
        # 1. รวบรวม Schema ของตารางทั้งหมดส่งให้ LLM
        schema_info = ""
        tables = self.con.execute("SHOW TABLES").fetchall()
        for t in tables:
            t_name = t[0]
            cols = self.con.execute(f"DESCRIBE {t_name}").fetchall()
            col_str = ", ".join([f"{c[0]} ({c[1]})" for c in cols])
            schema_info += f"Table {t_name}: {col_str}\n"

        # 2. สร้าง Prompt ส่งให้ Gemini (หรือ LLM ที่ใช้งาน)
        prompt = f"""
        You are a DuckDB SQL expert. Given the following schema:
        {schema_info}
        
        Write a valid DuckDB SQL query to answer this user request:
        "{user_query}"
        
        Return ONLY the raw SQL query without codeblock formatting or explanations.
        """
        
        try:
            # แปลงคำถามเป็น SQL
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            sql_query = response.text.strip().replace("```sql", "").replace("```", "").strip()
            logs.append(f"Generated SQL: {sql_query}")
            
            # 3. รันคำสั่ง SQL บน DuckDB
            df_result = self.con.execute(sql_query).df()
            return df_result, sql_query, logs

        except Exception as e:
            error_msg = f"Execution Error: {str(e)}"
            logs.append(error_msg)
            return None, "", logs
