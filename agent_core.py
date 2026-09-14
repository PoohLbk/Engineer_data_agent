import os
import duckdb
import streamlit as st
from google import genai

class EnterpriseDataAgent:
    def __init__(self):
        # 1. จัดการดึง GEMINI_API_KEY จาก Streamlit Secrets หรือ Environment Variable
        api_key = None
        try:
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
            else:
                api_key = os.environ.get("GEMINI_API_KEY")
        except Exception:
            api_key = os.environ.get("GEMINI_API_KEY")

        # 2. เริ่มต้นสร้าง Gemini Client (ป้องกัน AttributeError: 'client')
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None
            print("Warning: GEMINI_API_KEY not found.")

        # 3. เชื่อมต่อ DuckDB ใน Memory
        self.con = duckdb.connect(database=':memory:')
        
        # 4. โหลดไฟล์ข้อมูล 5 ตารางจาก GitHub Release v1.0
        base_url = "https://github.com/PoohLbk/Engineer_data_agent/releases/download/v1.0"
        
        files_to_load = {
            "customer_master": f"{base_url}/customer_master.csv",
            "dataset_statistics": f"{base_url}/dataset_statistics.csv",
            "ecommerce_sales": f"{base_url}/ecommerce_sales_customer_analytics_150k.csv",
            "order_items": f"{base_url}/order_items.csv",
            "product_catalog": f"{base_url}/product_catalog.csv"
        }
        
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
        
        if not self.client:
            logs.append("Execution Error: GEMINI_API_KEY is missing.")
            return None, "", logs
        
        # 1. รวบรวม Schema ของตารางทั้งหมดส่งให้ LLM
        schema_info = ""
        tables = self.con.execute("SHOW TABLES").fetchall()
        for t in tables:
            t_name = t[0]
            cols = self.con.execute(f"DESCRIBE {t_name}").fetchall()
            col_str = ", ".join([f"{c[0]} ({c[1]})" for c in cols])
            schema_info += f"Table {t_name}: {col_str}\n"

        # 2. สร้าง Prompt ส่งให้ Gemini
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
            model='gemini-1.5-flash',
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

    def generate_executive_summary(self, user_query, df_result):
        """
        ฟังก์ชันใช้ Gemini สรุปผลลัพธ์ข้อมูลจาก DataFrame ให้อยู่ในรูปแบบคำอธิบายสำหรับผู้บริหาร
        """
        if df_result is None or df_result.empty:
            return "ไม่พบข้อมูลสำหรับสรุปผลลัพธ์"

        if not self.client:
            return "ไม่สามารถสรุปผลลัพธ์ได้เนื่องจากขาด GEMINI_API_KEY"

        data_preview = df_result.head(20).to_string(index=False)
        
        prompt = f"""
        คุณเป็น Data Analyst ผู้เชี่ยวชาญ กรุณาสรุปผลลัพธ์จากข้อมูลด้านล่างนี้ เพื่อตอบคำถามของผู้ใช้:

        คำถามของผู้ใช้: "{user_query}"

        ผลลัพธ์ข้อมูลที่ได้จาก Database:
        {data_preview}

        คำแนะนำในการตอบ:
        1. อธิบายคำตอบหลักให้ชัดเจน ตรงประเด็น
        2. สรุปจุดสำคัญหรือ Insight ที่น่าสนใจจากข้อมูล เป็นข้อๆ (Bullet points)
        3. ตอบเป็นภาษาไทยที่สุภาพ เข้าใจง่าย และเป็นทางการ
        """
        
        try:
            rresponse = self.client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"เกิดข้อผิดพลาดในการสร้างสรุปผลลัพธ์: {str(e)}"
