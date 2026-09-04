import os
import re
import duckdb
import pandas as pd
import ollama
from dotenv import load_dotenv

load_dotenv()

class EnterpriseDataAgent:
    def __init__(self):
        self.data_folder = os.getenv("DATA_FOLDER", "./data_input")
        self.model_name = os.getenv("MODEL_NAME", "qwen2.5-coder:7b")
        self.ollama_client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        
        # Isolated In-Memory Engine for Execution
        self.con = duckdb.connect(database=':memory:')
        self.registered_tables = []
        self.auto_load_files()

    def auto_load_files(self):
        """ระบบ Automatic Ingestion อ่านไฟล์ในเครื่องเข้า Database ใน Memory"""
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            
        for file in os.listdir(self.data_folder):
            file_path = os.path.join(self.data_folder, file)
            if os.path.isdir(file_path):
                continue
                
            table_name = os.path.splitext(file)[0].lower()
            table_name = re.sub(r'\W+', '_', table_name)

            try:
                if file.endswith('.csv'):
                    self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")
                    self.registered_tables.append(table_name)
                elif file.endswith('.parquet'):
                    self.con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{file_path}')")
                    self.registered_tables.append(table_name)
            except Exception as e:
                print(f"Error loading {file}: {e}")

    def get_database_schema(self):
        """ดึง Schema ของทุกตารางในระบบ"""
        schema_text = ""
        for table in self.registered_tables:
            columns_info = self.con.execute(f"DESCRIBE {table}").fetchall()
            cols_str = ", ".join([f"{col[0]} ({col[1]})" for col in columns_info])
            schema_text += f"Table: {table}\nColumns: {cols_str}\n\n"
        return schema_text

    def clean_sql(self, raw_llm_output):
        """ล้าง Markdown Formatting ให้เหลือเฉพาะคำสั่ง SQL"""
        sql_match = re.search(r"```sql\s*(.*?)\s*```", raw_llm_output, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        return raw_llm_output.replace("```", "").strip()

    def execute_with_self_correction(self, user_query):
        """Agentic Self-Correction Loop ลูปซ่อมแซมคำสั่ง SQL อัตโนมัติเมื่อเกิด Error"""
        max_retries = int(os.getenv("MAX_RETRIES", 3))
        schema_info = self.get_database_schema()
        logs = []
        
        prompt = f"""You are an Enterprise Data Engineer. Write ONLY a valid DuckDB SQL query to answer the user request.
Do not include markdown text explanations outside code blocks.

Database Schema:
{schema_info}

User Request: {user_query}
SQL Query:"""

        current_prompt = prompt
        
        for attempt in range(max_retries):
            logs.append(f"--- Processing Attempt {attempt + 1} ---")
            
            response = self.ollama_client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': current_prompt}]
            )
            
            raw_response = response['message']['content']
            sql_query = self.clean_sql(raw_response)
            logs.append(f"Generated SQL:\n{sql_query}")

            try:
                result_df = self.con.execute(sql_query).df()
                logs.append("Status: Query Executed Successfully")
                return result_df, sql_query, logs
            except Exception as e:
                error_msg = str(e)
                logs.append(f"Status: Error Encountered -> {error_msg}")
                
                # Dynamic Feedback Prompt for Self-Correction
                current_prompt = f"""The previous DuckDB SQL query failed with an error. Fix it.

User Request: {user_query}
Database Schema:
{schema_info}

Failed SQL Query:
{sql_query}

Error Log:
{error_msg}

Return ONLY the corrected SQL query:"""

        return None, None, logs

    def generate_executive_summary(self, user_query, result_df):
        """วิเคราะห์และสรุปผลลัพธ์เป็นภาษาไทยเชิงบริหาร"""
        if result_df is None or result_df.empty:
            return "ไม่พบข้อมูลที่ตรงกับเงื่อนไขคำสั่ง"
            
        sample_data = result_df.head(5).to_string()
        prompt = f"""คุณเป็น Data Analyst ประจำองค์กร จงสรุปผลลัพธ์จากข้อมูลนี้เป็นภาษาไทยเชิงธุรกิจ สั้น กระชับ และตรงประเด็น:

คำถามผู้บริหาร: {user_query}
ข้อมูลสรุป (Top Rows):
{sample_data}

สรุปผลการวิเคราะห์:"""

        response = self.ollama_client.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
