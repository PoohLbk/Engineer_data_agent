import os
import duckdb
import streamlit as st
from google import genai

# ==========================================
# 1. Class: EnterpriseDataAgent (Data Engine)
# ==========================================
class EnterpriseDataAgent:
    def __init__(self):
        # 1. ดึง GEMINI_API_KEY
        api_key = None
        try:
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
            else:
                api_key = os.environ.get("GEMINI_API_KEY")
        except Exception:
            api_key = os.environ.get("GEMINI_API_KEY")

        # 2. สร้าง Gemini Client
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None
            print("Warning: GEMINI_API_KEY not found.")

        # ตั้งค่าโมเดลหลักและโมเดลสำรอง
        self.primary_model = "gemini-1.5-flash"
        self.fallback_model = "gemini-1.5-pro"

        # 3. เชื่อมต่อ DuckDB ใน Memory
        self.con = duckdb.connect(database=':memory:')
        
        # 4. โหลดข้อมูลแบบ VIEW (Lazy Load เปิดแอปเร็ว)
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
                self.con.execute(f"CREATE VIEW IF NOT EXISTS {table_name} AS SELECT * FROM '{url}'")
            except Exception as e:
                print(f"Error loading {table_name}: {e}")

    def _call_gemini_with_fallback(self, prompt):
        """เรียกใช้งาน API หากโมเดลหลักค้าง/หนาแน่น จะสลับไปใช้โมเดลสำรองให้อัตโนมัติ"""
        try:
            response = self.client.models.generate_content(
                model=self.primary_model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Primary model failed ({e}), retrying with fallback model...")
            response = self.client.models.generate_content(
                model=self.fallback_model,
                contents=prompt
            )
            return response.text

    def execute_with_self_correction(self, user_query):
        logs = [f"Received query: {user_query}"]
        
        if not self.client:
            logs.append("Execution Error: GEMINI_API_KEY is missing.")
            return None, "", logs
        
        schema_info = ""
        tables = self.con.execute("SHOW TABLES").fetchall()
        for t in tables:
            t_name = t[0]
            cols = self.con.execute(f"DESCRIBE {t_name}").fetchall()
            col_str = ", ".join([f"{c[0]} ({c[1]})" for c in cols])
            schema_info += f"Table {t_name}: {col_str}\n"

        prompt = f"""
        You are a DuckDB SQL expert. Given the following schema:
        {schema_info}
        
        Write a valid DuckDB SQL query to answer this user request:
        "{user_query}"
        
        Return ONLY the raw SQL query without codeblock formatting or explanations.
        """
        
        try:
            raw_response = self._call_gemini_with_fallback(prompt)
            sql_query = raw_response.strip().replace("```sql", "").replace("```", "").strip()
            logs.append(f"Generated SQL: {sql_query}")
            
            df_result = self.con.execute(sql_query).df()
            return df_result, sql_query, logs

        except Exception as e:
            error_msg = f"Execution Error: {str(e)}"
            logs.append(error_msg)
            return None, "", logs

    def generate_executive_summary(self, user_query, df_result):
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
            summary_text = self._call_gemini_with_fallback(prompt)
            return summary_text.strip()
        except Exception as e:
            return f"เกิดข้อผิดพลาดในการสร้างสรุปผลลัพธ์: {str(e)}"

# ==========================================
# 2. Streamlit Web UI Application
# ==========================================
st.set_page_config(page_title="Enterprise Data Agent", layout="wide")

@st.cache_resource
def load_agent():
    return EnterpriseDataAgent()

with st.spinner("กำลังเชื่อมต่อฐานข้อมูล และสร้าง Data Engine Views..."):
    agent = load_agent()

st.title("Enterprise Data Agent")

tab1, tab2 = st.tabs(["💬 AI Query Engine", "🔍 Data Schema Explorer"])

with tab1:
    user_query = st.text_input("พิมพ์คำถามของคุณที่นี่ (เช่น: ขอ 5 อันดับสินค้าที่มียอดขายรวมสูงสุด):")
    
    if st.button("ประมวลผลคำสั่ง"):
        if user_query:
            with st.spinner("กำลังสร้างคำสั่ง SQL และดึงข้อมูล..."):
                df_result, final_sql, logs = agent.execute_with_self_correction(user_query)
            
            if df_result is not None and not df_result.empty:
                st.subheader("💡 บทสรุปการวิเคราะห์ (Executive Summary)")
                with st.spinner("กำลังวิเคราะห์และสรุป Insight..."):
                    summary = agent.generate_executive_summary(user_query, df_result)
                st.write(summary)
                
                st.subheader("📊 ผลลัพธ์ตารางข้อมูล (Query Results)")
                st.dataframe(df_result, use_container_width=True)
            else:
                st.warning("ไม่พบข้อมูล หรือเกิดข้อผิดพลาดในการรัน SQL")
            
            with st.expander("🔍 Audit Logs & Generated SQL Pipeline (สำหรับงานเทคนิค)"):
                st.code(final_sql, language="sql")
                for log in logs:
                    st.write(log)

with tab2:
    st.subheader("📋 Schema และตัวอย่างข้อมูลของฐานข้อมูลทั้งหมด")
    search_term = st.text_input("🔍 ค้นหาชื่อตาราง หรือ ชื่อคอลัมน์:")
    
    tables = agent.con.execute("SHOW TABLES").fetchall()
    for t in tables:
        t_name = t[0]
        cols = agent.con.execute(f"DESCRIBE {t_name}").fetchall()
        
        col_names = [c[0] for c in cols]
        if search_term.lower() in t_name.lower() or any(search_term.lower() in c.lower() for c in col_names):
            with st.expander(f"📌 Table: {t_name}", expanded=False):
                st.markdown("**📌 Data Types & Schema:**")
                st.dataframe(
                    [{"Column Name": c[0], "Data Type": c[1]} for c in cols],
                    use_container_width=True
                )
                
                st.markdown("**👀 Sample Data (Top 3 rows):**")
                try:
                    sample_df = agent.con.execute(f"SELECT * FROM {t_name} LIMIT 3").df()
                    st.dataframe(sample_df, use_container_width=True)
                except Exception as e:
                    st.error(f"ไม่สามารถโหลดตัวอย่างข้อมูลได้: {e}")
