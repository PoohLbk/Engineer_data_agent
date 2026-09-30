import os
import sys
import sqlite3
import time
import duckdb
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import traceback
from plotly.subplots import make_subplots
from google import genai

# ==========================================
# 0. นำเข้า Data Pipeline (พร้อมระบบ Debug Error)
# ==========================================
# เพิ่ม Path ปัจจุบันเข้าไปให้ Python รู้จักโฟลเดอร์รอบข้าง
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # พยายามนำเข้าจากโฟลเดอร์ของคุณ
    # ถ้ายัง Error อีก ให้เช็คโครงสร้างไฟล์ว่าถูกต้องตามที่คุยกันไว้หรือไม่
    from data_pipeline.pipeline.runner import run_pipeline
    HAS_PIPELINE = True
    PIPELINE_ERROR = ""
except Exception as e:
    HAS_PIPELINE = False
    PIPELINE_ERROR = traceback.format_exc()

# ==========================================
# 1. Feedback & Rating System
# ==========================================
FEEDBACK_DB_PATH = os.environ.get("FEEDBACK_DB_PATH", "feedback.db")

def _get_feedback_conn():
    conn = sqlite3.connect(FEEDBACK_DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_query TEXT NOT NULL,
            generated_sql TEXT,
            engine TEXT NOT NULL,
            local_model TEXT,
            rating TEXT NOT NULL,
            row_count INTEGER
        )
        """
    )
    conn.commit()
    return conn

def save_feedback(user_query, generated_sql, engine, local_model, rating, row_count):
    conn = _get_feedback_conn()
    try:
        conn.execute(
            "INSERT INTO feedback (timestamp, user_query, generated_sql, engine, local_model, rating, row_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pd.Timestamp.now().isoformat(), user_query, generated_sql, engine, local_model, rating, row_count),
        )
        conn.commit()
    finally:
        conn.close()

def get_feedback_stats():
    conn = _get_feedback_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM feedback ORDER BY timestamp DESC", conn)
    finally:
        conn.close()
    return df

# ==========================================
# 2. Local Engine Config & Data Viz Tools
# ==========================================
LOCAL_MODEL_INFO = {
    "qwen2.5-coder:7b": {
        "display_name": "Qwen2.5-Coder 7B",
        "recommended": True,
        "accuracy_label": "19.33% (29/150)",
        "empty_response_label": "0/150 — ไม่พบอาการตอบกลับว่างเปล่าเลย",
        "note": "Best Operational Local Model: ความแม่นยำต่ำกว่า CodeLlama เล็กน้อย แต่เสถียรที่สุด ตอบกลับครบทุกข้อ",
    },
    "codellama:7b": {
        "display_name": "CodeLlama 7B",
        "recommended": False,
        "accuracy_label": "22.00% (33/150) — แม่นยำสูงสุดในกลุ่ม Local",
        "empty_response_label": "65/150 (~43%) — พบอัตราตอบกลับว่างเปล่าสูง",
        "note": "แม่นยำที่สุดแต่เสี่ยง Empty Response สูง เหมาะกับงานทดลอง/เปรียบเทียบเชิงวิจัยมากกว่าใช้งานจริง",
    },
}

def _find_datetime_col(df: pd.DataFrame):
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    if not datetime_cols:
        for col in df.select_dtypes(include="object").columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="raise")
                datetime_cols.append(col)
                break
            except Exception:
                continue
    return datetime_cols[0] if datetime_cols else None

def generate_charts(df: pd.DataFrame):
    charts = []
    if df is None or df.empty or df.shape[1] < 2:
        return charts

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return charts 

    y_col = numeric_cols[0]
    date_col = _find_datetime_col(df)
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols and c != date_col]

    if date_col:
        df_sorted = df.sort_values(by=date_col)
        fig = px.line(df_sorted, x=date_col, y=numeric_cols, markers=True, title=f"แนวโน้ม {', '.join(numeric_cols)} ตาม {date_col}")
        charts.append(("แนวโน้มตามช่วงเวลา (Trend)", fig))

    if non_numeric_cols:
        cat_col = non_numeric_cols[0]
        grouped = df.groupby(cat_col, as_index=False)[y_col].sum().sort_values(y_col, ascending=False)
        bar_df = grouped.head(15)
        fig_bar = px.bar(bar_df, x=cat_col, y=y_col, title=f"{y_col} แยกตาม {cat_col} (Top {len(bar_df)})")
        fig_bar.update_layout(xaxis_tickangle=-30)
        charts.append(("เปรียบเทียบตามหมวดหมู่ (Bar)", fig_bar))

        if grouped[cat_col].nunique() <= 8:
            fig_pie = px.pie(grouped, names=cat_col, values=y_col, title=f"สัดส่วน {y_col} ตาม {cat_col}", hole=0.35)
            charts.append(("สัดส่วนโดยรวม (Pie)", fig_pie))

        if grouped[cat_col].nunique() >= 3:
            pareto_df = grouped.head(15).reset_index(drop=True)
            total = pareto_df[y_col].sum()
            if total:
                pareto_df["cum_pct"] = pareto_df[y_col].cumsum() / total * 100
                fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
                fig_pareto.add_trace(go.Bar(x=pareto_df[cat_col].astype(str), y=pareto_df[y_col], name=y_col), secondary_y=False)
                fig_pareto.add_trace(go.Scatter(x=pareto_df[cat_col].astype(str), y=pareto_df["cum_pct"], name="สัดส่วนสะสม (%)", mode="lines+markers"), secondary_y=True)
                fig_pareto.add_hline(y=80, line_dash="dot", secondary_y=True, annotation_text="เส้น 80%")
                fig_pareto.update_layout(title=f"Pareto Analysis: {y_col} ตาม {cat_col} (กฎ 80/20)")
                fig_pareto.update_yaxes(title_text=y_col, secondary_y=False)
                fig_pareto.update_yaxes(title_text="สัดส่วนสะสม (%)", range=[0, 110], secondary_y=True)
                charts.append(("Pareto Analysis (80/20)", fig_pareto))

    if not date_col and not non_numeric_cols and len(numeric_cols) >= 2:
        fig_scatter = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"{numeric_cols[1]} เทียบกับ {numeric_cols[0]}")
        charts.append(("ความสัมพันธ์ระหว่างตัวแปร (Scatter)", fig_scatter))

    return charts

def compute_statistical_insights(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "ไม่มีข้อมูลเพียงพอสำหรับวิเคราะห์เชิงสถิติ"
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return "ไม่มีคอลัมน์ตัวเลขในผลลัพธ์นี้ จึงไม่สามารถวิเคราะห์เชิงสถิติเพิ่มเติมได้"

    main_col = numeric_cols[0]
    series = df[main_col].dropna()
    lines = []

    if series.empty:
        return "ไม่มีข้อมูลตัวเลขเพียงพอสำหรับวิเคราะห์"

    lines.append(f"สถิติเบื้องต้นของคอลัมน์ '{main_col}': ค่าเฉลี่ย={series.mean():,.2f}, มัธยฐาน={series.median():,.2f}, ส่วนเบี่ยงเบนมาตรฐาน={series.std():,.2f}, ต่ำสุด={series.min():,.2f}, สูงสุด={series.max():,.2f} (n={len(series)})")

    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    lines.append(f"Outlier Detection (กฎ IQR) บนคอลัมน์ '{main_col}': พบ {len(outliers)} รายการ จากทั้งหมด {len(series)} แถว (ช่วงปกติโดยประมาณ {lower:,.2f} ถึง {upper:,.2f})")

    date_col_probe = _find_datetime_col(df)
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols and c != date_col_probe]
    if non_numeric_cols:
        cat_col = non_numeric_cols[0]
        grouped = df.groupby(cat_col)[main_col].sum().sort_values(ascending=False)
        total = grouped.sum()
        if total and len(grouped) > 0:
            cum_pct = grouped.cumsum() / total * 100
            n_items_80 = int((cum_pct < 80).sum()) + 1
            n_items_80 = min(n_items_80, len(grouped))
            pct_of_items = n_items_80 / len(grouped) * 100
            lines.append(f"Pareto Analysis (80/20) บน '{cat_col}': มีเพียง {n_items_80} รายการ (คิดเป็น {pct_of_items:.1f}% ของ {cat_col} ทั้งหมด {len(grouped)} รายการ) ที่รวมกันสร้าง '{main_col}' ได้ถึงประมาณ 80% ของยอดรวมทั้งหมด")

    if date_col_probe:
        df_sorted = df.sort_values(by=date_col_probe)
        mid = len(df_sorted) // 2
        if mid > 0:
            first_half = df_sorted.iloc[:mid][main_col].sum()
            second_half = df_sorted.iloc[mid:][main_col].sum()
            if first_half != 0:
                growth = (second_half - first_half) / abs(first_half) * 100
                lines.append(f"อัตราการเติบโตของ '{main_col}' เมื่อเทียบครึ่งแรกกับครึ่งหลังของช่วงข้อมูล (เรียงตาม '{date_col_probe}'): เปลี่ยนแปลง {growth:+.1f}%")

    return "\n".join(lines)

# ==========================================
# 3. Class: EnterpriseDataAgent (Data Engine)
# ==========================================
class OllamaConnectionError(RuntimeError):
    pass

class EnterpriseDataAgent:
    def __init__(self, api_key: str = None):
        if not api_key:
            try:
                if "GEMINI_API_KEY" in st.secrets:
                    api_key = st.secrets["GEMINI_API_KEY"]
                else:
                    api_key = os.environ.get("GEMINI_API_KEY")
            except Exception:
                api_key = os.environ.get("GEMINI_API_KEY")

        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None
            print("Warning: GEMINI_API_KEY not found.")

        self.primary_model = "gemini-3.6-flash"
        self.fallback_model = "gemini-3.5-flash"
        self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

        # ----------------------------------------------------
        # การเชื่อมต่อ Database (รองรับ Pipeline Integration)
        # ----------------------------------------------------
        db_path = "data_pipeline/data/warehouse.db" 
        
        try:
            if os.path.exists(db_path):
                self.con = duckdb.connect(database=db_path, read_only=True)
                print(f"🔗 เชื่อมต่อฐานข้อมูล Local จาก Pipeline สำเร็จ ({db_path})")
            else:
                self.con = duckdb.connect(database=':memory:')
                print("⚠️ ไม่พบไฟล์ Database จาก Pipeline กำลังใช้ In-Memory และดึงข้อมูลจาก GitHub")
                self._load_fallback_data()
        except Exception as e:
            print(f"Error connecting to DB: {e}. Fallback to memory.")
            self.con = duckdb.connect(database=':memory:')
            self._load_fallback_data()

    def _load_fallback_data(self):
        """โหลดข้อมูลแบบ View จาก GitHub"""
        
        # ⚠️ อัปเดตลิงก์ตรงนี้ให้ตรงกับหน้า Release ล่าสุด
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
        try:
            response = self.client.models.generate_content(model=self.primary_model, contents=prompt)
            return response.text
        except Exception as e:
            print(f"Primary model failed ({e}), retrying with fallback model...")
            response = self.client.models.generate_content(model=self.fallback_model, contents=prompt)
            return response.text

    def _call_ollama(self, prompt, model_name, max_empty_retries=1, base_url=None):
        url = (base_url or self.ollama_base_url or "").strip().rstrip("/")
        last_err_detail = "unknown error"
        if not url or (not url.startswith("http://") and not url.startswith("https://")):
            raise OllamaConnectionError(f"Ollama Server URL '{url}' ไม่ถูกต้อง")

        headers = {"ngrok-skip-browser-warning": "true", "Bypass-Tunnel-Reminder": "true", "X-Pinggy-No-Screen": "true"}

        for attempt in range(max_empty_retries + 1):
            try:
                resp = requests.post(
                    f"{url}/api/generate", json={"model": model_name, "prompt": prompt, "stream": False},
                    headers=headers, timeout=120,
                )
                resp.raise_for_status()
                text = resp.json().get("response", "").strip()
                if text: return text
                last_err_detail = "Empty Response จากโมเดล Local"
            except requests.exceptions.ConnectionError as e:
                raise OllamaConnectionError(f"เชื่อมต่อ Ollama ไม่ได้: {e}")
            except Exception as e:
                raise RuntimeError(f"Ollama API error: {e}")
        raise RuntimeError(last_err_detail)

    def _call_llm(self, prompt, engine="cloud", local_model=None, ollama_url=None):
        if engine == "local":
            if not local_model: raise ValueError("ต้องระบุ local_model เมื่อเลือกใช้ Local Engine")
            return self._call_ollama(prompt, local_model, base_url=ollama_url)
        return self._call_gemini_with_fallback(prompt)

    # ----------------------------------------------------
    # RAG Schema Router: เลือกตารางที่เกี่ยวข้องก่อนเขียน SQL
    # ----------------------------------------------------
    def _route_tables(self, user_query, all_table_names, engine, local_model, ollama_url):
        prompt = f"""
        คุณคือ Data Architect หน้าที่ของคุณคือเลือกตารางที่จำเป็นสำหรับตอบคำถาม
        ตารางในฐานข้อมูลทั้งหมดมีดังนี้: {', '.join(all_table_names)}
        
        คำถามของผู้ใช้: "{user_query}"
        
        จงเลือกเฉพาะชื่อตารางที่เกี่ยวข้องจริงๆ ตอบมาเป็นชื่อตารางคั่นด้วยเครื่องหมายจุลภาค (comma)
        ห้ามอธิบายเพิ่มเติม ห้ามมีเครื่องหมายคำพูด หรือ Markdown ใดๆ
        """
        try:
            raw_response = self._call_llm(prompt, engine=engine, local_model=local_model, ollama_url=ollama_url)
            selected = [t.strip() for t in raw_response.split(',')]
            valid_selected = [t for t in selected if t in all_table_names]
            if not valid_selected:
                return all_table_names
            return valid_selected
        except Exception:
            return all_table_names

    def execute_with_self_correction(self, user_query, engine="cloud", local_model=None, ollama_url=None, max_attempts=3):
        logs = [f"Received query: {user_query}", f"Engine: {engine}" + (f" ({local_model})" if local_model else "")]
        if engine == "local": logs.append(f"Ollama URL ที่ใช้จริง: {ollama_url!r}")

        if engine == "cloud" and not self.client:
            logs.append("Execution Error: GEMINI_API_KEY is missing.")
            return None, "", logs

        # ----------------------------------------------------
        # การใช้งาน RAG Router
        # ----------------------------------------------------
        tables_query = self.con.execute("SHOW TABLES").fetchall()
        all_table_names = [t[0] for t in tables_query if t]
        
        selected_tables = self._route_tables(user_query, all_table_names, engine, local_model, ollama_url)
        logs.append(f"🔍 [RAG Router] ตารางที่ระบบเลือกใช้งาน: {', '.join(selected_tables)}")

        schema_info = ""
        for t_name in selected_tables:
            try:
                cols = self.con.execute(f"DESCRIBE {t_name}").fetchall()
                col_str = ", ".join([f"{c[0]} ({c[1]})" for c in cols if len(c) >= 2])
                schema_info += f"Table {t_name}: {col_str}\n"
            except Exception as e:
                logs.append(f"[Schema] ข้าม table '{t_name}': {e}")
                continue

        base_prompt = f"""
        You are a DuckDB SQL expert. Given the following schema:
        {schema_info}

        Write a valid DuckDB SQL query to answer this user request:
        "{user_query}"

        Return ONLY the raw SQL query without codeblock formatting or explanations.
        """

        prompt = base_prompt
        sql_query = ""
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                raw_response = self._call_llm(prompt, engine=engine, local_model=local_model, ollama_url=ollama_url)
                sql_query = raw_response.strip().replace("```sql", "").replace("
