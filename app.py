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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # แก้ไขพาธการนำเข้าให้ตรงกับโฟลเดอร์จริง
    from data_pipeline.runner import run_pipeline
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

        db_path = "warehouse.db" 
        
        try:
            if os.path.exists(db_path):
                self.con = duckdb.connect(database=db_path, read_only=True)
                print(f"🔗 เชื่อมต่อฐานข้อมูล Local จาก Pipeline สำเร็จ ({db_path})")
            else:
                self.con = duckdb.connect(database=':memory:')
                print("⚠️ ไม่พบไฟล์ Database กำลังใช้ In-Memory และดึงข้อมูลจาก GitHub")
                self._load_fallback_data()
        except Exception as e:
            print(f"Error connecting to DB: {e}. Fallback to memory.")
            self.con = duckdb.connect(database=':memory:')
            self._load_fallback_data()

    def _load_fallback_data(self):
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

    def _route_tables(self, user_query, all_table_names, engine, local_model, ollama_url):
        prompt = f"""
        คุณคือ Data Architect หน้าที่ของคุณคือเลือกตารางที่จำเป็นสำหรับตอบคำถาม
        ตารางในฐานข้อมูลทั้งหมดมีดังนี้: {', '.join(all_table_names)}
        คำถามของผู้ใช้: "{user_query}"
        จงเลือกเฉพาะชื่อตารางที่เกี่ยวข้องจริงๆ ตอบมาเป็นชื่อตารางคั่นด้วยเครื่องหมายจุลภาค (comma) ห้ามอธิบายเพิ่มเติม ห้ามมีเครื่องหมายคำพูด
        """
        try:
            raw_response = self._call_llm(prompt, engine=engine, local_model=local_model, ollama_url=ollama_url)
            selected = [t.strip() for t in raw_response.split(',')]
            valid_selected = [t for t in selected if t in all_table_names]
            return valid_selected if valid_selected else all_table_names
        except Exception:
            return all_table_names

    def execute_with_self_correction(self, user_query, engine="cloud", local_model=None, ollama_url=None, max_attempts=3):
        logs = [f"Received query: {user_query}", f"Engine: {engine}" + (f" ({local_model})" if local_model else "")]
        if engine == "local": logs.append(f"Ollama URL ที่ใช้จริง: {ollama_url!r}")

        if engine == "cloud" and not self.client:
            logs.append("Execution Error: GEMINI_API_KEY is missing.")
            return None, "", logs

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
        Write a valid DuckDB SQL query to answer this user request: "{user_query}"
        Return ONLY the raw SQL query without codeblock formatting or explanations.
        """

        prompt = base_prompt
        sql_query = ""
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                raw_response = self._call_llm(prompt, engine=engine, local_model=local_model, ollama_url=ollama_url)
                sql_query = raw_response.strip().replace("```sql", "").replace("```", "").strip()
                if not sql_query: raise RuntimeError("โมเดลตอบกลับว่างเปล่าไม่มี SQL ให้รัน")
                logs.append(f"[Attempt {attempt}] Generated SQL: {sql_query}")

                df_result = self.con.execute(sql_query).df()
                logs.append(f"[Attempt {attempt}] Success.")
                return df_result, sql_query, logs

            except OllamaConnectionError as e:
                last_error = str(e)
                logs.append(f"[Attempt {attempt}] Infrastructure Error: {last_error}")
                break
            except Exception as e:
                last_error = str(e)
                logs.append(f"[Attempt {attempt}] Execution Error: {last_error}")
                prompt = f"{base_prompt}\n\nคำสั่ง SQL ที่เขียนก่อนหน้า: {sql_query}\nรันแล้วเจอ error: {last_error}\nกรุณาแก้ไข SQL ให้ถูกต้อง Return ONLY raw SQL."

        logs.append(f"ล้มเหลวหลังจากพยายาม {attempt} ครั้ง: {last_error}")
        return None, sql_query, logs

    def generate_executive_summary(self, user_query, df_result, engine="cloud", local_model=None, ollama_url=None):
        if df_result is None or df_result.empty: return "ไม่พบข้อมูลสำหรับสรุปผลลัพธ์"
        if engine == "cloud" and not self.client: return "ไม่สามารถสรุปผลลัพธ์ได้เนื่องจากขาด GEMINI_API_KEY"
        if engine == "local" and not local_model: return "ไม่สามารถสรุปผลลัพธ์ได้เนื่องจากยังไม่ได้เลือกโมเดล Local Engine"

        data_preview = df_result.head(20).to_string(index=False)
        stats_block = compute_statistical_insights(df_result)

        prompt = f"""
        คุณเป็น Data Analyst ผู้เชี่ยวชาญ กรุณาสรุปผลลัพธ์จากข้อมูลนี้เพื่อตอบคำถาม:
        คำถาม: "{user_query}"
        ตัวอย่างข้อมูล 20 แถว: {data_preview}
        สถิติเชิงลึก: {stats_block}
        ตอบเป็นภาษาไทย เป็นทางการ มี Bullet points ชัดเจน
        """
        try:
            summary_text = self._call_llm(prompt, engine=engine, local_model=local_model, ollama_url=ollama_url)
            return summary_text.strip() if summary_text.strip() else "โมเดลตอบกลับว่างเปล่า"
        except Exception as e:
            return f"เกิดข้อผิดพลาด: {str(e)}"

# ==========================================
# 4. Streamlit Web UI Application
# ==========================================
st.set_page_config(page_title="Enterprise Data Agent", layout="wide")

COLOR_BG = "#0F1620"
COLOR_SURFACE = "#161F2E"
COLOR_BORDER = "#28344A"
COLOR_TEXT = "#E8ECF3"
COLOR_TEXT_MUTED = "#8B9BB4"
COLOR_ACCENT = "#C9A227"
COLOR_ACCENT_SOFT = "#3FA796"
PLOTLY_COLORWAY = [COLOR_ACCENT, COLOR_ACCENT_SOFT, "#7D8FB3", "#C1584C", "#5B7FA6"]

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ background-color: {COLOR_BG} !important; color: {COLOR_TEXT}; font-family: 'IBM Plex Sans', sans-serif; }}
[data-testid="stHeader"] {{ background-color: transparent !important; }}
[data-testid="stAppViewContainer"] .main .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
h1, h2, h3 {{ font-family: 'Source Serif 4', serif !important; color: {COLOR_TEXT} !important; font-weight: 600 !important; }}
.exec-header {{ margin-bottom: 1.6rem; }}
.exec-header .eyebrow {{ font-family: 'IBM Plex Sans', sans-serif; font-size: 0.82rem; color: {COLOR_ACCENT_SOFT}; margin-bottom: 2px; }}
.exec-header h1 {{ font-size: 2.1rem !important; margin: 0 0 4px 0 !important; }}
.exec-header p {{ color: {COLOR_TEXT_MUTED}; font-size: 0.95rem; margin: 0; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 4px 0 26px 0; }}
.kpi-card {{ background: {COLOR_SURFACE}; border: 1px solid {COLOR_BORDER}; border-left: 3px solid {COLOR_ACCENT}; border-radius: 6px; padding: 16px 20px; }}
.kpi-card .kpi-label {{ font-size: 0.76rem; color: {COLOR_TEXT_MUTED}; margin-bottom: 6px; }}
.kpi-card .kpi-value {{ font-family: 'Source Serif 4', serif; font-size: 1.75rem; font-variant-numeric: tabular-nums; line-height: 1.15; color: {COLOR_TEXT}; }}
.kpi-card .kpi-sub {{ font-size: 0.74rem; color: {COLOR_ACCENT_SOFT}; margin-top: 6px; }}
[data-testid="stTextInput"] input {{ background-color: {COLOR_SURFACE} !important; color: {COLOR_TEXT} !important; border: 1px solid {COLOR_BORDER} !important; border-radius: 6px !important; }}
[data-testid="stButton"] button {{ background-color: {COLOR_ACCENT} !important; color: #FFFFFF !important; border: none !important; border-radius: 6px !important; font-weight: 600 !important; padding: 0.5rem 1.4rem !important; }}
[data-testid="stExpander"] {{ background-color: {COLOR_SURFACE}; border: 1px solid {COLOR_BORDER} !important; border-radius: 6px !important; }}
[data-testid="stDataFrame"] {{ border: 1px solid {COLOR_BORDER}; border-radius: 6px; }}
.privacy-badge {{ display: inline-block; font-size: 0.78rem; font-family: 'IBM Plex Sans', sans-serif; padding: 5px 12px; border-radius: 20px; margin: 6px 0 2px 0; }}
.privacy-badge.local {{ background: rgba(63, 167, 150, 0.15); color: {COLOR_ACCENT_SOFT}; border: 1px solid {COLOR_ACCENT_SOFT}; }}
.privacy-badge.cloud {{ background: rgba(193, 88, 76, 0.12); color: #D98A80; border: 1px solid #C1584C; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def format_kpi_number(x):
    if x is None: return "-"
    abs_x = abs(x)
    if abs_x >= 1_000_000: return f"{x/1_000_000:,.2f}M"
    if abs_x >= 1_000: return f"{x:,.0f}"
    return f"{x:,.2f}"

def compute_kpis(df: pd.DataFrame):
    kpis = [("จำนวนแถวผลลัพธ์", f"{len(df):,}", "")]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols[:2]:
        kpis.append((f"รวม {col}", format_kpi_number(df[col].sum()), f"เฉลี่ย {format_kpi_number(df[col].mean())} / แถว"))
    id_like_keywords = ["customer", "order", "user", "ลูกค้า"]
    for col in df.columns:
        if any(kw in col.lower() for kw in id_like_keywords) and len(kpis) < 4:
            kpis.append((f"จำนวน {col} ไม่ซ้ำ", f"{df[col].nunique():,}", ""))
    return kpis[:4]

def render_kpi_cards(df: pd.DataFrame):
    kpis = compute_kpis(df)
    cards_html = "".join(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>' for label, value, sub in kpis)
    st.markdown(f'<div class="kpi-grid">{cards_html}</div>', unsafe_allow_html=True)

def style_chart_theme(fig):
    fig.update_layout(paper_bgcolor=COLOR_SURFACE, plot_bgcolor=COLOR_SURFACE, font=dict(family="IBM Plex Sans", color=COLOR_TEXT), title_font=dict(family="Source Serif 4", size=18, color=COLOR_TEXT), colorway=PLOTLY_COLORWAY, margin=dict(t=56, l=10, r=10, b=10))
    fig.update_xaxes(gridcolor=COLOR_BORDER, zerolinecolor=COLOR_BORDER)
    fig.update_yaxes(gridcolor=COLOR_BORDER, zerolinecolor=COLOR_BORDER)
    return fig

def build_excel_report(user_query, df_result, summary_text, stats_text, final_sql):
    import io
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils.dataframe import dataframe_to_rows
    wb = Workbook()
    ws_data = wb.active
    ws_data.title = "ข้อมูล"
    for row in dataframe_to_rows(df_result, index=False, header=True): ws_data.append(row)
    header_fill = PatternFill(start_color="C9A227", end_color="C9A227", fill_type="solid")
    for cell in ws_data[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
    
    ws_summary = wb.create_sheet("สรุปผล")
    ws_summary.column_dimensions["A"].width = 100
    rows_to_write = [("คำถามของผู้ใช้", user_query), ("", ""), ("SQL ที่ AI สร้าง", final_sql), ("", ""), ("บทสรุปการวิเคราะห์ (Executive Summary)", ""), (summary_text, ""), ("", ""), ("สถิติที่คำนวณจริง (Statistical Insights)", ""), (stats_text, "")]
    for label, value in rows_to_write:
        ws_summary.append([f"{label}: {value}"] if value else [label])
        ws_summary.cell(row=ws_summary.max_row, column=1).alignment = Alignment(wrap_text=True, vertical="top")
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

@st.cache_resource
def load_agent(api_key: str = None):
    return EnterpriseDataAgent(api_key=api_key) if api_key else EnterpriseDataAgent()

# ----------------------------------------------------
# UI: Sidebar ควบคุม Data Pipeline
# ----------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Data Engineering Ops")
    if HAS_PIPELINE:
        st.markdown("ระบบ Ingest ➡️ Transform ➡️ Quality")
        if st.button("🚀 รัน ETL Pipeline", use_container_width=True):
            with st.spinner("กำลังดึงและคลีนข้อมูล..."):
                try:
                    pipeline_summary = run_pipeline()
                    st.cache_resource.clear()
                    st.success("✅ อัปเดตข้อมูลและทำ Quality Check เรียบร้อย!")
                    with st.expander("📊 ดูรายงาน Pipeline Run Log"):
                        st.json(pipeline_summary)
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดตอนรัน Pipeline: {e}")
    else:
        st.warning("⚠️ โหลด Data Pipeline ไม่สำเร็จ ฟีเจอร์นี้จึงถูกปิดใช้งาน")
        with st.expander("คลิกดูรายละเอียด Error", expanded=False):
            st.code(PIPELINE_ERROR, language="python")

manual_api_key = None
try: _has_secret_key = "GEMINI_API_KEY" in st.secrets
except Exception: _has_secret_key = False
if not _has_secret_key and not os.environ.get("GEMINI_API_KEY"):
    with st.expander("🔑 ยังไม่ได้ตั้งค่า GEMINI_API_KEY — กรอกที่นี่ชั่วคราว"):
        manual_api_key = st.text_input("Gemini API Key", type="password", key="manual_api_key_input")

with st.spinner("กำลังเชื่อมต่อฐานข้อมูล..."):
    agent = load_agent(api_key=manual_api_key)

st.markdown(
    """
    <div class="exec-header">
        <div class="eyebrow">Executive Data Dashboard</div>
        <h1>Enterprise Data Agent</h1>
        <p>ถามคำถามเป็นภาษาธรรมชาติ ระบบแปลงเป็น SQL, สรุปผล และแสดงกราฟให้อัตโนมัติ</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("**⚙ เลือก AI Engine**")
    engine_choice = st.radio("AI Engine", options=["☁️ Cloud (Gemini API)", "💻 Local (Ollama)"], horizontal=True, label_visibility="collapsed")
    engine = "cloud" if engine_choice.startswith("☁️") else "local"

    local_model = None
    ollama_url = None
    if engine == "local":
        model_keys = list(LOCAL_MODEL_INFO.keys())
        local_model = st.selectbox("เลือกโมเดล Local", options=model_keys, index=0, format_func=lambda m: LOCAL_MODEL_INFO[m]["display_name"] + (" ⭐ แนะนำ" if LOCAL_MODEL_INFO[m]["recommended"] else ""))
        info = LOCAL_MODEL_INFO[local_model]
        st.caption(f"📊 Accuracy (benchmark 150 ข้อ): {info['accuracy_label']}")
        st.caption(f"⚠️ Empty Response: {info['empty_response_label']}")
        st.caption(f"ℹ️ {info['note']}")
        ollama_url = st.text_input("Ollama Server URL", value=agent.ollama_base_url)
        st.markdown('<span class="privacy-badge local">🔒 Data Privacy: ส่งข้อมูลไปยัง Ollama server ที่ระบุเท่านั้น</span>', unsafe_allow_html=True)
    else:
        st.caption("ใช้ Gemini API ผ่าน Cloud")
        st.markdown('<span class="privacy-badge cloud">⚠️ Data Privacy: ส่งข้อมูลไปยัง Google Cloud</span>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💬 AI Query Engine", "🔍 Data Schema Explorer", "📝 Feedback Dashboard"])

with tab1:
    user_query = st.text_input("พิมพ์คำถามของคุณที่นี่ (เช่น: ขอ 5 อันดับสินค้าที่มียอดขายรวมสูงสุด):")
    if st.button("ประมวลผลคำสั่ง"):
        if user_query:
            with st.spinner("กำลังสร้างคำสั่ง SQL และดึงข้อมูล..."):
                df_result, final_sql, logs = agent.execute_with_self_correction(user_query, engine=engine, local_model=local_model, ollama_url=ollama_url)
            summary = ""
            if df_result is not None and not df_result.empty:
                with st.spinner("กำลังวิเคราะห์และสรุป Insight..."):
                    summary = agent.generate_executive_summary(user_query, df_result, engine=engine, local_model=local_model, ollama_url=ollama_url)
            
            st.session_state["last_result"] = {
                "user_query": user_query, "df_result": df_result, "final_sql": final_sql,
                "logs": logs, "summary": summary, "engine": engine, "local_model": local_model,
            }

    result = st.session_state.get("last_result")
    if result:
        user_query, df_result, final_sql, logs, summary, engine, local_model = result["user_query"], result["df_result"], result["final_sql"], result["logs"], result["summary"], result["engine"], result["local_model"]

        if df_result is not None and not df_result.empty:
            render_kpi_cards(df_result)
            st.subheader("💡 บทสรุปการวิเคราะห์ (Executive Summary)")
            st.write(summary)
            st.subheader("📊 ผลลัพธ์ตารางข้อมูล (Query Results)")
            st.dataframe(df_result, use_container_width=True)

            charts = generate_charts(df_result)
            if charts:
                st.subheader("📈 การวิเคราะห์เชิงภาพ (Visual Analytics)")
                chart_tabs = st.tabs([title for title, _ in charts])
                for tab, (title, fig) in zip(chart_tabs, charts):
                    with tab: st.plotly_chart(style_chart_theme(fig), use_container_width=True)

            stats_text = compute_statistical_insights(df_result)
            with st.expander("🧮 ตัวเลขสถิติที่คำนวณจริง (Statistical Insights)"):
                st.text(stats_text)

            st.markdown("**📥 Export รายงานสรุป**")
            try:
                excel_bytes = build_excel_report(user_query, df_result, summary, stats_text, final_sql)
                st.download_button("📊 ดาวน์โหลด Excel", data=excel_bytes, file_name="enterprise_data_agent_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.error(f"สร้างไฟล์ Excel ไม่สำเร็จ: {e}")
        else:
            st.warning("ไม่พบข้อมูล หรือเกิดข้อผิดพลาดในการรัน SQL")

        with st.expander("🔍 Audit Logs & Generated SQL Pipeline"):
            st.code(final_sql, language="sql")
            for log in logs: st.write(log)

        st.markdown("**SQL ที่ AI สร้างถูกต้องไหม?**")
        fb_col1, fb_col2, _ = st.columns([1, 1, 6])
        row_count_for_feedback = int(len(df_result)) if df_result is not None else 0
        with fb_col1:
            if st.button("👍 ถูกต้อง"):
                save_feedback(user_query, final_sql, engine, local_model, "up", row_count_for_feedback)
                st.toast("บันทึก Feedback 👍 แล้ว ขอบคุณครับ", icon="✅")
        with fb_col2:
            if st.button("👎 ไม่ถูกต้อง"):
                save_feedback(user_query, final_sql, engine, local_model, "down", row_count_for_feedback)
                st.toast("บันทึก Feedback 👎 แล้ว ขอบคุณครับ", icon="📝")

with tab2:
    st.subheader("📋 Schema และตัวอย่างข้อมูลของฐานข้อมูลทั้งหมด")
    search_term = st.text_input("🔍 ค้นหาชื่อตาราง หรือ ชื่อคอลัมน์:")
    tables = agent.con.execute("SHOW TABLES").fetchall()
    for t in tables:
        if not t: continue
        t_name = t[0]
        try:
            cols = agent.con.execute(f"DESCRIBE {t_name}").fetchall()
        except Exception:
            continue
        col_names = [c[0] for c in cols if len(c) >= 1]
        if search_term.lower() in t_name.lower() or any(search_term.lower() in c.lower() for c in col_names):
            with st.expander(f"📌 Table: {t_name}", expanded=False):
                st.markdown("**📌 Data Types & Schema:**")
                st.dataframe([{"Column Name": c[0], "Data Type": c[1]} for c in cols if len(c) >= 2], use_container_width=True)
                st.markdown("**👀 Sample Data (Top 3 rows):**")
                try:
                    sample_df = agent.con.execute(f"SELECT * FROM {t_name} LIMIT 3").df()
                    st.dataframe(sample_df, use_container_width=True)
                except Exception:
                    pass

with tab3:
    st.subheader("📝 สรุปผล Feedback จากผู้ใช้งานจริง")
    feedback_df = get_feedback_stats()
    if feedback_df.empty:
        st.info("ยังไม่มี Feedback เข้ามา")
    else:
        total_fb = len(feedback_df)
        up_count = int((feedback_df["rating"] == "up").sum())
        down_count = int((feedback_df["rating"] == "down").sum())
        accuracy_pct = (up_count / total_fb * 100) if total_fb else 0
        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        kpi_c1.metric("Feedback ทั้งหมด", f"{total_fb:,}")
        kpi_c2.metric("👍 ถูกต้อง", f"{up_count:,}")
        kpi_c3.metric("👎 ไม่ถูกต้อง", f"{down_count:,}")
        kpi_c4.metric("Accuracy จาก Feedback", f"{accuracy_pct:.1f}%")
        st.markdown("---")
        breakdown = (
            feedback_df.assign(engine_label=feedback_df.apply(lambda r: f"{r['engine']} ({r['local_model']})" if r["engine"] == "local" and r["local_model"] else r["engine"], axis=1))
            .groupby("engine_label")["rating"].value_counts().unstack(fill_value=0)
        )
        if "up" not in breakdown.columns: breakdown["up"] = 0
        if "down" not in breakdown.columns: breakdown["down"] = 0
        breakdown["total"] = breakdown["up"] + breakdown["down"]
        breakdown["accuracy_%"] = (breakdown["up"] / breakdown["total"] * 100).round(1)
        st.dataframe(breakdown, use_container_width=True)
        st.markdown("---")
        display_df = feedback_df.copy()
        display_df["rating"] = display_df["rating"].map({"up": "👍 ถูกต้อง", "down": "👎 ไม่ถูกต้อง"})
        st.dataframe(display_df[["timestamp", "user_query", "generated_sql", "engine", "local_model", "rating", "row_count"]], use_container_width=True)
        st.download_button("📥 ดาวน์โหลด Feedback ทั้งหมดเป็น CSV", data=feedback_df.to_csv(index=False).encode("utf-8-sig"), file_name="feedback_export.csv", mime="text/csv")
