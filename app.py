import os
import duckdb
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google import genai


def _find_datetime_col(df: pd.DataFrame):
    """หาคอลัมน์ที่เป็นวันที่/เวลา (รวมถึงลองแปลงจาก object ที่หน้าตาเหมือนวันที่)"""
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
    """
    วิเคราะห์รูปร่างของผลลัพธ์ query แล้วสร้างกราฟ Plotly ที่เหมาะสมโดยอัตโนมัติ
    คืนค่าเป็น list ของ (หัวข้อกราฟ, figure) — อาจมีมากกว่า 1 กราฟต่อผลลัพธ์
    """
    charts = []
    if df is None or df.empty or df.shape[1] < 2:
        return charts

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return charts  # ไม่มีตัวเลขให้พล็อตเลย

    y_col = numeric_cols[0]
    date_col = _find_datetime_col(df)
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols and c != date_col]

    # 1) มีคอลัมน์วันที่ -> Line chart แนวโน้ม
    if date_col:
        df_sorted = df.sort_values(by=date_col)
        fig = px.line(df_sorted, x=date_col, y=numeric_cols, markers=True,
                       title=f"แนวโน้ม {', '.join(numeric_cols)} ตาม {date_col}")
        charts.append(("แนวโน้มตามช่วงเวลา (Trend)", fig))

    # 2) มีคอลัมน์หมวดหมู่ + ตัวเลข -> Bar / Pie / Pareto
    if non_numeric_cols:
        cat_col = non_numeric_cols[0]
        grouped = df.groupby(cat_col, as_index=False)[y_col].sum().sort_values(y_col, ascending=False)

        # Bar chart (top 15 อันดับ)
        bar_df = grouped.head(15)
        fig_bar = px.bar(bar_df, x=cat_col, y=y_col, title=f"{y_col} แยกตาม {cat_col} (Top {len(bar_df)})")
        fig_bar.update_layout(xaxis_tickangle=-30)
        charts.append(("เปรียบเทียบตามหมวดหมู่ (Bar)", fig_bar))

        # Pie chart — เฉพาะกรณีหมวดหมู่ไม่เยอะเกินไป (สัดส่วนอ่านง่าย)
        if grouped[cat_col].nunique() <= 8:
            fig_pie = px.pie(grouped, names=cat_col, values=y_col,
                              title=f"สัดส่วน {y_col} ตาม {cat_col}", hole=0.35)
            charts.append(("สัดส่วนโดยรวม (Pie)", fig_pie))

        # Pareto chart (80/20) — เฉพาะกรณีมีหลายหมวดหมู่พอจะวิเคราะห์
        if grouped[cat_col].nunique() >= 3:
            pareto_df = grouped.head(15).reset_index(drop=True)
            total = pareto_df[y_col].sum()
            if total:
                pareto_df["cum_pct"] = pareto_df[y_col].cumsum() / total * 100
                fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
                fig_pareto.add_trace(
                    go.Bar(x=pareto_df[cat_col].astype(str), y=pareto_df[y_col], name=y_col),
                    secondary_y=False,
                )
                fig_pareto.add_trace(
                    go.Scatter(x=pareto_df[cat_col].astype(str), y=pareto_df["cum_pct"],
                               name="สัดส่วนสะสม (%)", mode="lines+markers"),
                    secondary_y=True,
                )
                fig_pareto.add_hline(y=80, line_dash="dot", secondary_y=True,
                                      annotation_text="เส้น 80%")
                fig_pareto.update_layout(title=f"Pareto Analysis: {y_col} ตาม {cat_col} (กฎ 80/20)")
                fig_pareto.update_yaxes(title_text=y_col, secondary_y=False)
                fig_pareto.update_yaxes(title_text="สัดส่วนสะสม (%)", range=[0, 110], secondary_y=True)
                charts.append(("Pareto Analysis (80/20)", fig_pareto))

    # 3) ไม่มีคอลัมน์หมวดหมู่/วันที่ แต่มีตัวเลขตั้งแต่ 2 คอลัมน์ -> Scatter
    if not date_col and not non_numeric_cols and len(numeric_cols) >= 2:
        fig_scatter = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                                  title=f"{numeric_cols[1]} เทียบกับ {numeric_cols[0]}")
        charts.append(("ความสัมพันธ์ระหว่างตัวแปร (Scatter)", fig_scatter))

    return charts


def compute_statistical_insights(df: pd.DataFrame) -> str:
    """
    คำนวณสถิติเชิงวิเคราะห์เบื้องต้นจริงจาก DataFrame (ไม่ใช่ให้ LLM เดาตัวเลขเอง):
    - สถิติพื้นฐาน (mean/median/std/min/max)
    - Outlier detection ด้วยกฎ IQR
    - Pareto (80/20): ต้องใช้กี่หมวดหมู่ถึงจะครอบคลุม 80% ของยอดรวม
    - % การเติบโตเทียบครึ่งแรก vs ครึ่งหลังของข้อมูล (ถ้ามีคอลัมน์วันที่)
    ผลลัพธ์เป็นข้อความสรุป ใช้เป็น "ground truth" ป้อนให้ Gemini เขียนบรรยายต่อ
    """
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

    lines.append(
        f"สถิติเบื้องต้นของคอลัมน์ '{main_col}': ค่าเฉลี่ย={series.mean():,.2f}, "
        f"มัธยฐาน={series.median():,.2f}, ส่วนเบี่ยงเบนมาตรฐาน={series.std():,.2f}, "
        f"ต่ำสุด={series.min():,.2f}, สูงสุด={series.max():,.2f} (n={len(series)})"
    )

    # Outlier detection ด้วย IQR
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    lines.append(
        f"Outlier Detection (กฎ IQR) บนคอลัมน์ '{main_col}': พบ {len(outliers)} รายการ "
        f"จากทั้งหมด {len(series)} แถว (ช่วงปกติโดยประมาณ {lower:,.2f} ถึง {upper:,.2f})"
    )

    # Pareto 80/20 (ต้องมีคอลัมน์หมวดหมู่)
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
            lines.append(
                f"Pareto Analysis (80/20) บน '{cat_col}': มีเพียง {n_items_80} รายการ "
                f"(คิดเป็น {pct_of_items:.1f}% ของ {cat_col} ทั้งหมด {len(grouped)} รายการ) "
                f"ที่รวมกันสร้าง '{main_col}' ได้ถึงประมาณ 80% ของยอดรวมทั้งหมด"
            )

    # % การเติบโต ครึ่งแรก vs ครึ่งหลัง (ถ้ามีคอลัมน์วันที่)
    if date_col_probe:
        df_sorted = df.sort_values(by=date_col_probe)
        mid = len(df_sorted) // 2
        if mid > 0:
            first_half = df_sorted.iloc[:mid][main_col].sum()
            second_half = df_sorted.iloc[mid:][main_col].sum()
            if first_half != 0:
                growth = (second_half - first_half) / abs(first_half) * 100
                lines.append(
                    f"อัตราการเติบโตของ '{main_col}' เมื่อเทียบครึ่งแรกกับครึ่งหลังของช่วงข้อมูล "
                    f"(เรียงตาม '{date_col_probe}'): เปลี่ยนแปลง {growth:+.1f}%"
                )

    return "\n".join(lines)

# ==========================================
# 1. Class: EnterpriseDataAgent (Data Engine)
# ==========================================
class EnterpriseDataAgent:
    def __init__(self):
        # 1. ดึง GEMINI_API_KEY จาก Streamlit Secrets หรือ Environment Variable
        api_key = None
        try:
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
            else:
                api_key = os.environ.get("GEMINI_API_KEY")
        except Exception:
            api_key = os.environ.get("GEMINI_API_KEY")

        # 2. เริ่มต้นสร้าง Gemini Client
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None
            print("Warning: GEMINI_API_KEY not found.")

        # ตั้งค่าโมเดลหลักและโมเดลสำรองให้เป็นรุ่นที่รองรับบน API ปัจจุบัน
        self.primary_model = "gemini-3.6-flash"
        self.fallback_model = "gemini-3.5-flash"

        # 3. เชื่อมต่อ DuckDB ใน Memory
        self.con = duckdb.connect(database=':memory:')

        # 4. โหลดข้อมูลแบบ VIEW (Lazy Load ช่วยให้เปิดแอปได้เร็ว)
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
        """เรียกใช้งาน API หากโมเดลหลักติด Error ให้สลับไปใช้โมเดลสำรองอัตโนมัติ"""
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

    def execute_with_self_correction(self, user_query, max_attempts=3):
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
                raw_response = self._call_gemini_with_fallback(prompt)
                sql_query = raw_response.strip().replace("```sql", "").replace("```", "").strip()
                logs.append(f"[Attempt {attempt}] Generated SQL: {sql_query}")

                df_result = self.con.execute(sql_query).df()
                logs.append(f"[Attempt {attempt}] Success.")
                return df_result, sql_query, logs

            except Exception as e:
                last_error = str(e)
                logs.append(f"[Attempt {attempt}] Execution Error: {last_error}")

                # ป้อน error กลับไปให้ Gemini แก้ SQL ในรอบถัดไป (Self-Correction)
                prompt = f"""
                {base_prompt}

                คำสั่ง SQL ที่คุณเขียนก่อนหน้านี้:
                {sql_query}

                รันแล้วเจอ error นี้:
                {last_error}

                กรุณาแก้ไขคำสั่ง SQL ให้ถูกต้องตาม schema ที่ให้ไว้ข้างต้น
                (เช่น ถ้า error บอกว่าไม่มีตารางนี้ ให้ตรวจดู schema แล้วใช้ชื่อตารางที่ถูกต้องจริง ๆ)
                Return ONLY the raw SQL query without codeblock formatting or explanations.
                """

        logs.append(f"ล้มเหลวหลังจากพยายาม {max_attempts} ครั้ง: {last_error}")
        return None, sql_query, logs


    def generate_executive_summary(self, user_query, df_result):
        if df_result is None or df_result.empty:
            return "ไม่พบข้อมูลสำหรับสรุปผลลัพธ์"

        if not self.client:
            return "ไม่สามารถสรุปผลลัพธ์ได้เนื่องจากขาด GEMINI_API_KEY"

        data_preview = df_result.head(20).to_string(index=False)
        stats_block = compute_statistical_insights(df_result)

        prompt = f"""
        คุณเป็น Data Analyst / Data Scientist ผู้เชี่ยวชาญ กรุณาสรุปผลลัพธ์จากข้อมูลด้านล่างนี้ เพื่อตอบคำถามของผู้ใช้:

        คำถามของผู้ใช้: "{user_query}"

        ผลลัพธ์ข้อมูลที่ได้จาก Database (ตัวอย่าง 20 แถวแรก):
        {data_preview}

        ผลการวิเคราะห์เชิงสถิติที่คำนวณไว้ล่วงหน้าแล้ว (เป็นตัวเลขจริงที่คำนวณจากข้อมูลทั้งหมด ไม่ใช่การประมาณ
        ให้ใช้ตัวเลขชุดนี้อ้างอิงในการตอบ ห้ามคำนวณตัวเลขสถิติขึ้นใหม่เอง):
        {stats_block}

        คำแนะนำในการตอบ:
        1. อธิบายคำตอบหลักให้ชัดเจน ตรงประเด็นกับคำถามของผู้ใช้ก่อน
        2. สรุปจุดสำคัญหรือ Insight ที่น่าสนใจจากข้อมูล เป็นข้อๆ (Bullet points)
        3. อ้างอิงผลการวิเคราะห์เชิงสถิติที่ให้ไว้ข้างต้น (Outlier, Pareto 80/20, % การเติบโต ถ้ามี)
           แล้วตีความเป็นภาษาที่ผู้บริหารเข้าใจง่าย พร้อมข้อเสนอแนะเชิงธุรกิจถ้าเป็นไปได้
        4. ตอบเป็นภาษาไทยที่สุภาพ เข้าใจง่าย และเป็นทางการ
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

# ------------------------------------------
# Design tokens — "boardroom" executive theme
# ------------------------------------------
COLOR_BG = "#0F1620"          # deep navy-charcoal base
COLOR_SURFACE = "#161F2E"     # card / panel surface
COLOR_BORDER = "#28344A"      # hairline borders
COLOR_TEXT = "#E8ECF3"        # primary text
COLOR_TEXT_MUTED = "#8B9BB4"  # secondary text
COLOR_ACCENT = "#C9A227"      # muted brass/gold — the one bold accent
COLOR_ACCENT_SOFT = "#3FA796" # muted teal — secondary/positive signal
PLOTLY_COLORWAY = [COLOR_ACCENT, COLOR_ACCENT_SOFT, "#7D8FB3", "#C1584C", "#5B7FA6"]

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: {COLOR_BG} !important;
    color: {COLOR_TEXT};
    font-family: 'IBM Plex Sans', sans-serif;
}}
[data-testid="stHeader"] {{ background-color: transparent !important; }}
[data-testid="stAppViewContainer"] .main .block-container {{
    padding-top: 2.2rem;
    max-width: 1180px;
}}

h1, h2, h3 {{
    font-family: 'Source Serif 4', serif !important;
    color: {COLOR_TEXT} !important;
    font-weight: 600 !important;
}}

.exec-header {{ margin-bottom: 1.6rem; }}
.exec-header .eyebrow {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.82rem;
    color: {COLOR_ACCENT_SOFT};
    margin-bottom: 2px;
}}
.exec-header h1 {{
    font-size: 2.1rem !important;
    margin: 0 0 4px 0 !important;
}}
.exec-header p {{
    color: {COLOR_TEXT_MUTED};
    font-size: 0.95rem;
    margin: 0;
}}

/* KPI card grid */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 14px;
    margin: 4px 0 26px 0;
}}
.kpi-card {{
    background: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-left: 3px solid {COLOR_ACCENT};
    border-radius: 6px;
    padding: 16px 20px;
}}
.kpi-card .kpi-label {{
    font-size: 0.76rem;
    color: {COLOR_TEXT_MUTED};
    margin-bottom: 6px;
}}
.kpi-card .kpi-value {{
    font-family: 'Source Serif 4', serif;
    font-size: 1.75rem;
    font-variant-numeric: tabular-nums;
    line-height: 1.15;
    color: {COLOR_TEXT};
}}
.kpi-card .kpi-sub {{
    font-size: 0.74rem;
    color: {COLOR_ACCENT_SOFT};
    margin-top: 6px;
}}

/* Inputs, buttons, tabs, expanders, dataframe */
[data-testid="stTextInput"] input {{
    background-color: {COLOR_SURFACE} !important;
    color: {COLOR_TEXT} !important;
    border: 1px solid {COLOR_BORDER} !important;
    border-radius: 6px !important;
}}
[data-testid="stButton"] button {{
    background-color: {COLOR_ACCENT} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.4rem !important;
}}
[data-testid="stButton"] button p {{
    color: #FFFFFF !important;
}}
[data-testid="stButton"] button:hover {{
    background-color: #DDB542 !important;
    color: #FFFFFF !important;
}}
[data-testid="stTabs"] button [data-testid="stMarkdownContainer"] p {{
    font-family: 'IBM Plex Sans', sans-serif;
    color: {COLOR_TEXT_MUTED};
}}
[data-testid="stTabs"] [aria-selected="true"] [data-testid="stMarkdownContainer"] p {{
    color: {COLOR_ACCENT} !important;
}}
[data-testid="stExpander"] {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER} !important;
    border-radius: 6px !important;
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
}}
[data-testid="stAlert"] {{
    background-color: {COLOR_SURFACE} !important;
    border: 1px solid {COLOR_BORDER} !important;
    color: {COLOR_TEXT} !important;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def format_kpi_number(x):
    """จัดรูปแบบตัวเลขให้อ่านง่ายแบบสรุปผู้บริหาร (K/M)"""
    if x is None:
        return "-"
    abs_x = abs(x)
    if abs_x >= 1_000_000:
        return f"{x/1_000_000:,.2f}M"
    if abs_x >= 1_000:
        return f"{x:,.0f}"
    return f"{x:,.2f}"


def compute_kpis(df: pd.DataFrame):
    """
    สร้างชุด KPI จากผลลัพธ์ query:
    - จำนวนแถว
    - ผลรวม/ค่าเฉลี่ยของคอลัมน์ตัวเลข (สูงสุด 2 คอลัมน์แรก)
    - จำนวนนับไม่ซ้ำ (distinct) ของคอลัมน์ที่ดูเหมือน ID เช่น customer_id, order_id ถ้ามี
    """
    kpis = [("จำนวนแถวผลลัพธ์", f"{len(df):,}", "")]

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols[:2]:
        total = df[col].sum()
        avg = df[col].mean()
        kpis.append((f"รวม {col}", format_kpi_number(total), f"เฉลี่ย {format_kpi_number(avg)} / แถว"))

    id_like_keywords = ["customer", "order", "user", "ลูกค้า"]
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in id_like_keywords) and len(kpis) < 4:
            distinct_count = df[col].nunique()
            kpis.append((f"จำนวน {col} ไม่ซ้ำ", f"{distinct_count:,}", ""))

    return kpis[:4]


def render_kpi_cards(df: pd.DataFrame):
    kpis = compute_kpis(df)
    cards_html = "".join(
        f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>"""
        for label, value, sub in kpis
    )
    st.markdown(f'<div class="kpi-grid">{cards_html}</div>', unsafe_allow_html=True)


def style_chart_theme(fig):
    """ปรับโทนกราฟ Plotly ให้เข้ากับธีม Dashboard"""
    fig.update_layout(
        paper_bgcolor=COLOR_SURFACE,
        plot_bgcolor=COLOR_SURFACE,
        font=dict(family="IBM Plex Sans, sans-serif", color=COLOR_TEXT),
        title_font=dict(family="Source Serif 4, serif", size=18, color=COLOR_TEXT),
        colorway=PLOTLY_COLORWAY,
        margin=dict(t=56, l=10, r=10, b=10),
    )
    fig.update_xaxes(gridcolor=COLOR_BORDER, zerolinecolor=COLOR_BORDER)
    fig.update_yaxes(gridcolor=COLOR_BORDER, zerolinecolor=COLOR_BORDER)
    return fig


@st.cache_resource
def load_agent():
    return EnterpriseDataAgent()

with st.spinner("กำลังเชื่อมต่อฐานข้อมูล และสร้าง Data Engine Views..."):
    agent = load_agent()

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

tab1, tab2 = st.tabs(["💬 AI Query Engine", "🔍 Data Schema Explorer"])

with tab1:
    user_query = st.text_input("พิมพ์คำถามของคุณที่นี่ (เช่น: ขอ 5 อันดับสินค้าที่มียอดขายรวมสูงสุด):")

    if st.button("ประมวลผลคำสั่ง"):
        if user_query:
            with st.spinner("กำลังสร้างคำสั่ง SQL และดึงข้อมูล..."):
                df_result, final_sql, logs = agent.execute_with_self_correction(user_query)

            if df_result is not None and not df_result.empty:
                render_kpi_cards(df_result)

                st.subheader("💡 บทสรุปการวิเคราะห์ (Executive Summary)")
                with st.spinner("กำลังวิเคราะห์และสรุป Insight..."):
                    summary = agent.generate_executive_summary(user_query, df_result)
                st.write(summary)

                st.subheader("📊 ผลลัพธ์ตารางข้อมูล (Query Results)")
                st.dataframe(df_result, use_container_width=True)

                charts = generate_charts(df_result)
                if charts:
                    st.subheader("📈 การวิเคราะห์เชิงภาพ (Visual Analytics)")
                    chart_tabs = st.tabs([title for title, _ in charts])
                    for tab, (title, fig) in zip(chart_tabs, charts):
                        with tab:
                            st.plotly_chart(style_chart_theme(fig), use_container_width=True)

                with st.expander("🧮 ตัวเลขสถิติที่คำนวณจริง (Statistical Insights — raw numbers)"):
                    st.text(compute_statistical_insights(df_result))
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
