import io
import pandas as pd
import streamlit as st
from agent_core import EnterpriseDataAgent

st.set_page_config(page_title="Enterprise Data Agent", layout="wide")

# โหลด Agent และแคชไว้ในเซิร์ฟเวอร์เพื่อความรวดเร็ว

@st.cache_resource
def get_agent():
    return EnterpriseDataAgent()

agent = get_agent()

# Sidebar
with st.sidebar:
    st.header("🛡️ Secured Data Sources")
    tables = agent.con.execute("SHOW TABLES").fetchall()
    if tables:
        for t in tables:
            st.success(f"Isolated Table: {t[0]}")
    else:
        st.error("ไม่พบตารางข้อมูล")

st.title("🔒 On-Premise Data Agent")
st.caption("ระบบวิเคราะห์ข้อมูลองค์กรระดับ Enterprise | Fully Offline & Private Network Only")

tab1, tab2 = st.tabs(["💬 AI Query Engine", "📊 Data Schema & Dictionary"])

# -------------------------------------------------------------
# TAB 1: AI Query Engine & Export Features
# -------------------------------------------------------------
with tab1:
    st.subheader("ระบุคำถามภาษาไทยที่ต้องการวิเคราะห์ข้อมูล:")
    user_query = st.text_input("คำถาม:", placeholder="เช่น ขอ 5 ประเทศที่มีลูกค้ามากที่สุด พร้อม CAC เฉลี่ย", label_visibility="collapsed")

    if st.button("ประมวลผลข้อมูล", type="primary"):
        if user_query:
            with st.spinner("กำลังเขียน SQL และประมวลผลผ่าน DuckDB..."):
                df_result, final_sql, logs = agent.execute_with_self_correction(user_query)

                st.markdown("### 💡 บทสรุปการวิเคราะห์ (Executive Summary)")
                summary = agent.generate_executive_summary(user_query, df_result)
                st.info(summary)

                if df_result is not None and not df_result.empty:
                    st.markdown("### 📋 ตารางข้อมูลผลลัพธ์ (Result Set)")
                    st.dataframe(df_result, use_container_width=True)

                    col_dl1, col_dl2 = st.columns(2)
                    
                    csv_data = df_result.to_csv(index=False).encode('utf-8-sig')
                    col_dl1.download_button(
                        label="📥 ดาวน์โหลดผลลัพธ์ (CSV File)",
                        data=csv_data,
                        file_name="analyzed_data_export.csv",
                        mime="text/csv"
                    )

                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df_result.to_excel(writer, index=False, sheet_name='Analyzed_Data')
                    
                    col_dl2.download_button(
                        label="📊 ดาวน์โหลดผลลัพธ์ (Excel File)",
                        data=buffer.getvalue(),
                        file_name="analyzed_data_export.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    numeric_cols = df_result.select_dtypes(include=['number']).columns.tolist()
                    if len(numeric_cols) > 0 and len(df_result) > 1:
                        st.markdown("### 📈 กราฟแสดงผลอัตโนมัติ")
                        st.bar_chart(df_result.set_index(df_result.columns[0])[numeric_cols[0]])

                with st.expander("🔍 Audit Logs & Generated SQL Pipeline (สำหรับงานเทคนิค)"):
                    st.code(final_sql, language="sql")
                    st.text("\n".join(logs))

# -------------------------------------------------------------
# TAB 2: Data Schema Explorer
# -------------------------------------------------------------
with tab2:
    st.subheader("📊 โครงสร้างตารางข้อมูลและรายละเอียดคอลัมน์ (Data Dictionary)")
    tables = agent.con.execute("SHOW TABLES").fetchall()
    if not tables:
        st.warning("ยังไม่มีตารางในระบบ กรุณาตรวจสอบโฟลเดอร์ data_input/")
    else:
        for t in tables:
            table_name = t[0]
            st.markdown(f"#### 📁 ตาราง: `{table_name}`")
            columns_df = agent.con.execute(f"DESCRIBE {table_name}").df()
            columns_df = columns_df[['column_name', 'column_type']]
            columns_df.columns = ['ชื่อคอลัมน์ (Column)', 'ชนิดข้อมูล (Data Type)']
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**รายการคอลัมน์ในตาราง:**")
                st.dataframe(columns_df, use_container_width=True)
            with col2:
                st.markdown("**ตัวอย่างข้อมูล 5 บรรทัดแรก (Data Preview):**")
                preview_df = agent.con.execute(f"SELECT * FROM {table_name} LIMIT 5").df()
                st.dataframe(preview_df, use_container_width=True)
            st.divider()
