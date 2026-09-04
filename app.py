import streamlit as st
import plotly.express as px
from agent_core import EnterpriseDataAgent

st.set_page_config(page_title="Enterprise Private AI Data Agent", layout="wide")

st.title("🔒 On-Premise Data Agent")
st.caption("ระบบวิเคราะห์ข้อมูลองค์กรระดับ Enterprise | Fully Offline & Private Network Only")

@st.cache_resource
def init_agent():
    return EnterpriseDataAgent()

agent = init_agent()

# Sidebar - Managed Local Tables
st.sidebar.header("🛡️ Secured Data Sources")
if agent.registered_tables:
    for table in agent.registered_tables:
        st.sidebar.success(f"Isolated Table: `{table}`")
else:
    st.sidebar.error("ไม่พบตารางข้อมูล กรุณาเพิ่มไฟล์ CSV/Parquet ในโฟลเดอร์ `data_input/`")

# User Query Form
user_query = st.text_input("ระบุคำถามภาษาไทยที่ต้องการวิเคราะห์ข้อมูล:", placeholder="เช่น แสดงหมวดหมู่สินค้าที่มียอดขายสูงสุด 5 อันดับแรก")

if st.button("ประมวลผลข้อมูล") and user_query:
    with st.spinner("Agentกำลังอ่าน Schema และสร้าง SQL..."):
        result_df, final_sql, logs = agent.execute_with_self_correction(user_query)

    # 1. Executive Summary
    st.subheader("💡 บทสรุปการวิเคราะห์ (Executive Summary)")
    if result_df is not None and not result_df.empty:
        summary = agent.generate_executive_summary(user_query, result_df)
        st.info(summary)
    else:
        st.warning("ระบบไม่สามารถดึงข้อมูลได้หรือไม่มีข้อมูลตรงเงื่อนไข")

    # 2. Data Table
    st.subheader("📋 ตารางข้อมูลผลลัพธ์ (Result Set)")
    if result_df is not None:
        st.dataframe(result_df, use_container_width=True)

        # 3. Dynamic Auto-Visualization
        if len(result_df.columns) >= 2:
            st.subheader("📊 การแสดงผลเชิงภาพ (Auto-Visualization)")
            cols = result_df.columns
            try:
                fig = px.bar(result_df, x=cols[0], y=cols[1], title=f"แผนภูมิแสดง {cols[1]} จำแนกตาม {cols[0]}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.caption("โครงสร้างข้อมูลชุดนี้ไม่เหมาะสำหรับการสร้างแผนภูมิแท่งอัตโนมัติ")

    # 4. System Developer Audit Logs
    with st.expander("🔍 Audit Logs & Generated SQL Pipeline (สำหรับงานเทคนิค)"):
        st.markdown("**Executable SQL Code:**")
        st.code(final_sql, language="sql")
        st.markdown("**Agent Self-Correction & Execution History:**")
        for log in logs:
            st.text(log)
