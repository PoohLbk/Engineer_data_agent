"""แท็บ Pipeline Health สำหรับ Streamlit  ใช้: from pipeline.health_tab import render_pipeline_health"""
import pandas as pd
import streamlit as st

from . import db


def render_pipeline_health(db_path=None):
    st.subheader("🩺 Pipeline Health")
    conn = db.connect(db_path, read_only=True)
    try:
        runs = pd.read_sql_query(
            "SELECT * FROM pipeline_runs ORDER BY finished_at DESC LIMIT 30", conn)
        if runs.empty:
            st.info("ยังไม่มีประวัติการรัน pipeline")
            return
        last_run = runs.iloc[0]["run_id"]
        this_run = runs[runs["run_id"] == last_run]
        c1, c2, c3 = st.columns(3)
        c1.metric("รันล่าสุด", this_run["finished_at"].max())
        c2.metric("สถานะ", "❌ ล้มเหลว" if (this_run["status"] == "failed").any() else "✅ สำเร็จ")
        c3.metric("แถวใน Gold", pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM gold_fact_orders", conn)["n"][0])

        st.markdown("**ผลตรวจคุณภาพข้อมูล (รันล่าสุด)**")
        dq = pd.read_sql_query(
            "SELECT check_name, severity, passed, detail FROM dq_results WHERE run_id = ?",
            conn, params=(last_run,))
        dq["passed"] = dq["passed"].map({1: "✅", 0: "❌"})
        st.dataframe(dq, use_container_width=True, hide_index=True)

        st.markdown("**ประวัติการรัน**")
        st.dataframe(runs, use_container_width=True, hide_index=True)
    finally:
        conn.close()
