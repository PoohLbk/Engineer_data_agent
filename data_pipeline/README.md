# Data Pipeline สำหรับ AI Query Agent

```
data/raw/*.csv → Bronze (ข้อมูลดิบ) → Silver (ทำความสะอาด + rejects) → Gold (พร้อมให้ agent ถาม)
                                   ↘ Data quality checks → dq_results / pipeline_runs
```

| ไฟล์ | หน้าที่ |
|---|---|
| `pipeline/ingest.py` | โหลด CSV เข้า Bronze แบบ idempotent (เทียบ hash ไฟล์) |
| `pipeline/transform.py` | Silver: trim, cast, normalize สถานะ, dedupe, แยกแถวเสียลง `silver_rejects` / Gold: fact + summary tables |
| `pipeline/quality.py` | ตรวจคุณภาพ 6 ข้อ ล้ม pipeline เมื่อ critical ไม่ผ่าน |
| `pipeline/runner.py` | สั่งรันทั้งหมด + retry + บันทึกประวัติลง `pipeline_runs` |
| `pipeline/sql_guard.py` | ตรวจ SQL จาก LLM (SELECT เท่านั้น, เฉพาะตาราง Gold, ใส่ LIMIT) และรันแบบ read-only |
| `pipeline/llm_retry.py` | retry + fallback ระหว่าง Gemini กับ local model |
| `pipeline/health_tab.py` | แท็บ Pipeline Health สำหรับ Streamlit |

## วิธีรัน

```bash
python -m pipeline.sample_data      # สร้างข้อมูลตัวอย่าง (ข้ามได้ถ้ามีไฟล์จริงใน data/raw/)
python -m pipeline.runner           # รัน pipeline
python -m unittest discover -s tests -v
```

## เชื่อมกับแอป Streamlit เดิม

```python
from pipeline import db, sql_guard, llm_retry
from pipeline.health_tab import render_pipeline_health

# 1) สร้าง schema prompt จากฐานข้อมูลจริง
conn = db.connect(read_only=True)
schema = db.describe_gold_schema(conn)

# 2) เรียก LLM พร้อม fallback (แก้ปัญหา 503)
name, sql = llm_retry.call_with_fallback(
    [("gemini", call_gemini), ("local", call_local_model)],   # ฟังก์ชัน fn(prompt) -> str ของคุณ
    f"Schema:\n{schema}\n\nคำถาม: {question}\nตอบเป็น SQL อย่างเดียว")

# 3) ตรวจ + รันแบบ read-only  (เก็บ safe_sql ลง Audit Log ด้วย)
cols, rows, safe_sql = sql_guard.run_safe_query(sql)

# 4) เพิ่มแท็บใหม่  tab_health: render_pipeline_health()
```

## ขั้นต่อไป
- ย้าย SQL ใน `transform.py` ไปเป็น dbt models + tests
- ครอบด้วย Dagster หรือ Airflow แทน cron/GitHub Actions
- เปลี่ยน SQLite เป็น DuckDB หรือ Postgres เมื่อข้อมูลใหญ่ขึ้น
- ทำ eval set วัดความแม่นยำของ text-to-SQL
