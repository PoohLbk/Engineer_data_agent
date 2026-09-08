import os
import duckdb
import pandas as pd
from agent_core import EnterpriseDataAgent

class AutonomousRDAgent(EnterpriseDataAgent):
    def generate_hypotheses(self):
        """1. อ่าน Schema และสร้างรายการสมมติฐานทางธุรกิจอัตโนมัติ"""
        schema_info = self.get_database_schema()
        prompt = f"""คุณเป็น Lead Data Scientist จงสร้าง 3 สมมติฐานทางธุรกิจที่น่าสนใจจาก Schema ต่อไปนี้ เพื่อนำไปวิเคราะห์ต่อ:
{schema_info}

ตอบกลับเป็นรายการข้อ 1, 2, 3 สั้นๆ"""
        
        response = self.ollama_client.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']

    def run_research_cycle(self):
        """2. รันวงจร R&D: สร้างสมมติฐาน -> เขียน SQL ทดลอง -> สรุป Insight"""
        print("🔍 [R&D Agent]กำลังวิเคราะห์ตารางข้อมูลเพื่อตั้งสมมติฐาน...")
        hypotheses = self.generate_hypotheses()
        print(f"\n💡 สมมติฐานที่ตั้งขึ้น:\n{hypotheses}\n")

        research_report = []
        
        # รันการวิเคราะห์ตามสมมติฐาน
        prompt = f"จากสมมติฐานเหล่านี้: {hypotheses}\nจงเปลี่ยนให้เป็นคำถามวิเคราะห์ข้อมูล 1 ข้อที่สามารถรัน SQL หาตอบได้"
        question_response = self.ollama_client.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        auto_question = question_response['message']['content'].strip()

        print(f"🧪 [R&D Agent] กำลังทดลองหาคำตอบสำหรับ: {auto_question}")
        result_df, final_sql, logs = self.execute_with_self_correction(auto_question)
        
        if result_df is not None and not result_df.empty:
            summary = self.generate_executive_summary(auto_question, result_df)
            research_report.append({
                "hypothesis": hypotheses,
                "question": auto_question,
                "sql": final_sql,
                "summary": summary,
                "data": result_df
            })
            
        return research_report

if __name__ == "__main__":
    agent = AutonomousRDAgent()
    report = agent.run_research_cycle()
    print("\n✅ [R&D Agent] สำเร็จ! ได้รายงานผลการวิจัยข้อมูลเรียบร้อยแล้ว")
