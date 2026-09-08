import json
import duckdb
import pandas as pd
from agent_core import EnterpriseDataAgent

def run_evaluation(benchmark_json_path):
    agent = EnterpriseDataAgent()
    
    with open(benchmark_json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    correct_count = 0
    total_count = len(dataset)
    
    print(f"--- Starting Evaluation on {total_count} Queries ---")
    
    for item in dataset:
        q_id = item['id']
        question = item['question_th']
        ground_truth_sql = item['sql']
        
        # รัน Agent เพื่อเจน SQL
        predicted_df, pred_sql, _ = agent.execute_with_self_correction(question)
        
        # Execute Ground Truth SQL เพื่อเอา Result มาเทียบ
        try:
            target_df = agent.con.execute(ground_truth_sql).df()
            
            # เช็กว่าผลลัพธ์ของ DataFrame ตรงกันหรือไม่ (Execution Accuracy)
            if predicted_df is not None and predicted_df.equals(target_df):
                correct_count += 1
                print(f"ID {q_id}: PASSED")
            else:
                print(f"ID {q_id}: FAILED (Mismatched Output)")
        except Exception as e:
            print(f"ID {q_id}: FAILED (Ground Truth Exec Error)")
            
    accuracy = (correct_count / total_count) * 100
    print(f"\nFinal Execution Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    run_evaluation("../dataset/olist_qa_dataset_150.json")
