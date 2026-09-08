import json

def convert_to_instruction_format(input_json, output_jsonl):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    formatted_data = []
    for item in data:
        entry = {
            "instruction": "You are an expert Data Engineer. Write a DuckDB SQL query to answer the following question.",
            "input": f"Question: {item['question_th']}",
            "output": f"```sql\n{item['sql']}\n```"
        }
        formatted_data.append(entry)
        
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in formatted_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"Converted {len(formatted_data)} items to {output_jsonl}")

if __name__ == "__main__":
    convert_to_instruction_format("../dataset/olist_qa_dataset_150.json", "train_data.jsonl")
