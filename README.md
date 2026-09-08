# 🔒 Local Enterprise Data Engineering & Analytics Agent

An end-to-end, privacy-focused Autonomous AI Agent pipeline designed to automate Data Engineering and Analytics tasks locally. Built for high data privacy and security requirements, this project transforms natural Thai language queries into optimized DuckDB SQL code, executes execution loops, handles automated self-correction, and visualizes analytical results without sending data to public clouds.

---

## 🌟 Key Features

* **100% Private & Offline Execution**: Operates completely local via Ollama and in-memory DuckDB engine. No data leaks or cloud API dependencies.
* **Automated Data Ingestion**: Automatically reads and inspects CSV/Parquet file schema upon dropping into the watched folder.
* **Agentic Self-Correction Loop**: Validates SQL query execution. If a syntax or logic error occurs, the feedback loop instructs the model to self-correct up to $N$ retries.
* **Business-Level Thai Summarization**: Converts query output dataframes into concise executive summaries in Thai.
* **Auto-Visualization**: Generates interactive charts (Plotly) automatically based on query results.

---

## 🏗️ System Architecture

```text
  [ User Interface (Streamlit) ]
                │
                ▼
  [ Local File Watcher & Schema Extractor ] ──► DuckDB (In-Memory)
                │
                ▼
  [ Agentic Core Loop ] ◄── Retry Loop ──┐
                │                         │
                ▼                         │
  [ Local LLM Engine (Ollama/Qwen2.5) ] ──┤ Error Logs (Self-Correct)
                │                         │
                ▼ (Valid SQL Execution)   │
  [ Result DataFrame / Auto-Viz Chart ] ──┘
