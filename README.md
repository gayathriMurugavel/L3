# 🧠 Enterprise Knowledge Assistant

> AI-powered knowledge retrieval using **CrewAI**, **RAG**, **RAGAS**, and **MCP**
> powered by **Ollama** (llama3.1:8b + nomic-embed-text)

---

## 📋 Project Overview

### Business Problem
Organizations store information across multiple disconnected systems — policy
documents, knowledge repositories, CRM tools, and ticketing systems. Employees
waste significant time searching these sources manually.

### Solution
An AI-powered Enterprise Knowledge Assistant that:
- Indexes internal documents into a vector database
- Uses 3 required CrewAI agents to research, generate, and evaluate answers
- Integrates **Filesystem MCP** and **SQLite MCP** servers
- Evaluates output quality using **RAGAS** metrics
- Exposes a user-friendly **Streamlit** UI

---

## 🏗️ Architecture

```
Streamlit UI
     │
     ▼
CrewAI (3 Agents)
├── Research Agent      → RAG search + Filesystem MCP + SQLite MCP
├── Response Agent      → answer generation (saves via SQLite MCP)
└── Evaluation Agent    → final polish + quality score
     │
     ├── RAG Pipeline   → ChromaDB + nomic-embed-text
     ├── MCP Servers    → Filesystem + SQLite
     └── RAGAS          → Faithfulness, Relevancy, Precision, Recall
```

---

## ⚙️ Environment Setup

### 1. Python Version
```bash
# Install Python 3.11.15 (latest stable 3.11.x)
# Download: https://www.python.org/downloads/release/python-31115/

# Verify after installation:
python --version
# Expected: Python 3.11.15
```

### 2. Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Mac / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Ollama
```bash
# Download from https://ollama.com and install
ollama pull llama3.1:8b
ollama pull nomic-embed-text:latest
ollama serve      # Run in a separate terminal
```

### 5. Environment File
```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

---

## 🗂️ Project Structure

```
enterprise-knowledge-assistant/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── config/
│   ├── __init__.py
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   └── vector_store.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── crew_setup.py
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── mcp_tools.py
│   │   └── sqlite_mcp_server.py
│   └── evaluation/
│       ├── __init__.py
│       └── ragas_evaluator.py
├── data/
│   └── documents/
└── chroma_db/
```

---

## 🚀 Running the Application

```bash
# Terminal 1 – keep Ollama running
ollama serve

# Terminal 2 – run Streamlit with the project environment
./start_app.ps1
```

Open browser at: **http://localhost:8501**

---

## 📋 Step-by-Step Usage

### Step 1 – Upload Documents
1. Open the sidebar → **📚 Knowledge Base**
2. Click **Browse files** and upload PDF/TXT/MD files
3. Click **🔄 Build / Rebuild Vector Index**
4. Wait for the success message

### Step 2 – Ask a Question
1. Type your question in the main text area
2. Check **📊 Run RAGAS Evaluation** (recommended)
3. Click **🚀 Run Agentic Workflow**

### Step 3 – Review Output
- **Agent Execution Workflow** — see all 3 required agents completing
- **Final Answer** — structured response with citations
- **Retrieved Context** — raw chunks used by agents
- **RAGAS Scores** — quality metrics with interpretation

---

## 🤖 Agent Design

| # | Agent | Role | Tools Used |
|---|---|---|---|
| 1 | **Research** | Retrieves context | RAG search, Filesystem MCP, SQLite MCP |
| 2 | **Response** | Generates answer | SQLite MCP (save answer) |
| 3 | **Evaluation** | Final polish | None (synthesis only) |

---

## 🛠️ MCP Servers

### Filesystem MCP
- `MCP_Filesystem_List` — list files in KB directory
- `MCP_Filesystem_Read` — read text file content

### SQLite MCP
- `MCP_SQLite_Get_Documents` — list indexed documents
- `MCP_SQLite_Search_History` — find past similar queries
- `MCP_SQLite_Search_Knowledge` — search KB entries
- `MCP_SQLite_Save_Answer` — persist answers to DB
- `MCP_SQLite_Get_Stats` — system statistics

---

## 📊 RAGAS Evaluation

| Metric | Description | Target |
|---|---|---|
| **Faithfulness** | Answer grounded in context | ≥ 0.70 |
| **Answer Relevancy** | Answer addresses the question | ≥ 0.70 |
| **Context Precision** | Retrieved chunks are relevant | ≥ 0.70 |
| **Context Recall** | Enough context was retrieved | ≥ 0.70 |

---

## 🧪 Sample Inputs & Expected Outputs

### Sample 1
**Input:** `"What is the company leave policy?"`
**Expected:** Structured answer about leave types, days, approval process

### Sample 2
**Input:** `"How do I submit a support ticket?"`
**Expected:** Step-by-step ticket submission guide

### Sample 3
**Input:** `"What are the data security requirements?"`
**Expected:** Summary of data security policies with references

---

## 🎯 Evaluation Criteria Coverage

| Criteria | Weight | Implementation |
|---|---|---|
| CrewAI Multi-Agent Design | 25% | 3 mandatory agents |
| RAG Implementation | 25% | ChromaDB + nomic-embed-text |
| RAGAS Evaluation | 20% | All 4 mandatory metrics |
| MCP Integration | 20% | Filesystem + SQLite MCP |
| Documentation | 10% | This README |

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11.15 |
| UI | Streamlit 1.39 |
| Agent Framework | CrewAI 0.86 |
| LLM | Ollama / llama3.1:8b |
| Embeddings | Ollama / nomic-embed-text |
| Vector DB | ChromaDB |
| RAG Framework | LangChain |
| Evaluation | RAGAS |
| MCP Servers | Filesystem + SQLite |
| Persistence | SQLite3 |