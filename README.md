# 🧠 Codebase Intelligence Platform

An AI-powered code understanding platform that enables developers to analyze, search, and chat with any codebase using Retrieval-Augmented Generation (RAG), hybrid search, call graph analysis, and automated security scanning.

Supports Python, Java, JavaScript, TypeScript, and C++ repositories.

---

## ✨ Features

### 🔍 Intelligent Code Search
- Hybrid retrieval using:
  - Semantic embeddings (FAISS + BGE)
  - BM25 keyword retrieval
- Cross-encoder reranking for improved precision

### 💬 Natural Language Code Understanding
Ask questions like:
- *How does authentication work?*
- *Where is request dispatching handled?*
- *What functions call login()?*

Answers are generated using a local LLM via Ollama.

### 🔀 Call Graph Visualization
- Static graph generation
- Interactive dependency graphs
- Function-level caller/callee tracing

### 🛡️ Security Analysis
Automatically detects:
- Unsafe `eval()` usage
- `exec()` execution
- Weak cryptographic hashes
- Hardcoded secrets
- Suspicious exception handling

### 📊 Impact Analysis
Analyze:
- Which functions depend on a target function
- Multi-hop dependency chains
- Change impact risk

### 🐳 Containerized Deployment
Fully containerized using:
- Docker
- Docker Compose

---

## 🏗️ System Architecture

```text
User Query
   ↓
FastAPI Backend
   ↓
Hybrid Retrieval (BM25 + FAISS)
   ↓
Cross Encoder Reranking
   ↓
Ollama LLM
   ↓
Answer + Graph + Relevant Code
```

---

## 🧠 Models Used

### Embeddings
- `BAAI/bge-base-en-v1.5`

### Reranker
- `cross-encoder/ms-marco-MiniLM-L-6-v2`

### LLM
- `llama3.2:3b`

---

## 📁 Project Structure

```text
codebase-rag/
│
├── api.py
├── app.py
├── config.py
├── main.py
├── vulnerability_scanner.py
│
├── cache/
├── embeddings/
├── parser/
├── retrieval/
├── generator/
├── services/
├── visualization/
├── templates/
├── static/
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

### Backend
- FastAPI
- Python 3.11

### AI / Retrieval
- FAISS
- Sentence Transformers
- BM25
- Transformers

### Visualization
- NetworkX
- Matplotlib
- Pyvis

### Deployment
- Docker
- Docker Compose

---

## 🚀 Running the Project

### Local Development

```bash
uvicorn api:app --reload
```

Open:

```text
http://localhost:8000/docs
```

---

### Docker

```bash
docker compose up --build
```

Open:

```text
http://localhost:8000
```

---

## 📡 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/chat` | Ask questions about the repository |
| `/clone` | Clone a GitHub repository |
| `/load` | Load a local repository |
| `/impact` | Multi-hop dependency analysis |
| `/vulnerabilities` | Security scanning |
| `/export` | Generate HTML security report |
| `/metrics` | Performance metrics |
| `/health` | System health check |

---

## 📈 Performance

Evaluated on the Flask repository:

| Metric | Hybrid + Rerank |
|--------|-----------------|
| Hit@1 | 0.76 |
| Hit@3 | 0.92 |
| Hit@5 | 1.00 |
| MRR | 0.845 |

---

## 🎯 Use Cases

- Large codebase exploration
- Developer onboarding
- Security auditing
- Dependency analysis
- Architecture understanding
- Impact analysis before refactoring

---

## 📄 License

MIT License
