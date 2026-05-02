# 🔍 APIMatic Doc Search

**AI-powered semantic search over APIMatic documentation** — powered by FAISS embeddings + Groq LLM (Qwen 3 32B).

A production-grade internal tool that lets your team search APIMatic docs with natural language and get intelligent, context-aware answers.

---

## ✨ Features

- **Semantic Search** — Understands meaning, not just keywords. Ask questions in plain English.
- **RAG-Powered Answers** — AI reads the relevant docs and generates comprehensive answers with source citations.
- **Streaming Responses** — LLM answers stream in real-time for instant feedback.
- **Fast Vector Search** — FAISS index returns results in <50ms.
- **Beautiful Web UI** — Premium dark theme with glassmorphism design, responsive layout.
- **100% Free** — Uses free-tier Groq API (1K requests/day) + local embeddings.
- **Production Ready** — FastAPI backend, Docker support, one-click Render deployment.

---

## 🏗️ Architecture

```
User (Browser) → FastAPI Backend
                   ├─ 1. Embed query (sentence-transformers)
                   ├─ 2. FAISS similarity search (top-K chunks)
                   └─ 3. RAG prompt → Groq API (Qwen 3 32B) → Streaming answer
```

| Component | Technology | Cost |
|-----------|-----------|------|
| Backend | FastAPI (Python) | Free |
| Frontend | Vanilla HTML/CSS/JS | Free |
| Embeddings | `all-mpnet-base-v2` | Free (local) |
| Vector DB | FAISS (persistent) | Free |
| LLM | Groq — Qwen 3 32B | Free tier |
| Deployment | Render.com | Free tier |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git
- [Groq API key](https://console.groq.com) (free)

### 1. Install Dependencies
```bash
pip install -r requirement.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Add Documentation Data

**Option A** — Clone your docs repo:
```bash
python build_index.py --clone --repo-url https://github.com/your-org/your-docs-repo.git
```

**Option B** — Copy markdown files manually:
```bash
# Place .md files in the data/ directory
mkdir data/my-docs
cp /path/to/docs/*.md data/my-docs/
python build_index.py --data-dir data/my-docs
```

**Option C** — Use the sample dataset:
```bash
python build_index.py --data-dir data/sample_dataset
```

### 4. Start the Server
```bash
python -m app.server
```

Open **http://localhost:8000** in your browser. 🎉

---

## 📁 Project Structure

```
doc-search-apimatic/
├── app/
│   ├── __init__.py         # Package init
│   ├── config.py           # Configuration (models, LLM, paths, server)
│   ├── utils.py            # Document loading, chunking, embeddings, FAISS
│   ├── llm_service.py      # Groq LLM integration (RAG answers)
│   ├── server.py           # FastAPI REST API
│   ├── gui.py              # Legacy Tkinter GUI (still works)
│   └── main.py             # Legacy GUI entry point
├── templates/
│   └── index.html          # Web frontend
├── static/
│   ├── styles.css          # Premium dark theme
│   └── app.js              # Frontend logic
├── data/                   # Place docs here
│   └── sample_dataset/     # Sample AI documents
├── Vector_Store/           # Pre-built FAISS index
├── build_index.py          # Index builder script
├── Dockerfile              # Docker deployment
├── render.yaml             # Render.com one-click deploy
├── requirement.txt         # Python dependencies
├── .env.example            # Environment template
└── .gitignore
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/stats` | Index statistics |
| `POST` | `/api/search` | Full search with LLM answer |
| `GET` | `/api/search/quick?q=...&k=5` | Fast vector-only search |
| `POST` | `/api/search/stream` | Streaming search (SSE) |

### Example — Full Search
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I generate SDKs?", "top_k": 5, "use_llm": true}'
```

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t apimatic-doc-search .

# Run
docker run -p 8000:8000 --env-file .env apimatic-doc-search
```

---

## ☁️ Deploy to Render (Free)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml`
5. Add `GROQ_API_KEY` in the Environment tab
6. Deploy! 🚀

> **Note:** Free tier sleeps after 15 min inactivity. First request after sleep takes ~30s.

---

## 🔧 Rebuilding the Index

After adding/updating documentation:

```bash
# Re-clone and rebuild
python build_index.py --clone

# Or rebuild from existing data
python build_index.py --data-dir data/apimatic-docs
```

---

## 🤝 Legacy GUI

The original Tkinter GUI still works:
```bash
python -m app.main
```

---

## 📄 License

Internal tool — APIMatic.
