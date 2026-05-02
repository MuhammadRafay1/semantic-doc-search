"""
Configuration module for APIMatic Doc Search
Contains all configurable settings including embedding models, vector stores, LLM, and paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
VECTOR_STORE_DIR = PROJECT_ROOT / "Vector_Store"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Create directories if they don't exist
for dir_path in [DATA_DIR, EMBEDDINGS_DIR, VECTOR_STORE_DIR, EXPERIMENTS_DIR, STATIC_DIR, TEMPLATES_DIR]:
    dir_path.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# APIMatic Docs Repository
# ─────────────────────────────────────────────
APIMATIC_DOCS_REPO = os.getenv(
    "APIMATIC_DOCS_REPO",
    "https://github.com/apimatic/apimatic-docs.git"  # Placeholder — update with actual repo
)
APIMATIC_DOCS_DIR = DATA_DIR / "apimatic-docs"

# ─────────────────────────────────────────────
# Embedding Model Configuration
# ─────────────────────────────────────────────
# Using the best free sentence-transformer for semantic search
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-mpnet-base-v2"
)
EMBEDDING_MODEL_KEY = "all-mpnet-base-v2"

EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "description": "Fast and efficient, good for general use"
    },
    "all-mpnet-base-v2": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "dimension": 768,
        "description": "High quality, balanced speed/performance"
    },
    "multi-qa-MiniLM-L6-cos-v1": {
        "name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "dimension": 384,
        "description": "Optimized for question-answering"
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "description": "Supports 50+ languages"
    },
    "all-distilroberta-v1": {
        "name": "sentence-transformers/all-distilroberta-v1",
        "dimension": 768,
        "description": "High quality RoBERTa-based model"
    }
}

# Vector Store Options
VECTOR_STORES = {
    "FAISS": "Facebook AI Similarity Search - Fast in-memory search",
    "ChromaDB": "Chroma - Open-source embedding database"
}

# ─────────────────────────────────────────────
# Text Processing Settings
# ─────────────────────────────────────────────
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Markdown-aware separators (prioritize header boundaries)
MARKDOWN_SEPARATORS = [
    "\n## ",      # H2 headers
    "\n### ",     # H3 headers
    "\n#### ",    # H4 headers
    "\n\n",       # Paragraphs
    "\n",         # Lines
    " ",          # Words
    ""            # Characters
]

# ─────────────────────────────────────────────
# Search Settings
# ─────────────────────────────────────────────
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

# Supported document formats
SUPPORTED_FORMATS = ['.txt', '.pdf', '.docx', '.md']

# ─────────────────────────────────────────────
# LLM Configuration (Groq — Free Tier)
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "2048"))
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.3"))

# Rate limiting for Groq free tier
GROQ_RATE_LIMIT_RPM = 30  # Requests per minute
GROQ_RATE_LIMIT_RPD = 1000  # Requests per day

# ─────────────────────────────────────────────
# FAISS Index Persistence
# ─────────────────────────────────────────────
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "apimatic_faiss_index"

# ─────────────────────────────────────────────
# Server Configuration
# ─────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ─────────────────────────────────────────────
# GUI Settings (legacy — kept for backward compat)
# ─────────────────────────────────────────────
WINDOW_TITLE = "APIMatic Doc Search"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
