"""
FastAPI Server for APIMatic Doc Search
Production-grade REST API with semantic search and RAG-powered answers.
"""

import time
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from app.config import (
    HOST,
    PORT,
    DEBUG,
    FAISS_INDEX_PATH,
    VECTOR_STORE_DIR,
    STATIC_DIR,
    TEMPLATES_DIR,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    GROQ_API_KEY,
    GROQ_MODEL,
    EMBEDDING_MODEL_NAME,
)
from app.utils import EmbeddingEngine, VectorStoreManager
from app.llm_service import get_llm_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────
_vector_manager: Optional[VectorStoreManager] = None
_index_stats: dict = {}


def _load_index() -> bool:
    """Load the pre-built FAISS index on startup."""
    global _vector_manager, _index_stats

    index_path = FAISS_INDEX_PATH
    if not Path(f"{index_path}").exists() and not Path(f"{index_path}.faiss").exists():
        # Check if the directory-based FAISS index exists
        index_dir = VECTOR_STORE_DIR / "apimatic_faiss_index_faiss"
        if not index_dir.exists():
            logger.warning(f"No FAISS index found at {index_path}. Run build_index.py first.")
            return False

    try:
        engine = EmbeddingEngine.get_instance()
        _vector_manager = VectorStoreManager("FAISS", engine)

        # Try the standard naming convention
        if _vector_manager.load_vector_store("apimatic_faiss_index"):
            logger.info("✅ FAISS index loaded successfully")
            return True

        # Try loading directly from the path
        from langchain_community.vectorstores import FAISS as FAISSStore
        _vector_manager.vector_store = FAISSStore.load_local(
            str(index_path),
            engine.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("✅ FAISS index loaded successfully (direct path)")
        return True

    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        return False


# ─────────────────────────────────────────────
# Application Lifecycle
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle handler."""
    logger.info("🚀 Starting APIMatic Doc Search API...")

    # Load index
    index_loaded = _load_index()
    if index_loaded:
        logger.info("📚 Index ready for queries")
    else:
        logger.warning("⚠️ No index loaded — run `python build_index.py` first")

    # Initialize LLM
    llm = get_llm_service()
    if llm.is_available:
        logger.info(f"🤖 LLM ready: {GROQ_MODEL}")
    else:
        logger.warning("⚠️ LLM unavailable — set GROQ_API_KEY in .env")

    yield

    logger.info("👋 Shutting down APIMatic Doc Search API")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="APIMatic Doc Search",
    description="Semantic search over APIMatic documentation with RAG-powered answers",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K, description="Number of results")
    use_llm: bool = Field(True, description="Whether to use LLM for answer generation")


class SearchResult(BaseModel):
    filename: str
    source: str
    title: str
    category: str
    relevance_score: float
    chunk_preview: str


class SearchResponse(BaseModel):
    query: str
    answer: Optional[str] = None
    results: list[SearchResult]
    llm_model: Optional[str] = None
    llm_latency_ms: Optional[float] = None
    llm_tokens: Optional[dict] = None
    llm_error: Optional[str] = None
    search_latency_ms: float
    total_results: int


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main search UI."""
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>APIMatic Doc Search</h1><p>Frontend not found. Place index.html in templates/</p>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "index_loaded": _vector_manager is not None and _vector_manager.vector_store is not None,
        "llm_available": get_llm_service().is_available,
        "llm_model": GROQ_MODEL if get_llm_service().is_available else None,
        "embedding_model": EMBEDDING_MODEL_NAME,
    }


@app.get("/api/stats")
async def get_stats():
    """Get index statistics."""
    if _vector_manager is None or _vector_manager.vector_store is None:
        raise HTTPException(status_code=503, detail="Index not loaded")

    try:
        # FAISS-specific stats
        store = _vector_manager.vector_store
        num_vectors = store.index.ntotal if hasattr(store, 'index') else "unknown"

        return {
            "index_type": "FAISS",
            "total_vectors": num_vectors,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "llm_model": GROQ_MODEL,
            "llm_available": get_llm_service().is_available,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform semantic search with optional RAG-powered answer generation.

    - Embeds the query using sentence-transformers
    - Finds top-K similar document chunks via FAISS
    - Optionally generates an LLM answer using Groq (Qwen)
    """
    if _vector_manager is None or _vector_manager.vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Search index not loaded. Run `python build_index.py` first.",
        )

    # Vector search
    search_start = time.time()
    try:
        search_results = _vector_manager.similarity_search(request.query, k=request.top_k)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    search_latency = (time.time() - search_start) * 1000

    # Format results
    results = []
    for doc, score in search_results:
        results.append(SearchResult(
            filename=doc.metadata.get("filename", "Unknown"),
            source=doc.metadata.get("source", "Unknown"),
            title=doc.metadata.get("title", doc.metadata.get("fm_title", "")),
            category=doc.metadata.get("category", ""),
            relevance_score=round(float(score), 4),
            chunk_preview=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
        ))

    # LLM answer generation
    answer = None
    llm_model = None
    llm_latency = None
    llm_tokens = None
    llm_error = None

    if request.use_llm and search_results:
        llm_service = get_llm_service()
        if llm_service.is_available:
            llm_response = llm_service.generate_answer(request.query, search_results)
            if llm_response.success:
                answer = llm_response.answer
                llm_model = llm_response.model
                llm_latency = llm_response.latency_ms
                llm_tokens = llm_response.usage
            else:
                llm_error = llm_response.error
        else:
            llm_error = "LLM not configured — set GROQ_API_KEY"

    return SearchResponse(
        query=request.query,
        answer=answer,
        results=results,
        llm_model=llm_model,
        llm_latency_ms=llm_latency,
        llm_tokens=llm_tokens,
        llm_error=llm_error,
        search_latency_ms=round(search_latency, 2),
        total_results=len(results),
    )


@app.get("/api/search/quick")
async def quick_search(
    q: str = Query(..., min_length=1, max_length=1000, description="Search query"),
    k: int = Query(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K, description="Number of results"),
):
    """
    Fast vector-only search (no LLM). Good for autocomplete or quick lookups.
    """
    if _vector_manager is None or _vector_manager.vector_store is None:
        raise HTTPException(status_code=503, detail="Index not loaded")

    start = time.time()
    results = _vector_manager.similarity_search(q, k=k)
    latency = (time.time() - start) * 1000

    return {
        "query": q,
        "results": [
            {
                "filename": doc.metadata.get("filename", "Unknown"),
                "title": doc.metadata.get("title", doc.metadata.get("fm_title", "")),
                "category": doc.metadata.get("category", ""),
                "relevance_score": round(float(score), 4),
                "preview": doc.page_content[:300],
            }
            for doc, score in results
        ],
        "latency_ms": round(latency, 2),
    }


@app.post("/api/search/stream")
async def search_stream(request: SearchRequest):
    """
    Streaming search — returns LLM answer as a Server-Sent Event stream.
    The vector search results are sent first, then the LLM answer streams in.
    """
    if _vector_manager is None or _vector_manager.vector_store is None:
        raise HTTPException(status_code=503, detail="Index not loaded")

    search_results = _vector_manager.similarity_search(request.query, k=request.top_k)

    # Format sources for initial payload
    import json
    sources = []
    for doc, score in search_results:
        sources.append({
            "filename": doc.metadata.get("filename", "Unknown"),
            "title": doc.metadata.get("title", doc.metadata.get("fm_title", "")),
            "category": doc.metadata.get("category", ""),
            "relevance_score": round(float(score), 4),
            "preview": doc.page_content[:300],
        })

    def event_stream():
        # First, send the sources
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

        # Then stream the LLM answer
        llm_service = get_llm_service()
        if llm_service.is_available and request.use_llm:
            for chunk in llm_service.generate_answer_stream(request.query, search_results):
                yield f"data: {json.dumps({'type': 'token', 'data': chunk})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'error', 'data': 'LLM not available'})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.server:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info",
    )
