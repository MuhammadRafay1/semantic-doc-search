"""
FastAPI Server for APIMatic Doc Search
Production-grade REST API with semantic search, web search, citations, and confidence scoring.
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
    CONFIDENCE_THRESHOLD,
)
from app.utils import EmbeddingEngine, VectorStoreManager
from app.llm_service import get_llm_service
from app.web_search import get_serper_service

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

    # Initialize Serper
    serper = get_serper_service()
    if serper.is_available:
        logger.info("🌐 Serper web search ready")
    else:
        logger.warning("⚠️ Serper unavailable — set SERPER_API_KEY in .env")

    yield

    logger.info("👋 Shutting down APIMatic Doc Search API")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="APIMatic Doc Search",
    description="Semantic search over APIMatic documentation with RAG-powered answers, citations, and confidence scoring",
    version="2.0.0",
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
    use_web_search: bool = Field(True, description="Whether to include Serper web search")


class CitationItem(BaseModel):
    id: int
    source: str
    title: str
    type: str  # "embedding" or "web"
    excerpt: str
    url: Optional[str] = None


class SearchResult(BaseModel):
    filename: str
    source: str
    title: str
    category: str
    relevance_score: float
    chunk_preview: str
    result_type: str = "embedding"  # "embedding" or "web"
    url: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    answer: Optional[str] = None
    citations: list[CitationItem] = []
    confidence: Optional[float] = None
    is_confident: Optional[bool] = None
    results: list[SearchResult]
    web_results: list[SearchResult] = []
    llm_model: Optional[str] = None
    llm_latency_ms: Optional[float] = None
    llm_tokens: Optional[dict] = None
    llm_error: Optional[str] = None
    web_search_query: Optional[str] = None
    web_search_latency_ms: Optional[float] = None
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
    serper = get_serper_service()
    return {
        "status": "healthy",
        "index_loaded": _vector_manager is not None and _vector_manager.vector_store is not None,
        "llm_available": get_llm_service().is_available,
        "llm_model": GROQ_MODEL if get_llm_service().is_available else None,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "web_search_available": serper.is_available,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
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
            "web_search_available": get_serper_service().is_available,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform semantic search with optional RAG-powered answer generation.

    - Embeds the query using sentence-transformers
    - Finds top-K similar document chunks via FAISS
    - Optionally searches docs.apimatic.io via Serper
    - Generates an LLM answer with citations and confidence scoring
    """
    if _vector_manager is None or _vector_manager.vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Search index not loaded. Run `python build_index.py` first.",
        )

    # ── 1. FAISS Vector Search ──
    search_start = time.time()
    try:
        search_results = _vector_manager.similarity_search(request.query, k=request.top_k)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    search_latency = (time.time() - search_start) * 1000

    # Format embedding results
    results = []
    for doc, score in search_results:
        results.append(SearchResult(
            filename=doc.metadata.get("filename", "Unknown"),
            source=doc.metadata.get("source", "Unknown"),
            title=doc.metadata.get("title", doc.metadata.get("fm_title", "")),
            category=doc.metadata.get("category", ""),
            relevance_score=round(float(score), 4),
            chunk_preview=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
            result_type="embedding",
        ))

    # ── 2. Serper Web Search ──
    web_results_formatted = []
    web_results_raw = []
    web_search_query = None
    web_search_latency = None

    serper = get_serper_service()
    if request.use_web_search and serper.is_available:
        web_response = serper.search_for_question(request.query)
        web_search_query = web_response.query
        web_search_latency = web_response.latency_ms

        if web_response.success:
            for wr in web_response.results:
                web_results_formatted.append(SearchResult(
                    filename=wr.title,
                    source=wr.url,
                    title=wr.title,
                    category="docs.apimatic.io",
                    relevance_score=0,
                    chunk_preview=wr.snippet,
                    result_type="web",
                    url=wr.url,
                ))
                web_results_raw.append({
                    "title": wr.title,
                    "url": wr.url,
                    "snippet": wr.snippet,
                })

    # ── 3. LLM Answer Generation ──
    answer = None
    llm_model = None
    llm_latency = None
    llm_tokens = None
    llm_error = None
    citations = []
    confidence = None
    is_confident = None

    if request.use_llm and search_results:
        llm_service = get_llm_service()
        if llm_service.is_available:
            llm_response = llm_service.generate_answer(
                request.query,
                search_results,
                web_results=web_results_raw if web_results_raw else None,
            )
            if llm_response.success:
                answer = llm_response.answer
                llm_model = llm_response.model
                llm_latency = llm_response.latency_ms
                llm_tokens = llm_response.usage
                citations = [
                    CitationItem(
                        id=c.get("id", i + 1),
                        source=c.get("source", "Unknown"),
                        title=c.get("title", ""),
                        type=c.get("type", "embedding"),
                        excerpt=c.get("excerpt", ""),
                        url=c.get("url"),
                    )
                    for i, c in enumerate(llm_response.citations)
                ]
                confidence = llm_response.confidence
                is_confident = llm_response.is_confident
            else:
                llm_error = llm_response.error
        else:
            llm_error = "LLM not configured — set GROQ_API_KEY"

    return SearchResponse(
        query=request.query,
        answer=answer,
        citations=citations,
        confidence=confidence,
        is_confident=is_confident,
        results=results,
        web_results=web_results_formatted,
        llm_model=llm_model,
        llm_latency_ms=llm_latency,
        llm_tokens=llm_tokens,
        llm_error=llm_error,
        web_search_query=web_search_query,
        web_search_latency_ms=web_search_latency,
        search_latency_ms=round(search_latency, 2),
        total_results=len(results) + len(web_results_formatted),
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
    The vector search results are sent first, then web results, then the LLM answer streams in.
    Finally, citations and confidence are sent as structured events.
    """
    if _vector_manager is None or _vector_manager.vector_store is None:
        raise HTTPException(status_code=503, detail="Index not loaded")

    search_results = _vector_manager.similarity_search(request.query, k=request.top_k)

    # Format embedding sources
    import json
    sources = []
    for doc, score in search_results:
        sources.append({
            "filename": doc.metadata.get("filename", "Unknown"),
            "title": doc.metadata.get("title", doc.metadata.get("fm_title", "")),
            "category": doc.metadata.get("category", ""),
            "relevance_score": round(float(score), 4),
            "preview": doc.page_content[:300],
            "type": "embedding",
        })

    # Web search
    web_results_raw = []
    web_sources = []
    serper = get_serper_service()
    if request.use_web_search and serper.is_available:
        web_response = serper.search_for_question(request.query)
        if web_response.success:
            for wr in web_response.results:
                web_results_raw.append({
                    "title": wr.title,
                    "url": wr.url,
                    "snippet": wr.snippet,
                })
                web_sources.append({
                    "filename": wr.title,
                    "title": wr.title,
                    "url": wr.url,
                    "category": "docs.apimatic.io",
                    "relevance_score": 0,
                    "preview": wr.snippet,
                    "type": "web",
                })

    def event_stream():
        # First, send the embedding sources
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

        # Send web sources
        if web_sources:
            yield f"data: {json.dumps({'type': 'web_sources', 'data': web_sources})}\n\n"

        # Then stream the LLM answer
        llm_service = get_llm_service()
        if llm_service.is_available and request.use_llm:
            answer_text = ""
            for chunk in llm_service.generate_answer_stream(
                request.query,
                search_results,
                web_results=web_results_raw if web_results_raw else None,
            ):
                # Check for special metadata tokens
                if chunk.startswith("\n__CITATIONS__"):
                    citations_json = chunk.replace("\n__CITATIONS__", "").replace("__END_CITATIONS__", "")
                    try:
                        citations = json.loads(citations_json)
                        yield f"data: {json.dumps({'type': 'citations', 'data': citations})}\n\n"
                    except json.JSONDecodeError:
                        pass
                elif chunk.startswith("\n__CONFIDENCE__"):
                    conf_json = chunk.replace("\n__CONFIDENCE__", "").replace("__END_CONFIDENCE__", "")
                    try:
                        conf = json.loads(conf_json)
                        yield f"data: {json.dumps({'type': 'confidence', 'data': conf})}\n\n"
                    except json.JSONDecodeError:
                        pass
                else:
                    answer_text += chunk
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
