"""
LLM Service for APIMatic Doc Search
Handles RAG-powered answer generation using Groq API with Qwen model.
"""

import time
import logging
from typing import List, Dict, Optional, AsyncGenerator
from dataclasses import dataclass

from openai import OpenAI, APIError, RateLimitError
from langchain_core.documents import Document

from app.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_BASE_URL,
    GROQ_MAX_TOKENS,
    GROQ_TEMPERATURE,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# RAG Prompt Template
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert APIMatic documentation assistant. Your role is to help developers find and understand APIMatic's features, APIs, SDKs, and tools.

RULES:
1. Answer ONLY based on the provided documentation context. Never fabricate information.
2. If the context doesn't contain enough information, say so clearly and suggest what the user might search for instead.
3. Use clear, concise language appropriate for developers.
4. When referencing specific features or configurations, cite the source document.
5. Format your response using markdown for readability (headers, code blocks, lists).
6. If multiple documents are relevant, synthesize the information coherently.
7. For code-related questions, provide code examples when available in the context."""

USER_PROMPT_TEMPLATE = """Based on the following documentation excerpts, answer the user's question.

--- DOCUMENTATION CONTEXT ---
{context}
--- END CONTEXT ---

USER QUESTION: {question}

Provide a comprehensive, accurate answer based solely on the above context. Cite the source documents when possible."""


@dataclass
class LLMResponse:
    """Structured LLM response with metadata."""
    answer: str
    model: str
    usage: Dict
    latency_ms: float
    sources: List[Dict]
    success: bool
    error: Optional[str] = None


class GroqLLMService:
    """
    Manages LLM interactions via Groq API.
    Implements retry logic, rate limit handling, and context formatting.
    """

    def __init__(self):
        """Initialize the Groq client."""
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not set — LLM features will be unavailable")
            self.client = None
            return

        self.client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
        )
        self.model = GROQ_MODEL
        self.max_tokens = GROQ_MAX_TOKENS
        self.temperature = GROQ_TEMPERATURE

        # Rate limiting state
        self._request_timestamps: List[float] = []
        self._daily_count = 0
        self._daily_reset = time.time()

        logger.info(f"Groq LLM service initialized with model: {self.model}")

    @property
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self.client is not None

    def _format_context(self, search_results: List[tuple]) -> str:
        """
        Format search results into a context string for the LLM prompt.

        Args:
            search_results: List of (Document, score) tuples from vector search

        Returns:
            Formatted context string
        """
        context_parts = []
        for idx, (doc, score) in enumerate(search_results, 1):
            source = doc.metadata.get("filename", doc.metadata.get("source", "Unknown"))
            title = doc.metadata.get("title", doc.metadata.get("fm_title", ""))
            category = doc.metadata.get("category", "")

            header = f"[Document {idx}]"
            if title:
                header += f" Title: {title}"
            header += f" | Source: {source}"
            if category:
                header += f" | Category: {category}"
            header += f" | Relevance: {score:.4f}"

            context_parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(context_parts)

    def _extract_sources(self, search_results: List[tuple]) -> List[Dict]:
        """Extract source metadata from search results for citation."""
        sources = []
        for doc, score in search_results:
            sources.append({
                "filename": doc.metadata.get("filename", "Unknown"),
                "source": doc.metadata.get("source", "Unknown"),
                "title": doc.metadata.get("title", doc.metadata.get("fm_title", "")),
                "category": doc.metadata.get("category", ""),
                "relevance_score": round(float(score), 4),
                "chunk_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })
        return sources

    def generate_answer(
        self,
        question: str,
        search_results: List[tuple],
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate an LLM-powered answer using RAG.

        Args:
            question: User's question
            search_results: Vector search results [(Document, score), ...]
            max_tokens: Override default max tokens

        Returns:
            LLMResponse with answer and metadata
        """
        sources = self._extract_sources(search_results)

        if not self.is_available:
            return LLMResponse(
                answer="",
                model="none",
                usage={},
                latency_ms=0,
                sources=sources,
                success=False,
                error="LLM service unavailable — GROQ_API_KEY not configured",
            )

        context = self._format_context(search_results)
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)

        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
                stream=False,
            )

            latency_ms = (time.time() - start_time) * 1000

            answer = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            logger.info(
                f"LLM response generated in {latency_ms:.0f}ms "
                f"({usage.get('total_tokens', 0)} tokens)"
            )

            return LLMResponse(
                answer=answer,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                sources=sources,
                success=True,
            )

        except RateLimitError as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.warning(f"Groq rate limit hit: {e}")
            return LLMResponse(
                answer="",
                model=self.model,
                usage={},
                latency_ms=latency_ms,
                sources=sources,
                success=False,
                error="Rate limit exceeded. Please try again in a moment.",
            )

        except APIError as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Groq API error: {e}")
            return LLMResponse(
                answer="",
                model=self.model,
                usage={},
                latency_ms=latency_ms,
                sources=sources,
                success=False,
                error=f"LLM API error: {str(e)}",
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Unexpected LLM error: {e}", exc_info=True)
            return LLMResponse(
                answer="",
                model=self.model,
                usage={},
                latency_ms=latency_ms,
                sources=sources,
                success=False,
                error=f"Unexpected error: {str(e)}",
            )

    def generate_answer_stream(
        self,
        question: str,
        search_results: List[tuple],
    ) -> AsyncGenerator:
        """
        Generate a streaming LLM answer for real-time display.

        Args:
            question: User's question
            search_results: Vector search results

        Yields:
            Chunks of the LLM response text
        """
        if not self.is_available:
            yield "⚠️ LLM service unavailable — GROQ_API_KEY not configured. Showing vector search results only."
            return

        context = self._format_context(search_results)
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except RateLimitError:
            yield "\n\n⚠️ Rate limit reached. Showing vector search results only."
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\n\n⚠️ LLM error: {str(e)}"


# ─────────────────────────────────────────────
# Singleton instance
# ─────────────────────────────────────────────
_llm_service: Optional[GroqLLMService] = None


def get_llm_service() -> GroqLLMService:
    """Get or create the singleton LLM service."""
    global _llm_service
    if _llm_service is None:
        _llm_service = GroqLLMService()
    return _llm_service
