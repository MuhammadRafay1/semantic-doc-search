"""
LLM Service for APIMatic Doc Search
Handles RAG-powered answer generation with citations, confidence scoring,
and merged context from both embeddings and web search results.
"""

import json
import time
import logging
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, field

from openai import OpenAI, APIError, RateLimitError
from langchain_core.documents import Document

from app.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_BASE_URL,
    GROQ_MAX_TOKENS,
    GROQ_TEMPERATURE,
    CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# RAG Prompt Templates
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert APIMatic documentation assistant. Your role is to help developers find and understand APIMatic's features, APIs, SDKs, and tools.

You MUST respond with a valid JSON object. No text before or after the JSON.

RESPONSE FORMAT (strict JSON):
{
  "answer": "Your detailed answer in markdown format...",
  "citations": [
    {
      "id": 1,
      "source": "filename or URL",
      "title": "Document title",
      "type": "embedding|web",
      "excerpt": "Brief relevant excerpt from the source"
    }
  ],
  "confidence": 0.85
}

RULES:
1. Answer ONLY based on the provided documentation context. Never fabricate information.
2. The "citations" array MUST contain every source you used. Reference them in your answer as [1], [2], etc.
3. The "confidence" field is a float from 0.0 to 1.0 representing how confident you are that your answer is correct and complete based on the available context.
   - 0.8-1.0: High confidence — context clearly answers the question
   - 0.5-0.79: Medium confidence — partial information available
   - 0.2-0.49: Low confidence — very limited relevant information
   - 0.0-0.19: No confidence — context does not address the question
4. If the context doesn't contain enough information, set confidence LOW and explain what's missing.
5. Use clear, concise language appropriate for developers.
6. Format the "answer" field using markdown for readability (headers, code blocks, lists).
7. For code-related questions, provide code examples when available in the context.
8. If multiple documents are relevant, synthesize the information coherently.
9. ALWAYS cite your sources inline using [1], [2], etc.
10. Do NOT include ```json or any wrapper around your response — output raw JSON only."""

USER_PROMPT_TEMPLATE = """Based on the following documentation context, answer the user's question.
Respond with a JSON object containing "answer" (markdown string), "citations" (array), and "confidence" (float 0-1).

--- EMBEDDING SEARCH RESULTS ---
{embedding_context}
--- END EMBEDDING RESULTS ---

--- WEB SEARCH RESULTS (from docs.apimatic.io) ---
{web_context}
--- END WEB RESULTS ---

USER QUESTION: {question}

Remember: Output ONLY valid JSON. Include inline citations [1], [2] etc. in your answer and list all sources in the citations array."""


@dataclass
class Citation:
    """A single citation reference."""
    id: int
    source: str
    title: str
    type: str  # "embedding" or "web"
    excerpt: str
    url: Optional[str] = None


@dataclass
class LLMResponse:
    """Structured LLM response with citations and confidence."""
    answer: str
    model: str
    usage: Dict
    latency_ms: float
    citations: List[Dict]
    confidence: float
    is_confident: bool
    sources: List[Dict]
    success: bool
    error: Optional[str] = None


class GroqLLMService:
    """
    Manages LLM interactions via Groq API.
    Implements citations, confidence scoring, and merged context from
    embedding search + web search results.
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
        self.confidence_threshold = CONFIDENCE_THRESHOLD

        # Rate limiting state
        self._request_timestamps: List[float] = []
        self._daily_count = 0
        self._daily_reset = time.time()

        logger.info(f"Groq LLM service initialized with model: {self.model}")

    @property
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self.client is not None

    def _format_embedding_context(self, search_results: List[tuple]) -> str:
        """
        Format embedding search results into a context string for the LLM prompt.

        Args:
            search_results: List of (Document, score) tuples from vector search

        Returns:
            Formatted context string
        """
        if not search_results:
            return "(No embedding results available)"

        context_parts = []
        for idx, (doc, score) in enumerate(search_results, 1):
            source = doc.metadata.get("filename", doc.metadata.get("source", "Unknown"))
            title = doc.metadata.get("title", doc.metadata.get("fm_title", ""))
            category = doc.metadata.get("category", "")

            header = f"[Embedding Doc {idx}]"
            if title:
                header += f" Title: {title}"
            header += f" | Source: {source}"
            if category:
                header += f" | Category: {category}"
            header += f" | Relevance: {score:.4f}"

            context_parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(context_parts)

    def _format_web_context(self, web_results: List[Dict]) -> str:
        """
        Format web search results into a context string for the LLM prompt.

        Args:
            web_results: List of web search result dicts with title, url, snippet

        Returns:
            Formatted context string
        """
        if not web_results:
            return "(No web search results available)"

        context_parts = []
        for idx, result in enumerate(web_results, 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            snippet = result.get("snippet", "")

            header = f"[Web Doc {idx}] Title: {title} | URL: {url}"
            context_parts.append(f"{header}\n{snippet}")

        return "\n\n---\n\n".join(context_parts)

    def _extract_sources(self, search_results: List[tuple], web_results: List[Dict] = None) -> List[Dict]:
        """Extract source metadata from all search results for citation."""
        sources = []

        # Embedding sources
        for doc, score in search_results:
            sources.append({
                "filename": doc.metadata.get("filename", "Unknown"),
                "source": doc.metadata.get("source", "Unknown"),
                "title": doc.metadata.get("title", doc.metadata.get("fm_title", "")),
                "category": doc.metadata.get("category", ""),
                "relevance_score": round(float(score), 4),
                "type": "embedding",
                "chunk_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })

        # Web sources
        if web_results:
            for result in web_results:
                sources.append({
                    "filename": result.get("title", "Web Result"),
                    "source": result.get("url", ""),
                    "title": result.get("title", ""),
                    "category": "docs.apimatic.io",
                    "relevance_score": 0,
                    "type": "web",
                    "url": result.get("url", ""),
                    "chunk_preview": result.get("snippet", ""),
                })

        return sources

    def _parse_llm_json(self, raw_text: str) -> Dict:
        """
        Parse the LLM's JSON response, handling common formatting issues.

        Args:
            raw_text: Raw text from LLM

        Returns:
            Parsed dictionary
        """
        text = raw_text.strip()

        # Remove markdown JSON wrapper if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Handle <think> tags from Qwen model
        if "<think>" in text:
            # Remove everything between <think> and </think>
            import re
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

            # Fallback: return raw text as answer with low confidence
            logger.warning(f"Failed to parse LLM JSON response, using raw text fallback")
            return {
                "answer": raw_text,
                "citations": [],
                "confidence": 0.3,
            }

    def generate_answer(
        self,
        question: str,
        search_results: List[tuple],
        web_results: List[Dict] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate an LLM-powered answer using RAG with citations and confidence scoring.

        Args:
            question: User's question
            search_results: Vector search results [(Document, score), ...]
            web_results: Web search results [{"title", "url", "snippet"}, ...]
            max_tokens: Override default max tokens

        Returns:
            LLMResponse with answer, citations, and confidence
        """
        sources = self._extract_sources(search_results, web_results)

        if not self.is_available:
            return LLMResponse(
                answer="",
                model="none",
                usage={},
                latency_ms=0,
                citations=[],
                confidence=0.0,
                is_confident=False,
                sources=sources,
                success=False,
                error="LLM service unavailable — GROQ_API_KEY not configured",
            )

        embedding_context = self._format_embedding_context(search_results)
        web_context = self._format_web_context(web_results or [])
        user_prompt = USER_PROMPT_TEMPLATE.format(
            embedding_context=embedding_context,
            web_context=web_context,
            question=question,
        )

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

            raw_answer = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            # Parse structured JSON response
            parsed = self._parse_llm_json(raw_answer)
            answer = parsed.get("answer", raw_answer)
            citations = parsed.get("citations", [])
            confidence = float(parsed.get("confidence", 0.5))

            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))
            is_confident = confidence >= self.confidence_threshold

            # If not confident, replace answer with uncertainty message
            if not is_confident:
                answer = (
                    "⚠️ **I'm not confident enough to answer this question accurately.**\n\n"
                    f"Based on the available documentation, I found limited relevant information "
                    f"(confidence: {confidence:.0%}). Here's what I can share, but please verify:\n\n"
                    f"---\n\n{answer}"
                )

            logger.info(
                f"LLM response: {latency_ms:.0f}ms, "
                f"{usage.get('total_tokens', 0)} tokens, "
                f"confidence={confidence:.2f}, "
                f"citations={len(citations)}"
            )

            return LLMResponse(
                answer=answer,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                citations=citations,
                confidence=confidence,
                is_confident=is_confident,
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
                citations=[],
                confidence=0.0,
                is_confident=False,
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
                citations=[],
                confidence=0.0,
                is_confident=False,
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
                citations=[],
                confidence=0.0,
                is_confident=False,
                sources=sources,
                success=False,
                error=f"Unexpected error: {str(e)}",
            )

    def generate_answer_stream(
        self,
        question: str,
        search_results: List[tuple],
        web_results: List[Dict] = None,
    ) -> Generator:
        """
        Generate a streaming LLM answer, then yield citations and confidence at the end.
        Since we need structured JSON for citations/confidence, we collect the full
        response, parse it, then yield the answer in chunks + metadata.

        Args:
            question: User's question
            search_results: Vector search results
            web_results: Web search results

        Yields:
            Chunks of the LLM response text
        """
        if not self.is_available:
            yield "⚠️ LLM service unavailable — GROQ_API_KEY not configured. Showing search results only."
            return

        embedding_context = self._format_embedding_context(search_results)
        web_context = self._format_web_context(web_results or [])
        user_prompt = USER_PROMPT_TEMPLATE.format(
            embedding_context=embedding_context,
            web_context=web_context,
            question=question,
        )

        try:
            # For structured output, we need the full response (not truly streaming tokens)
            # because we need to parse JSON. We'll do a non-stream call and simulate chunking.
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False,
            )

            raw_answer = response.choices[0].message.content or ""
            parsed = self._parse_llm_json(raw_answer)

            answer = parsed.get("answer", raw_answer)
            citations = parsed.get("citations", [])
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            is_confident = confidence >= self.confidence_threshold

            # If not confident, prepend warning
            if not is_confident:
                warning = (
                    "⚠️ **I'm not confident enough to answer this question accurately.**\n\n"
                    f"Based on the available documentation, I found limited relevant information "
                    f"(confidence: {confidence:.0%}). Here's what I can share, but please verify:\n\n---\n\n"
                )
                answer = warning + answer

            # Yield the answer in chunks for streaming effect
            chunk_size = 12
            for i in range(0, len(answer), chunk_size):
                yield answer[i:i + chunk_size]

            # Yield metadata as special JSON tokens (the frontend knows to parse these)
            yield f"\n__CITATIONS__{json.dumps(citations)}__END_CITATIONS__"
            yield f"\n__CONFIDENCE__{json.dumps({'confidence': confidence, 'is_confident': is_confident})}__END_CONFIDENCE__"

        except RateLimitError:
            yield "\n\n⚠️ Rate limit reached. Showing search results only."
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
