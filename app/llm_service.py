"""
LLM Service for APIMatic Doc Search
Handles RAG-powered answer generation with citations, confidence scoring,
and merged context from both embeddings and web search results.
"""

import json
import re
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
SYSTEM_PROMPT = """/no_think
You are an expert APIMatic documentation assistant. You help developers understand APIMatic's platform — API transformation, SDK generation, developer portals, API validation, and related features.

You MUST respond with a single valid JSON object. No markdown fencing, no text before or after the JSON.

JSON SCHEMA:
{
  "answer": "<detailed markdown answer>",
  "citations": [
    {
      "id": 1,
      "source": "<filename or page title>",
      "title": "<document title>",
      "type": "embedding|web",
      "url": "<full URL if web source, empty string if embedding>",
      "excerpt": "<a short verbatim excerpt you used from this source>"
    }
  ],
  "confidence": 0.85
}

CRITICAL RULES:
1. Your answer MUST be detailed, thorough, and developer-friendly. Use markdown headers (##), bullet points, code blocks, and step-by-step instructions where relevant. Do NOT give one-liner answers.
2. Answer ONLY based on the provided context. Never invent information.
3. EVERY source you reference must appear in the "citations" array. Use inline references like [1], [2] in your answer text.
4. For WEB sources, ALWAYS include the full URL in the "url" field. This is critical — users need clickable links.
5. For EMBEDDING sources, set "url" to an empty string "".
6. "confidence" is a float 0.0-1.0:
   - 0.8-1.0 = context clearly answers the question
   - 0.5-0.79 = partial/incomplete information
   - 0.0-0.49 = insufficient context to answer well
7. If the context doesn't answer the question, set confidence below 0.4 and explain what information is missing.
8. Synthesize information from BOTH embedding and web sources to give the most complete answer possible.
9. When web sources have relevant content, ALWAYS cite them with their URL so users can click through."""

USER_PROMPT_TEMPLATE = """Answer the user's question using ALL of the context below. Be detailed and thorough. Cite every source you use.

══════════ EMBEDDING SOURCES (from local index) ══════════
{embedding_context}
══════════ END EMBEDDING SOURCES ══════════

══════════ WEB SOURCES (from docs.apimatic.io) ══════════
{web_context}
══════════ END WEB SOURCES ══════════

USER QUESTION: {question}

Respond with a JSON object containing "answer" (detailed markdown), "citations" (array with URLs for web sources), and "confidence" (float 0-1). Output raw JSON only."""


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

            header = f"[Embedding Source {idx}]"
            if title:
                header += f" Title: {title}"
            header += f" | File: {source}"
            if category:
                header += f" | Category: {category}"
            header += f" | Relevance: {score:.4f}"

            context_parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(context_parts)

    def _format_web_context(self, web_results: List[Dict]) -> str:
        """
        Format web search results into a context string for the LLM prompt.
        Includes scraped page content when available, falling back to snippet.

        Args:
            web_results: List of web search result dicts with title, url, snippet, page_content

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
            page_content = result.get("page_content", "")

            header = f"[Web Source {idx}] Title: {title}\nURL: {url}"

            # Use scraped page content if available, otherwise use snippet
            if page_content and len(page_content) > len(snippet):
                content = page_content
            else:
                content = snippet

            context_parts.append(f"{header}\n\n{content}")

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
        Parse the LLM's JSON response, handling Qwen think tags and common formatting issues.

        Args:
            raw_text: Raw text from LLM

        Returns:
            Parsed dictionary
        """
        text = raw_text.strip()

        # Strip <think>...</think> blocks (Qwen model reasoning)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Remove markdown JSON wrapper if present
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text (handles leading/trailing garbage)
            brace_start = text.find("{")
            brace_end = text.rfind("}") + 1
            if brace_start >= 0 and brace_end > brace_start:
                candidate = text[brace_start:brace_end]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Try fixing common issues: unescaped newlines in JSON strings
                    try:
                        # Replace literal newlines inside strings with \\n
                        fixed = candidate.replace("\n", "\\n").replace("\r", "")
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        pass

            # Final fallback: return raw text as answer
            logger.warning("Failed to parse LLM JSON response, using raw text fallback")
            # Try to clean up the text for display
            display_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
            display_text = re.sub(r"```json\s*", "", display_text)
            display_text = re.sub(r"\s*```", "", display_text)
            return {
                "answer": display_text,
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
            web_results: Web search results [{"title", "url", "snippet", "page_content"}, ...]
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

            # If not confident, prepend uncertainty notice
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
        Generate LLM answer and yield answer text in chunks + metadata at end.
        Uses non-streaming API call since we need to parse structured JSON,
        then simulates streaming for the UI.

        Args:
            question: User's question
            search_results: Vector search results
            web_results: Web search results

        Yields:
            Chunks of the answer text, then special metadata tokens
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
            chunk_size = 15
            for i in range(0, len(answer), chunk_size):
                yield answer[i:i + chunk_size]

            # Yield metadata as special tokens (frontend parses these)
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
