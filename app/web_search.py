"""
Serper Web Search Service for APIMatic Doc Search
Searches docs.apimatic.io via Google Serper API with LLM-optimized query generation.
Also scrapes actual page content from search result URLs for richer LLM context.
"""

import logging
import time
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from openai import OpenAI

from app.config import (
    SERPER_API_KEY,
    SERPER_BASE_URL,
    SERPER_SITE_FILTER,
    SERPER_NUM_RESULTS,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────
@dataclass
class WebSearchResult:
    """A single web search result."""
    title: str
    url: str
    snippet: str
    page_content: str = ""  # Scraped full page text
    position: int = 0


@dataclass
class WebSearchResponse:
    """Aggregated web search response."""
    query: str
    results: List[WebSearchResult] = field(default_factory=list)
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


# ─────────────────────────────────────────────
# Query Generation Prompt
# ─────────────────────────────────────────────
QUERY_GEN_PROMPT = """You are a search query optimizer. Given a user's natural language question about APIMatic documentation, generate an optimized Google search query.

RULES:
1. The query will be automatically scoped to site:docs.apimatic.io — do NOT include site: in your output.
2. Use concise, keyword-rich phrases that Google would match well.
3. Remove filler words (how do I, what is the, can you, etc.) and keep technical terms.
4. Output ONLY the search query string, nothing else. No quotes, no explanation.
5. Do NOT use <think> tags or any reasoning — just output the query directly.

Examples:
- User: "How do I generate SDKs from my API specification?" → generate SDK API specification
- User: "What authentication methods does APIMatic support?" → authentication methods supported
- User: "How to customize the developer portal theme?" → customize developer portal theme
- User: "Can I import an OpenAPI 3.0 file?" → import OpenAPI 3.0"""


def _scrape_page_content(url: str, timeout: int = 8) -> str:
    """
    Scrape text content from a URL, stripping HTML tags.
    Returns cleaned text content (up to 3000 chars for LLM context window).

    Args:
        url: Page URL to scrape
        timeout: Request timeout in seconds

    Returns:
        Cleaned text content from the page
    """
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; APIMatic-DocSearch/2.0)",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        resp.raise_for_status()
        html = resp.text

        # Remove script, style, nav, header, footer tags and their content
        for tag in ["script", "style", "nav", "header", "footer", "noscript", "svg", "iframe"]:
            html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE)

        # Remove all HTML tags
        text = re.sub(r"<[^>]+>", " ", html)

        # Decode HTML entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&nbsp;", " ").replace("&quot;", '"').replace("&#39;", "'")

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Limit to ~3000 chars to avoid blowing up the context window
        if len(text) > 3000:
            text = text[:3000] + "..."

        return text

    except Exception as e:
        logger.debug(f"Failed to scrape {url}: {e}")
        return ""


def _scrape_pages_parallel(urls: List[str], max_workers: int = 3) -> Dict[str, str]:
    """
    Scrape multiple pages in parallel.

    Args:
        urls: List of URLs to scrape
        max_workers: Max concurrent scrapers

    Returns:
        Dict mapping URL to scraped content
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(_scrape_page_content, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception:
                results[url] = ""
    return results


class SerperSearchService:
    """
    Manages web search via Serper API, scoped to APIMatic documentation.
    Uses LLM to generate optimized search queries from natural language.
    Scrapes actual page content from result URLs for richer context.
    """

    def __init__(self):
        """Initialize Serper search service."""
        self._serper_available = bool(SERPER_API_KEY)
        self._llm_available = bool(GROQ_API_KEY)

        if not self._serper_available:
            logger.warning("SERPER_API_KEY not set — web search will be unavailable")

        if self._llm_available:
            self._llm_client = OpenAI(
                api_key=GROQ_API_KEY,
                base_url=GROQ_BASE_URL,
            )
        else:
            self._llm_client = None

        logger.info(
            f"Serper service initialized (serper={'✅' if self._serper_available else '❌'}, "
            f"query_gen={'✅' if self._llm_available else '❌'})"
        )

    @property
    def is_available(self) -> bool:
        """Check if Serper search is available."""
        return self._serper_available

    def generate_search_query(self, user_question: str) -> str:
        """
        Use LLM to transform a natural language question into an optimized search query.

        Args:
            user_question: The user's raw question

        Returns:
            Optimized search query string
        """
        if not self._llm_available:
            return self._simple_query_transform(user_question)

        try:
            response = self._llm_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": QUERY_GEN_PROMPT},
                    {"role": "user", "content": user_question},
                ],
                max_tokens=60,
                temperature=0.1,
                stream=False,
            )
            raw = response.choices[0].message.content or ""

            # Strip <think>...</think> tags if Qwen model outputs them
            query = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            query = query.strip("\"'").strip()

            # If empty after stripping, fallback
            if not query:
                return self._simple_query_transform(user_question)

            logger.info(f"LLM query transform: '{user_question}' → '{query}'")
            return query

        except Exception as e:
            logger.warning(f"Query generation failed, using fallback: {e}")
            return self._simple_query_transform(user_question)

    def _simple_query_transform(self, question: str) -> str:
        """Simple fallback query transformation without LLM."""
        filler_words = {
            "how", "do", "i", "what", "is", "the", "can", "you", "me",
            "tell", "about", "please", "does", "are", "a", "an", "to",
            "in", "of", "for", "with", "my", "this", "that",
        }
        words = question.lower().replace("?", "").replace("!", "").split()
        keywords = [w for w in words if w not in filler_words]
        return " ".join(keywords) if keywords else question

    def search(self, query: str, num_results: int = None, scrape_pages: bool = True) -> WebSearchResponse:
        """
        Search docs.apimatic.io via Serper API, then scrape result pages for full content.

        Args:
            query: Search query (will be scoped with site: filter)
            num_results: Number of results to return
            scrape_pages: Whether to scrape full page content from result URLs

        Returns:
            WebSearchResponse with results (including scraped page content)
        """
        if not self._serper_available:
            return WebSearchResponse(
                query=query,
                success=False,
                error="Serper API key not configured",
            )

        num = num_results or SERPER_NUM_RESULTS
        scoped_query = f"site:{SERPER_SITE_FILTER} {query}"

        start_time = time.time()

        try:
            response = requests.post(
                SERPER_BASE_URL,
                json={"q": scoped_query, "num": num},
                headers={
                    "X-API-KEY": SERPER_API_KEY,
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            urls_to_scrape = []
            for idx, item in enumerate(data.get("organic", []), 1):
                url = item.get("link", "")
                results.append(WebSearchResult(
                    title=item.get("title", ""),
                    url=url,
                    snippet=item.get("snippet", ""),
                    position=idx,
                ))
                if url and scrape_pages:
                    urls_to_scrape.append(url)

            # Scrape actual page content in parallel
            if urls_to_scrape:
                logger.info(f"Scraping {len(urls_to_scrape)} pages for full content...")
                scraped = _scrape_pages_parallel(urls_to_scrape)
                for result in results:
                    content = scraped.get(result.url, "")
                    if content:
                        result.page_content = content
                        logger.debug(f"Scraped {len(content)} chars from {result.url}")

            latency = (time.time() - start_time) * 1000

            logger.info(
                f"Serper search: '{scoped_query}' → {len(results)} results in {latency:.0f}ms "
                f"({sum(1 for r in results if r.page_content)} pages scraped)"
            )

            return WebSearchResponse(
                query=scoped_query,
                results=results,
                latency_ms=latency,
                success=True,
            )

        except requests.exceptions.Timeout:
            latency = (time.time() - start_time) * 1000
            logger.warning("Serper API timeout")
            return WebSearchResponse(
                query=scoped_query,
                latency_ms=latency,
                success=False,
                error="Web search timed out",
            )

        except requests.exceptions.RequestException as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"Serper API error: {e}")
            return WebSearchResponse(
                query=scoped_query,
                latency_ms=latency,
                success=False,
                error=f"Web search failed: {str(e)}",
            )

    def search_for_question(self, user_question: str, num_results: int = None) -> WebSearchResponse:
        """
        End-to-end: generate optimized query from user question, then search + scrape.

        Args:
            user_question: Natural language question
            num_results: Number of results

        Returns:
            WebSearchResponse with results and scraped content
        """
        optimized_query = self.generate_search_query(user_question)
        return self.search(optimized_query, num_results)


# ─────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────
_serper_service: Optional[SerperSearchService] = None


def get_serper_service() -> SerperSearchService:
    """Get or create the singleton Serper search service."""
    global _serper_service
    if _serper_service is None:
        _serper_service = SerperSearchService()
    return _serper_service
