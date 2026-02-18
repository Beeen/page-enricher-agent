"""Search API and text processing utilities."""

import os
import re
from typing import Optional
from urllib.parse import urlparse

import httpx

try:
    import validators
except ImportError:
    validators = None

SEARCH_TIMEOUT = 10.0


def _compact(text: str, limit: int = 200) -> str:
    """Compact text to a maximum length, truncating at word boundaries."""
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    truncated = cleaned[:limit]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip(",.;- ")


def _search_api(query: str) -> Optional[str]:
    """Search the web using Tavily if configured, return None otherwise."""
    query = query.strip()
    if not query:
        return None

    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
                resp = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": query,
                        "max_results": 3,
                        "search_depth": "basic",
                        "include_answer": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer") or ""
                snippets = [
                    item.get("content") or item.get("snippet") or ""
                    for item in data.get("results", [])
                ]
                combined = " ".join([answer] + snippets).strip()
                if combined:
                    return _compact(combined)
        except Exception:
            pass

    return None



def _serp_search_raw(query: str) -> Optional[str]:
    """Search Google via SerpAPI and return full untruncated text.

    Requires the ``SERPAPI_API_KEY`` environment variable.

    Returns:
        Combined snippet text from organic results, or None on failure.
    """
    query = query.strip()
    if not query:
        return None

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        print("[SEARCH] SERPAPI_API_KEY not configured, skipping SerpAPI search")
        return None

    try:
        with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
            resp = client.get(
                "https://serpapi.com/search",
                params={
                    "engine": "google",
                    "api_key": api_key,
                    "q": query,
                    "num": 5,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("organic_results", [])
            print(f"[SEARCH] SerpAPI '{query[:80]}' — {len(items)} result(s)")
            for i, item in enumerate(items):
                print(f"[SEARCH]   [{i}] {item.get('title', '?')} — {item.get('link', '?')}")
            links = [item.get("link", "") for item in items]
            snippets = [item.get("snippet", "") for item in items]
            combined = " ".join(links + snippets).strip()
            print(f"[SEARCH] SerpAPI combined snippet length: {len(combined)}")
            if combined:
                return combined
    except Exception as e:
        print(f"[SEARCH] SerpAPI search failed for '{query[:80]}': {e}")

    return None


def _extract_url_from_text(text: str, platform: str) -> Optional[str]:
    """Extract a platform URL from text using regex patterns."""
    patterns = {
        "instagram": r"(?:https?://)?(?:www\.)?instagram\.com/([a-zA-Z0-9_\.]+)/?",
        "soundcloud": r"(?:https?://)?(?:www\.)?soundcloud\.com/([a-zA-Z0-9_-]+)/?",
        "spotify": r"(?:https?://)?open\.spotify\.com/artist/([a-zA-Z0-9]+)",
        "bandcamp": r"(?:https?://)?([a-zA-Z0-9-]+)\.bandcamp\.com/?",
        "website": r"(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})/?",
    }
    pattern = patterns.get(platform)
    if not pattern:
        return None
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        if platform == "instagram":
            return f"https://instagram.com/{match.group(1)}"
        elif platform == "soundcloud":
            return f"https://soundcloud.com/{match.group(1)}"
        elif platform == "spotify":
            return f"https://open.spotify.com/artist/{match.group(1)}"
        elif platform == "bandcamp":
            return f"https://{match.group(1)}.bandcamp.com"
        elif platform == "website":
            return f"https://{match.group(1)}"
    return None


def _validate_url(url: str) -> bool:
    """Check if a URL is valid and accessible."""
    if not url:
        return False
    if validators:
        return validators.url(url) is True
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False
