"""Search API and text processing utilities."""

import os
import re
from typing import List, Optional
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
                    "num": 10,
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


_IG_NON_PROFILE_SEGMENTS = frozenset({
    "p", "reel", "reels", "explore", "stories", "tv",
    "accounts", "directory", "developer", "about", "legal",
    "emails", "press", "api", "privacy", "terms", "help",
})


def _extract_ig_usernames_from_text(text: str, max_results: int = 5) -> List[str]:
    """Extract valid Instagram profile usernames from text.

    Uses findall to get ALL matches, filters out non-profile path segments
    (reels, posts, explore, etc.), deduplicates, and returns up to max_results.
    """
    if not text:
        return []

    pattern = r"(?:https?://)?(?:www\.)?instagram\.com/([a-zA-Z0-9_\.]+)/?"
    matches = re.findall(pattern, text, re.IGNORECASE)

    seen: set = set()
    usernames: list = []
    for username in matches:
        lower = username.lower()
        if lower in seen or lower in _IG_NON_PROFILE_SEGMENTS:
            continue
        seen.add(lower)
        usernames.append(username)
        if len(usernames) >= max_results:
            break

    return usernames


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
