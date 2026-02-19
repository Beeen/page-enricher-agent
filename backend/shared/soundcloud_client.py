"""SoundCloud API client for account discovery via OAuth client_credentials flow."""

import os
import time
from base64 import b64encode
from typing import Any, Dict, List, Optional

import httpx

SOUNDCLOUD_API_URL = "https://api.soundcloud.com"
SOUNDCLOUD_TOKEN_URL = "https://secure.soundcloud.com/oauth/token"

# In-memory token cache
_token_cache: Dict[str, Any] = {}


def _get_soundcloud_token() -> Optional[str]:
    """Obtain a SoundCloud OAuth access token using client_credentials flow.

    Tokens are cached in memory and reused until they expire (with a 60-second
    safety margin).

    Returns:
        Access token string, or ``None`` if credentials are missing or the
        request fails.
    """
    # Check cache
    if _token_cache.get("access_token") and _token_cache.get("expires_at", 0) > time.time():
        return _token_cache["access_token"]

    client_id = os.getenv("SOUNDCLOUD_CLIENT_ID")
    client_secret = os.getenv("SOUNDCLOUD_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("[SOUNDCLOUD] Client ID or Secret not configured, skipping")
        return None

    try:
        credentials = b64encode(f"{client_id}:{client_secret}".encode()).decode()
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                SOUNDCLOUD_TOKEN_URL,
                data={"grant_type": "client_credentials"},
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Authorization": f"Basic {credentials}",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        token = data.get("access_token")
        expires_in = data.get("expires_in", 3600)

        if token:
            _token_cache["access_token"] = token
            # Expire 60 seconds early to avoid edge cases
            _token_cache["expires_at"] = time.time() + expires_in - 60
            print(f"[SOUNDCLOUD] Token obtained, expires in {expires_in}s")

        return token
    except Exception as e:
        print(f"[SOUNDCLOUD] Token request failed: {e}")
        return None


def search_soundcloud_users(query: str) -> List[Dict[str, Any]]:
    """Search SoundCloud users by name/keyword.

    Uses ``GET /users?q=<query>&limit=5`` with OAuth Bearer authentication.

    Args:
        query: Search term (e.g. entity name).

    Returns:
        List of user dicts with keys like ``id``, ``permalink``, ``username``,
        ``full_name``, ``description``, ``avatar_url``, ``followers_count``.
        Returns an empty list when credentials are missing or the request fails.
    """
    token = _get_soundcloud_token()
    if not token:
        return []

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                f"{SOUNDCLOUD_API_URL}/users",
                params={"q": query, "limit": 5},
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            data = resp.json()

            # Response is typically a list; handle wrapped formats too
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("collection") or data.get("users") or []
            return []
    except Exception as e:
        print(f"[SOUNDCLOUD] User search failed for '{query}': {e}")
        return []


def get_soundcloud_user_by_permalink(permalink_or_url: str) -> Optional[Dict[str, Any]]:
    """Fetch a SoundCloud user profile by permalink or full URL.

    Uses ``GET /users/{permalink}`` with OAuth Bearer authentication.

    Args:
        permalink_or_url: Either a permalink slug (e.g. ``"dj-name"``) or a full
            SoundCloud URL (e.g. ``"https://soundcloud.com/dj-name"``).

    Returns:
        User dict or ``None`` on failure (including if the user does not exist).
    """
    token = _get_soundcloud_token()
    if not token:
        return None

    # Extract permalink from full URL if needed
    permalink = permalink_or_url.strip().rstrip("/")
    if "soundcloud.com" in permalink:
        permalink = permalink.split("soundcloud.com/")[-1].split("/")[0]

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(
                f"{SOUNDCLOUD_API_URL}/resolve",
                params={"url": f"https://soundcloud.com/{permalink}"},
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"[SOUNDCLOUD] Permalink '{permalink}' not found")
        else:
            print(f"[SOUNDCLOUD] User fetch by permalink failed for '{permalink}': {e}")
        return None
    except Exception as e:
        print(f"[SOUNDCLOUD] User fetch by permalink failed for '{permalink}': {e}")
        return None
