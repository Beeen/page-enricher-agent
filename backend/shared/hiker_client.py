"""HikerAPI client for Instagram account discovery."""

import os
from typing import Any, Dict, List, Optional

import httpx

HIKER_API_BASE_URL = "https://api.hikerapi.com"


def _get_hiker_headers() -> Dict[str, str]:
    """Build request headers with the HikerAPI access key."""
    api_key = os.getenv("HIKER_API_KEY")
    if not api_key:
        raise RuntimeError("HIKER_API_KEY environment variable is not set")
    return {
        "x-access-key": api_key,
        "accept": "application/json",
    }


def search_instagram_accounts(query: str) -> List[Dict[str, Any]]:
    """Search Instagram accounts by name/keyword via HikerAPI.

    Uses the ``/v2/fbsearch/accounts`` endpoint (fbsearch_accounts_v2).

    Args:
        query: Search term (e.g. entity name like "Mr. Scruff").

    Returns:
        List of account dicts with keys like ``pk``, ``username``,
        ``full_name``, ``biography``, ``profile_pic_url``,
        ``follower_count``, ``is_verified``.
        Returns an empty list when the API key is missing or the request fails.
    """
    try:
        headers = _get_hiker_headers()
    except RuntimeError:
        print("[HIKER] API key not configured, skipping Instagram search")
        return []

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                f"{HIKER_API_BASE_URL}/v2/fbsearch/accounts",
                params={"query": query},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            print(f"[HIKER] Search '{query}' — HTTP {resp.status_code}, response type: {type(data).__name__}")

            # The response may be a list directly or wrapped in a key
            if isinstance(data, list):
                results = data
            elif isinstance(data, dict):
                results = data.get("users") or data.get("results") or []
                print(f"[HIKER] Search '{query}' — dict keys: {list(data.keys())}")
            else:
                results = []

            usernames = [r.get("username") for r in results[:10]]
            print(f"[HIKER] Search '{query}' — {len(results)} result(s): {usernames}")
            return results
    except Exception as e:
        print(f"[HIKER] Instagram search failed for '{query}': {e}")
        return []


def get_instagram_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a detailed Instagram profile by numeric user ID.

    Uses the ``/v1/user/by/id`` endpoint.

    Args:
        user_id: Instagram numeric user ID (``pk``).

    Returns:
        Profile dict or ``None`` on failure.
    """
    try:
        headers = _get_hiker_headers()
    except RuntimeError:
        return None

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                f"{HIKER_API_BASE_URL}/v1/user/by/id",
                params={"id": user_id},
                headers=headers,
            )
            resp.raise_for_status()
            profile = resp.json()
            print(f"[HIKER] Profile fetched for user_id={user_id} — @{profile.get('username')}")
            return profile
    except Exception as e:
        print(f"[HIKER] Profile fetch failed for user_id={user_id}: {e}")
        return None


def get_instagram_profile_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Fetch a detailed Instagram profile by username.

    Uses the ``/v1/user/by/username`` endpoint.

    Args:
        username: Instagram username (with or without ``@`` prefix).

    Returns:
        Profile dict or ``None`` on failure (including if the username
        does not exist).
    """
    try:
        headers = _get_hiker_headers()
    except RuntimeError:
        return None

    clean_username = username.strip().lstrip("@").strip("/")

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                f"{HIKER_API_BASE_URL}/v1/user/by/username",
                params={"username": clean_username},
                headers=headers,
            )
            resp.raise_for_status()
            profile = resp.json()
            print(f"[HIKER] Profile fetched for username='{clean_username}' — @{profile.get('username')}")
            return profile
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"[HIKER] Username '{clean_username}' not found on Instagram")
        else:
            print(f"[HIKER] Profile fetch by username failed for '{clean_username}': {e}")
        return None
    except Exception as e:
        print(f"[HIKER] Profile fetch by username failed for '{clean_username}': {e}")
        return None
