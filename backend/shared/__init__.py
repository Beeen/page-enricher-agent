"""Shared utilities for the Signal backend."""

from .llm import llm, _init_llm
from .search import (
    _serp_search_raw,
    _compact,
    _extract_url_from_text,
    _extract_ig_usernames_from_text,
    _validate_url,
    SEARCH_TIMEOUT,
)
from .ra_client import parse_ra_url, fetch_ra_entity, normalize_ra_data
from .hiker_client import search_instagram_accounts, get_instagram_profile, get_instagram_profile_by_username
from .soundcloud_client import search_soundcloud_users, get_soundcloud_user_by_permalink
from .tracing import (
    _TRACING,
    using_prompt_template,
    using_metadata,
    using_attributes,
    trace,
    init_tracing,
)

__all__ = [
    "llm",
    "_init_llm",
    "_serp_search_raw",
    "_compact",
    "_extract_url_from_text",
    "_extract_ig_usernames_from_text",
    "_validate_url",
    "SEARCH_TIMEOUT",
    "_TRACING",
    "using_prompt_template",
    "using_metadata",
    "using_attributes",
    "trace",
    "init_tracing",
    "parse_ra_url",
    "fetch_ra_entity",
    "normalize_ra_data",
    "search_instagram_accounts",
    "get_instagram_profile",
    "get_instagram_profile_by_username",
    "search_soundcloud_users",
    "get_soundcloud_user_by_permalink",
]
