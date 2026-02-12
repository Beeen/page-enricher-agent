"""Shared utilities for the AI Trip Planner backend."""

from .llm import llm, _init_llm
from .search import (
    _search_api,
    _compact,
    _llm_fallback,
    _with_prefix,
    _extract_url_from_text,
    _validate_url,
    SEARCH_TIMEOUT,
)
from .rag import LocalGuideRetriever, GUIDE_RETRIEVER, ENABLE_RAG
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
    "_search_api",
    "_compact",
    "_llm_fallback",
    "_with_prefix",
    "_extract_url_from_text",
    "_validate_url",
    "SEARCH_TIMEOUT",
    "LocalGuideRetriever",
    "GUIDE_RETRIEVER",
    "ENABLE_RAG",
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
