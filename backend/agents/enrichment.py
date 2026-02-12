"""RA Entity Enrichment Pipeline — multi-agent graph for enriching Resident Advisor entities.

Accepts an RA URL, fetches structured data via the RA GraphQL API, identifies
missing fields, dispatches parallel discovery agents (Instagram via HikerAPI,
SoundCloud via SoundCloud API), and assembles a complete enriched profile.
"""

import json
import operator
import re
from typing import Any, Dict, List, Optional

from typing_extensions import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from shared import (
    llm,
    _search_api,
    _compact,
    _extract_url_from_text,
    _validate_url,
    _TRACING,
    trace,
    using_prompt_template,
    using_attributes,
    parse_ra_url,
    fetch_ra_entity,
    normalize_ra_data,
    search_instagram_accounts,
    get_instagram_profile,
    get_instagram_profile_by_username,
    search_soundcloud_users,
    get_soundcloud_user_by_permalink,
)


# =============================================================================
# STATE SCHEMA
# =============================================================================


class EnrichmentState(TypedDict):
    """State for the RA enrichment agent graph."""

    messages: Annotated[List[BaseMessage], operator.add]
    enrich_request: Dict[str, Any]

    # URL parsing
    ra_entity_type: Optional[str]
    ra_identifier: Optional[str]

    # RA GraphQL data
    ra_data: Optional[Dict[str, Any]]
    entity_name: Optional[str]

    # Gap analysis
    missing_fields: Optional[List[str]]

    # RA-provided social links (passed to discovery for validation)
    ra_instagram_hint: Optional[str]
    ra_soundcloud_hint: Optional[str]

    # Discovery results
    instagram_result: Optional[Dict[str, Any]]
    soundcloud_result: Optional[Dict[str, Any]]

    # Resolution results
    profile_picture: Optional[Dict[str, Any]]
    bio: Optional[Dict[str, Any]]
    city: Optional[Dict[str, Any]]

    # Final output
    final_profile: Optional[Dict[str, Any]]

    # Observability
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]


# =============================================================================
# HELPERS
# =============================================================================


def _parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM output, stripping markdown code fences if present."""
    if not text:
        return None
    # Strip ```json ... ``` wrappers
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


# =============================================================================
# NODE 1: RA SCRAPER
# =============================================================================


def ra_scraper_node(state: EnrichmentState) -> dict:
    """Parse the RA URL and fetch entity data via the GraphQL API."""
    req = state["enrich_request"]
    url = req["url"]
    calls: List[Dict[str, Any]] = []

    # --- Parse URL ---
    try:
        parsed = parse_ra_url(url)
        entity_type = parsed["entity_type"]
        identifier = parsed["identifier"]
        calls.append({
            "agent": "ra_scraper", "tool": "parse_ra_url",
            "args": {"url": url}, "result": parsed,
        })
    except ValueError as e:
        calls.append({"agent": "ra_scraper", "tool": "parse_ra_url", "error": str(e)})
        return {
            "messages": [SystemMessage(content=f"URL parse error: {e}")],
            "ra_entity_type": None,
            "ra_identifier": None,
            "ra_data": None,
            "entity_name": None,
            "tool_calls": calls,
        }

    # --- Fetch from RA GraphQL API ---
    try:
        raw = fetch_ra_entity(entity_type, identifier)
        normalized = normalize_ra_data(entity_type, raw)
        calls.append({
            "agent": "ra_scraper", "tool": "fetch_ra_entity",
            "args": {"entity_type": entity_type, "identifier": identifier},
            "result_fields": list(k for k, v in normalized.items() if v is not None),
        })
    except Exception as e:
        print(f"[ENRICHMENT] RA fetch failed: {e}")
        calls.append({"agent": "ra_scraper", "tool": "fetch_ra_entity", "error": str(e)})
        # Build minimal fallback from URL
        fallback_name = identifier.replace("-", " ").replace("_", " ").title()
        normalized = {"name": fallback_name, "page_type": entity_type}

    entity_name = normalized.get("name") or identifier

    return {
        "messages": [SystemMessage(content=f"RA data fetched for '{entity_name}' ({entity_type})")],
        "ra_entity_type": entity_type,
        "ra_identifier": identifier,
        "ra_data": normalized,
        "entity_name": entity_name,
        "tool_calls": calls,
    }


# =============================================================================
# NODE 2: GAP ANALYSIS
# =============================================================================


def gap_analysis_node(state: EnrichmentState) -> dict:
    """Examine RA data and identify which enrichment fields are missing.

    Discovery nodes always run regardless of missing fields, but the
    missing_fields list is still tracked for resolution/assembly logic.
    """
    ra_data = state.get("ra_data") or {}
    missing: List[str] = []

    if not ra_data.get("instagram"):
        missing.append("instagram")
    if not ra_data.get("soundcloud"):
        missing.append("soundcloud")
    if not ra_data.get("profile_picture"):
        missing.append("profile_picture")
    if not ra_data.get("bio"):
        missing.append("bio")

    ra_instagram_hint = ra_data.get("instagram")
    ra_soundcloud_hint = ra_data.get("soundcloud")

    return {
        "messages": [SystemMessage(content=f"Missing fields: {missing}")],
        "missing_fields": missing,
        "ra_instagram_hint": ra_instagram_hint,
        "ra_soundcloud_hint": ra_soundcloud_hint,
        "tool_calls": [{
            "agent": "gap_analysis", "tool": "analyze_gaps",
            "result": {
                "missing": missing,
                "total_ra_fields": sum(1 for v in ra_data.values() if v is not None),
                "ra_instagram_hint": ra_instagram_hint,
                "ra_soundcloud_hint": ra_soundcloud_hint,
            },
        }],
    }


def route_discovery_agents(state: EnrichmentState) -> List[str]:
    """Always dispatch both discovery agents for cross-source validation."""
    return ["instagram_node", "soundcloud_node"]


# =============================================================================
# NODE 3a: INSTAGRAM DISCOVERY
# =============================================================================


def _build_ig_candidate_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact summary of an Instagram profile for LLM evaluation."""
    return {
        "username": profile.get("username"),
        "full_name": profile.get("full_name"),
        "biography": profile.get("biography") or "",
        "follower_count": profile.get("follower_count"),
        "following_count": profile.get("following_count"),
        "media_count": profile.get("media_count"),
        "is_verified": profile.get("is_verified", False),
        "is_business": profile.get("is_business", False),
        "category": profile.get("category"),
        "external_url": profile.get("external_url"),
    }


def _validate_ig_profile_against_ra(
    profile: Dict[str, Any], entity_name: str, ra_data: Dict[str, Any], calls: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Ask the LLM whether a single Instagram profile matches the RA entity.

    Returns the instagram_result dict if the profile matches (confidence >= 0.6),
    or None if it does not.
    """
    candidate_summary = _build_ig_candidate_summary(profile)
    ra_context = {
        "name": entity_name,
        "type": ra_data.get("page_type"),
        "bio": (ra_data.get("bio") or "")[:300],
        "country": ra_data.get("country"),
    }

    prompt = (
        "You are evaluating whether an Instagram profile belongs to an "
        "electronic music entity.\n\n"
        f"Entity from Resident Advisor:\n{json.dumps(ra_context, indent=2)}\n\n"
        f"Instagram profile:\n{json.dumps(candidate_summary, indent=2)}\n\n"
        "Return a JSON object with:\n"
        '- "is_match": boolean\n'
        '- "confidence": float 0.0 to 1.0\n'
        '- "reasoning": brief explanation\n\n'
        "Only confirm the match if you are reasonably confident this is the same entity."
    )

    with using_attributes(tags=["instagram_discovery", "enrichment"]):
        response = llm.invoke([
            SystemMessage(content="You are a data matching specialist for the electronic music industry. Respond with JSON only."),
            HumanMessage(content=prompt),
        ])

    calls.append({"agent": "instagram_discovery", "tool": "llm_validate_ra_link"})

    result = _parse_llm_json(response.content if response.content else "")
    is_match = (result or {}).get("is_match", False)
    confidence = (result or {}).get("confidence", 0.0)

    if is_match and confidence >= 0.6:
        username = profile.get("username") or ""
        return {
            "url": f"https://instagram.com/{username}",
            "profile_pic_url": profile.get("hd_profile_pic_url_info", {}).get("url")
                or profile.get("profile_pic_url"),
            "biography": profile.get("biography"),
            "follower_count": profile.get("follower_count"),
            "source": "hiker_api",
            "confidence": confidence,
        }
    return None


def instagram_discovery_node(state: EnrichmentState) -> dict:
    """Discover and validate an Instagram account for the entity.

    Branch A: If RA provides an Instagram link, fetch the profile and validate it.
    Branch B: Search by entity name, fetch full profiles for top 3, LLM picks best.
    """
    entity_name = state.get("entity_name") or "Unknown"
    ra_data = state.get("ra_data") or {}
    ra_instagram_hint = state.get("ra_instagram_hint")
    calls: List[Dict[str, Any]] = []

    # =================================================================
    # Branch A: RA has an Instagram link — validate it
    # =================================================================
    if ra_instagram_hint:
        ra_username = ra_instagram_hint.rstrip("/").split("/")[-1].lstrip("@")
        ra_profile = get_instagram_profile_by_username(ra_username)
        calls.append({
            "agent": "instagram_discovery", "tool": "hiker_profile_by_username",
            "args": {"username": ra_username},
            "result": "found" if ra_profile else "not_found",
        })

        if ra_profile:
            match = _validate_ig_profile_against_ra(ra_profile, entity_name, ra_data, calls)
            if match:
                return {
                    "messages": [SystemMessage(content=f"RA Instagram link validated: @{ra_username} (confidence={match['confidence']:.2f})")],
                    "instagram_result": match,
                    "tool_calls": calls,
                }
            # RA link exists on Instagram but doesn't match — fall through to search

    # =================================================================
    # Branch B: Search by entity name
    # =================================================================
    results = search_instagram_accounts(entity_name)
    calls.append({
        "agent": "instagram_discovery", "tool": "hiker_search",
        "args": {"query": entity_name}, "result_count": len(results),
    })

    if not results:
        # Fallback: web search
        fallback_query = f"{entity_name} {ra_data.get('page_type', 'artist')} instagram official"
        search_result = _search_api(fallback_query)
        calls.append({"agent": "instagram_discovery", "tool": "web_search_fallback", "args": {"query": fallback_query}})
        if search_result:
            url = _extract_url_from_text(search_result, "instagram")
            if url and _validate_url(url):
                return {
                    "messages": [SystemMessage(content=f"Instagram found via web search: {url}")],
                    "instagram_result": {"url": url, "source": "web_search", "confidence": 0.6},
                    "tool_calls": calls,
                }
        return {
            "messages": [SystemMessage(content="Instagram not found")],
            "instagram_result": None,
            "tool_calls": calls,
        }

    # --- Fetch full profiles for top 3 search results ---
    enriched_candidates = []
    for r in results[:3]:
        user_id = str(r.get("pk") or r.get("id") or "")
        profile = get_instagram_profile(user_id) if user_id else None
        calls.append({
            "agent": "instagram_discovery", "tool": "hiker_profile",
            "args": {"user_id": user_id},
            "result": "found" if profile else "not_found",
        })
        enriched_candidates.append(profile or r)

    # --- LLM cross-reference to pick best match ---
    llm_candidates = [_build_ig_candidate_summary(c) for c in enriched_candidates]

    ra_context = {
        "name": entity_name,
        "type": ra_data.get("page_type"),
        "bio": (ra_data.get("bio") or "")[:300],
        "country": ra_data.get("country"),
    }

    prompt = (
        "You are evaluating Instagram search results to find the correct account "
        "for an electronic music entity.\n\n"
        f"Entity from Resident Advisor:\n{json.dumps(ra_context, indent=2)}\n\n"
        f"Instagram candidates (with full profile data):\n{json.dumps(llm_candidates, indent=2)}\n\n"
        "Return a JSON object with:\n"
        '- "best_match_index": index (0-based) of the best matching account, or -1 if none match\n'
        '- "confidence": float 0.0 to 1.0\n'
        '- "reasoning": brief explanation\n\n'
        "Only match if you are reasonably confident this is the same entity."
    )

    with using_attributes(tags=["instagram_discovery", "enrichment"]):
        response = llm.invoke([
            SystemMessage(content="You are a data matching specialist for the electronic music industry. Respond with JSON only."),
            HumanMessage(content=prompt),
        ])

    calls.append({"agent": "instagram_discovery", "tool": "llm_cross_reference", "args": {"candidates": len(llm_candidates)}})

    match_result = _parse_llm_json(response.content if response.content else "")
    idx = (match_result or {}).get("best_match_index", -1)
    confidence = (match_result or {}).get("confidence", 0.0)

    if isinstance(idx, int) and 0 <= idx < len(enriched_candidates) and confidence >= 0.6:
        matched = enriched_candidates[idx]
        username = matched.get("username") or ""
        return {
            "messages": [SystemMessage(content=f"Instagram matched: @{username} (confidence={confidence:.2f})")],
            "instagram_result": {
                "url": f"https://instagram.com/{username}",
                "profile_pic_url": matched.get("hd_profile_pic_url_info", {}).get("url")
                    or matched.get("profile_pic_url"),
                "biography": matched.get("biography"),
                "follower_count": matched.get("follower_count"),
                "source": "hiker_api",
                "confidence": confidence,
            },
            "tool_calls": calls,
        }

    return {
        "messages": [SystemMessage(content=f"No confident Instagram match (best confidence={confidence:.2f})")],
        "instagram_result": None,
        "tool_calls": calls,
    }


# =============================================================================
# NODE 3b: SOUNDCLOUD DISCOVERY
# =============================================================================


def _build_sc_candidate_summary(user: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact summary of a SoundCloud user for LLM evaluation."""
    return {
        "permalink": user.get("permalink"),
        "username": user.get("username"),
        "full_name": user.get("full_name") or user.get("username"),
        "description": (user.get("description") or "")[:300],
        "followers_count": user.get("followers_count"),
        "track_count": user.get("track_count"),
        "city": user.get("city"),
        "country": user.get("country"),
    }


def _validate_sc_profile_against_ra(
    profile: Dict[str, Any], entity_name: str, ra_data: Dict[str, Any], calls: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Ask the LLM whether a single SoundCloud profile matches the RA entity.

    Returns the soundcloud_result dict if the profile matches (confidence >= 0.6),
    or None if it does not.
    """
    candidate_summary = _build_sc_candidate_summary(profile)
    ra_context = {
        "name": entity_name,
        "type": ra_data.get("page_type"),
        "bio": (ra_data.get("bio") or "")[:300],
        "country": ra_data.get("country"),
    }

    prompt = (
        "You are evaluating whether a SoundCloud profile belongs to an "
        "electronic music entity.\n\n"
        f"Entity from Resident Advisor:\n{json.dumps(ra_context, indent=2)}\n\n"
        f"SoundCloud profile:\n{json.dumps(candidate_summary, indent=2)}\n\n"
        "Return a JSON object with:\n"
        '- "is_match": boolean\n'
        '- "confidence": float 0.0 to 1.0\n'
        '- "reasoning": brief explanation\n\n'
        "Only confirm the match if you are reasonably confident this is the same entity."
    )

    with using_attributes(tags=["soundcloud_discovery", "enrichment"]):
        response = llm.invoke([
            SystemMessage(content="You are a data matching specialist for the electronic music industry. Respond with JSON only."),
            HumanMessage(content=prompt),
        ])

    calls.append({"agent": "soundcloud_discovery", "tool": "llm_validate_ra_link"})

    result = _parse_llm_json(response.content if response.content else "")
    is_match = (result or {}).get("is_match", False)
    confidence = (result or {}).get("confidence", 0.0)

    if is_match and confidence >= 0.6:
        permalink = profile.get("permalink") or profile.get("username") or ""
        avatar = profile.get("avatar_url")
        if avatar:
            avatar = avatar.replace("-large.", "-t500x500.")
        return {
            "url": f"https://soundcloud.com/{permalink}",
            "avatar_url": avatar,
            "description": profile.get("description"),
            "city": profile.get("city"),
            "country": profile.get("country"),
            "source": "soundcloud_api",
            "confidence": confidence,
        }
    return None


def soundcloud_discovery_node(state: EnrichmentState) -> dict:
    """Discover and validate a SoundCloud account for the entity.

    Branch A: If RA provides a SoundCloud link, fetch the profile and validate it.
    Branch B: Search by entity name, use full user data for top 3, LLM picks best.
    """
    entity_name = state.get("entity_name") or "Unknown"
    ra_data = state.get("ra_data") or {}
    ra_soundcloud_hint = state.get("ra_soundcloud_hint")
    calls: List[Dict[str, Any]] = []

    # =================================================================
    # Branch A: RA has a SoundCloud link — validate it
    # =================================================================
    if ra_soundcloud_hint:
        ra_profile = get_soundcloud_user_by_permalink(ra_soundcloud_hint)
        calls.append({
            "agent": "soundcloud_discovery", "tool": "soundcloud_get_by_permalink",
            "args": {"permalink": ra_soundcloud_hint},
            "result": "found" if ra_profile else "not_found",
        })

        if ra_profile:
            match = _validate_sc_profile_against_ra(ra_profile, entity_name, ra_data, calls)
            if match:
                return {
                    "messages": [SystemMessage(content=f"RA SoundCloud link validated: {match['url']} (confidence={match['confidence']:.2f})")],
                    "soundcloud_result": match,
                    "tool_calls": calls,
                }
            # RA link exists on SoundCloud but doesn't match — fall through to search

    # =================================================================
    # Branch B: Search by entity name
    # =================================================================
    results = search_soundcloud_users(entity_name)
    calls.append({
        "agent": "soundcloud_discovery", "tool": "soundcloud_search",
        "args": {"query": entity_name}, "result_count": len(results),
    })

    if not results:
        # Fallback: web search
        fallback_query = f"{entity_name} soundcloud official"
        search_result = _search_api(fallback_query)
        calls.append({"agent": "soundcloud_discovery", "tool": "web_search_fallback", "args": {"query": fallback_query}})
        if search_result:
            url = _extract_url_from_text(search_result, "soundcloud")
            if url and _validate_url(url):
                return {
                    "messages": [SystemMessage(content=f"SoundCloud found via web search: {url}")],
                    "soundcloud_result": {"url": url, "source": "web_search", "confidence": 0.6},
                    "tool_calls": calls,
                }
        return {
            "messages": [SystemMessage(content="SoundCloud not found")],
            "soundcloud_result": None,
            "tool_calls": calls,
        }

    # --- LLM cross-reference top 3 results ---
    top_results = results[:3]
    llm_candidates = [_build_sc_candidate_summary(r) for r in top_results]

    ra_context = {
        "name": entity_name,
        "type": ra_data.get("page_type"),
        "bio": (ra_data.get("bio") or "")[:300],
        "country": ra_data.get("country"),
    }

    prompt = (
        "You are evaluating SoundCloud search results to find the correct account "
        "for an electronic music entity.\n\n"
        f"Entity from Resident Advisor:\n{json.dumps(ra_context, indent=2)}\n\n"
        f"SoundCloud candidates:\n{json.dumps(llm_candidates, indent=2)}\n\n"
        "Return a JSON object with:\n"
        '- "best_match_index": index (0-based) of the best matching account, or -1 if none match\n'
        '- "confidence": float 0.0 to 1.0\n'
        '- "reasoning": brief explanation\n\n'
        "Only match if you are reasonably confident this is the same entity."
    )

    with using_attributes(tags=["soundcloud_discovery", "enrichment"]):
        response = llm.invoke([
            SystemMessage(content="You are a data matching specialist for the electronic music industry. Respond with JSON only."),
            HumanMessage(content=prompt),
        ])

    calls.append({"agent": "soundcloud_discovery", "tool": "llm_cross_reference", "args": {"candidates": len(llm_candidates)}})

    match_result = _parse_llm_json(response.content if response.content else "")
    idx = (match_result or {}).get("best_match_index", -1)
    confidence = (match_result or {}).get("confidence", 0.0)

    if isinstance(idx, int) and 0 <= idx < len(top_results) and confidence >= 0.6:
        matched = top_results[idx]
        permalink = matched.get("permalink") or matched.get("username") or ""
        avatar = matched.get("avatar_url")
        if avatar:
            avatar = avatar.replace("-large.", "-t500x500.")
        return {
            "messages": [SystemMessage(content=f"SoundCloud matched: {permalink} (confidence={confidence:.2f})")],
            "soundcloud_result": {
                "url": f"https://soundcloud.com/{permalink}",
                "avatar_url": avatar,
                "description": matched.get("description"),
                "city": matched.get("city"),
                "country": matched.get("country"),
                "source": "soundcloud_api",
                "confidence": confidence,
            },
            "tool_calls": calls,
        }

    return {
        "messages": [SystemMessage(content=f"No confident SoundCloud match (best confidence={confidence:.2f})")],
        "soundcloud_result": None,
        "tool_calls": calls,
    }


# =============================================================================
# NODE 4: RESOLUTION (profile picture + bio priority cascade)
# =============================================================================


def resolution_node(state: EnrichmentState) -> dict:
    """Resolve profile picture and bio using priority-based cascades."""
    ra_data = state.get("ra_data") or {}
    ig = state.get("instagram_result") or {}
    sc = state.get("soundcloud_result") or {}
    entity_name = state.get("entity_name") or "Unknown"
    calls: List[Dict[str, Any]] = []

    # --- Profile picture: RA > Instagram > SoundCloud > Google Search ---
    profile_picture: Optional[Dict[str, Any]] = None

    if ra_data.get("profile_picture"):
        profile_picture = {
            "value": ra_data["profile_picture"],
            "source": "ra",
            "confidence": 0.9,
        }
    elif ig.get("profile_pic_url"):
        profile_picture = {
            "value": ig["profile_pic_url"],
            "source": "instagram",
            "confidence": ig.get("confidence", 0.8),
        }
    elif sc.get("avatar_url"):
        # Upgrade SoundCloud avatar to high-res if possible
        avatar = sc["avatar_url"]
        if avatar:
            avatar = avatar.replace("-large.", "-t500x500.")
        profile_picture = {
            "value": avatar,
            "source": "soundcloud",
            "confidence": sc.get("confidence", 0.7),
        }
    else:
        # Google Search fallback
        search_query = f"{entity_name} {ra_data.get('page_type', 'artist')} profile photo"
        search_result = _search_api(search_query)
        calls.append({"agent": "resolution", "tool": "google_image_fallback", "args": {"query": search_query}})
        if search_result:
            profile_picture = {
                "value": search_result,
                "source": "google_search",
                "confidence": 0.4,
            }

    # --- Bio: Instagram > SoundCloud > RA > LLM fallback ---
    bio: Optional[Dict[str, Any]] = None

    if ig.get("biography") and len(ig["biography"]) > 20:
        bio = {
            "value": ig["biography"],
            "source": "instagram",
            "confidence": ig.get("confidence", 0.8),
        }
    elif sc.get("description") and len(sc["description"]) > 20:
        bio = {
            "value": sc["description"],
            "source": "soundcloud",
            "confidence": sc.get("confidence", 0.7),
        }
    elif ra_data.get("bio") and len(ra_data["bio"]) > 20:
        bio = {
            "value": ra_data["bio"],
            "source": "ra",
            "confidence": 0.95,
        }
    else:
        # LLM fallback
        entity_type = ra_data.get("page_type", "artist")
        with using_attributes(tags=["resolution", "bio_generation"]):
            response = llm.invoke([
                SystemMessage(content="You are an electronic music journalist. Write a brief, factual biography."),
                HumanMessage(content=f"Write a 2-3 sentence biography for electronic music {entity_type} '{entity_name}'."),
            ])
        calls.append({"agent": "resolution", "tool": "llm_bio_generation"})
        if response.content:
            bio = {
                "value": _compact(response.content, limit=500),
                "source": "llm_generated",
                "confidence": 0.5,
            }

    # --- City: RA > SoundCloud ---
    city: Optional[Dict[str, Any]] = None

    if ra_data.get("city"):
        city = {
            "value": ra_data["city"],
            "source": "ra",
            "confidence": 0.95,
        }
    elif sc.get("city"):
        city = {
            "value": sc["city"],
            "source": "soundcloud",
            "confidence": sc.get("confidence", 0.7),
        }

    calls.append({
        "agent": "resolution", "tool": "priority_cascade",
        "result": {
            "pic_source": (profile_picture or {}).get("source"),
            "bio_source": (bio or {}).get("source"),
            "city_source": (city or {}).get("source"),
        },
    })

    return {
        "messages": [SystemMessage(content="Resolution complete")],
        "profile_picture": profile_picture,
        "bio": bio,
        "city": city,
        "tool_calls": calls,
    }


# =============================================================================
# NODE 5: ASSEMBLY
# =============================================================================


def assembly_node(state: EnrichmentState) -> dict:
    """Assemble the final enriched profile from all sources."""
    ra_data = state.get("ra_data") or {}
    ig = state.get("instagram_result") or {}
    sc = state.get("soundcloud_result") or {}
    profile_picture = state.get("profile_picture")
    bio = state.get("bio")
    calls: List[Dict[str, Any]] = []

    final_profile: Dict[str, Any] = {}

    # --- RA base fields (confidence 1.0 — structured data) ---
    ra_base_fields = [
        "name", "page_type", "address", "latitude", "longitude",
        "website", "facebook", "twitter", "bandcamp", "discogs",
        "country", "resident_country", "city", "capacity",
        "follower_count", "content_url", "cover_picture",
    ]
    for field in ra_base_fields:
        val = ra_data.get(field)
        if val is not None:
            final_profile[field] = {"value": val, "source": "ra", "confidence": 1.0}

    # --- Enriched: profile_picture ---
    if profile_picture:
        final_profile["profile_picture"] = profile_picture

    # --- Enriched: biography ---
    if bio:
        final_profile["biography"] = bio

    # --- Enriched: city (override RA base field with resolved value) ---
    city_data = state.get("city")
    if city_data:
        final_profile["city"] = city_data

    # --- Enriched: Instagram ---
    if ig.get("url"):
        # Discovery found a match — use it
        final_profile["instagram"] = {
            "value": ig["url"],
            "source": ig.get("source", "hiker_api"),
            "confidence": ig.get("confidence", 0.8),
        }
    elif ra_data.get("instagram"):
        # Discovery didn't find a match but RA has a link — use at reduced confidence
        final_profile["instagram"] = {
            "value": ra_data["instagram"],
            "source": "ra",
            "confidence": 0.85,
        }

    # --- Enriched: SoundCloud ---
    if sc.get("url"):
        # Discovery found a match — use it
        final_profile["soundcloud"] = {
            "value": sc["url"],
            "source": sc.get("source", "soundcloud_api"),
            "confidence": sc.get("confidence", 0.8),
        }
    elif ra_data.get("soundcloud"):
        # Discovery didn't find a match but RA has a link — use at reduced confidence
        final_profile["soundcloud"] = {
            "value": ra_data["soundcloud"],
            "source": "ra",
            "confidence": 0.85,
        }

    # --- LLM consistency review ---
    # Build a compact summary for the LLM to review
    review_summary = {
        k: {"value": str(v.get("value", ""))[:100], "source": v.get("source"), "confidence": v.get("confidence")}
        for k, v in final_profile.items()
        if isinstance(v, dict)
    }

    review_prompt = (
        f"Review this enriched profile for '{state.get('entity_name')}' "
        f"({ra_data.get('page_type', 'entity')}) and check for consistency.\n\n"
        f"Profile:\n{json.dumps(review_summary, indent=2)}\n\n"
        "If any fields look suspicious or contradictory, return a JSON object with:\n"
        '- "adjustments": list of {{"field": str, "new_confidence": float, "reason": str}}\n'
        '- "review_notes": brief summary\n\n'
        "If everything looks consistent, return: {\"adjustments\": [], \"review_notes\": \"All fields consistent\"}"
    )

    with using_attributes(tags=["assembly", "enrichment"]):
        response = llm.invoke([
            SystemMessage(content="You are a data quality specialist for electronic music entities. Respond with JSON only."),
            HumanMessage(content=review_prompt),
        ])

    calls.append({"agent": "assembly", "tool": "llm_consistency_review"})

    review = _parse_llm_json(response.content if response.content else "")
    if review and review.get("adjustments"):
        for adj in review["adjustments"]:
            field = adj.get("field")
            new_conf = adj.get("new_confidence")
            if field and field in final_profile and isinstance(new_conf, (int, float)):
                final_profile[field]["confidence"] = new_conf
                print(f"[ENRICHMENT] Confidence adjusted: {field} -> {new_conf} ({adj.get('reason')})")

    calls.append({
        "agent": "assembly", "tool": "build_profile",
        "result": {"field_count": len(final_profile), "review_notes": (review or {}).get("review_notes")},
    })

    return {
        "messages": [SystemMessage(content=f"Assembly complete: {len(final_profile)} fields")],
        "final_profile": final_profile,
        "tool_calls": calls,
    }


# =============================================================================
# GRAPH BUILDER
# =============================================================================


def build_enrichment_graph():
    """Build the LangGraph workflow for RA entity enrichment."""
    g = StateGraph(EnrichmentState)

    g.add_node("ra_scraper_node", ra_scraper_node)
    g.add_node("gap_analysis_node", gap_analysis_node)
    g.add_node("instagram_node", instagram_discovery_node)
    g.add_node("soundcloud_node", soundcloud_discovery_node)
    g.add_node("resolution_node", resolution_node)
    g.add_node("assembly_node", assembly_node)

    # Sequential: START -> RA scrape -> gap analysis
    g.add_edge(START, "ra_scraper_node")
    g.add_edge("ra_scraper_node", "gap_analysis_node")

    # Conditional parallel: gap analysis routes to discovery agents or straight to resolution
    g.add_conditional_edges(
        "gap_analysis_node",
        route_discovery_agents,
        {
            "instagram_node": "instagram_node",
            "soundcloud_node": "soundcloud_node",
            "resolution_node": "resolution_node",
        },
    )

    # Discovery agents converge to resolution
    g.add_edge("instagram_node", "resolution_node")
    g.add_edge("soundcloud_node", "resolution_node")

    # Resolution -> Assembly -> END
    g.add_edge("resolution_node", "assembly_node")
    g.add_edge("assembly_node", END)

    return g.compile()
