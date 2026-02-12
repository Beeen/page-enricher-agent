"""AI Trip Planner API - FastAPI interface for trip planning and entity enrichment."""

import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from shared import (
    _validate_url,
    _TRACING,
    trace,
    using_attributes,
    init_tracing,
)
from agents import build_graph, build_enrichment_graph


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class TripRequest(BaseModel):
    destination: str
    duration: str
    budget: Optional[str] = None
    interests: Optional[str] = None
    travel_style: Optional[str] = None
    user_input: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    turn_index: Optional[int] = None


class TripResponse(BaseModel):
    result: str
    tool_calls: List[Dict[str, Any]] = []


class EnrichRequest(BaseModel):
    """Request model for RA entity enrichment."""

    url: str = Field(..., description="Resident Advisor URL (e.g. https://ra.co/dj/mrscruff)")
    force_refresh: bool = Field(False, description="Ignore cached data")


class EnrichedField(BaseModel):
    """A single enriched field with source and confidence."""

    value: Optional[str] = None
    source: str = "unknown"
    confidence: float = 0.0


class EnrichResponse(BaseModel):
    """Response model for enriched entity data."""

    entity_type: str
    entity_name: str
    source_url: str
    enriched_fields: Dict[str, EnrichedField] = {}
    missing_fields: List[str] = []
    processing_time_ms: int = 0
    agent_calls: List[Dict[str, Any]] = []


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(title="AI Trip Planner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize tracing at startup
init_tracing()


@app.get("/")
def serve_frontend():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "frontend", "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "frontend/index.html not found"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "ai-trip-planner"}


@app.post("/plan-trip", response_model=TripResponse)
def plan_trip(req: TripRequest):
    graph = build_graph()

    state = {
        "messages": [],
        "trip_request": req.model_dump(),
        "tool_calls": [],
    }

    session_id = req.session_id
    user_id = req.user_id
    turn_idx = req.turn_index

    attrs_kwargs = {}
    if session_id:
        attrs_kwargs["session_id"] = session_id
    if user_id:
        attrs_kwargs["user_id"] = user_id

    if turn_idx is not None and _TRACING:
        with using_attributes(**attrs_kwargs):
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("turn_index", turn_idx)
            out = graph.invoke(state)
    else:
        with using_attributes(**attrs_kwargs):
            out = graph.invoke(state)

    return TripResponse(result=out.get("final", ""), tool_calls=out.get("tool_calls", []))


@app.post("/enrich", response_model=EnrichResponse)
def enrich_entity(req: EnrichRequest):
    """Enrich an RA entity (artist, venue, or promoter) with missing data fields."""
    start_time = time.time()

    # Validate URL format
    if not _validate_url(req.url):
        raise HTTPException(status_code=400, detail="Invalid URL provided")

    # Validate RA domain
    from urllib.parse import urlparse as _urlparse

    parsed_url = _urlparse(req.url)
    if parsed_url.netloc not in ("ra.co", "www.ra.co"):
        raise HTTPException(
            status_code=400,
            detail="URL must be a Resident Advisor URL (ra.co). "
            "Examples: https://ra.co/dj/mrscruff, https://ra.co/clubs/2587",
        )

    graph = build_enrichment_graph()

    state = {
        "messages": [],
        "enrich_request": req.model_dump(),
        "ra_entity_type": None,
        "ra_identifier": None,
        "ra_data": None,
        "entity_name": None,
        "missing_fields": None,
        "ra_instagram_hint": None,
        "ra_soundcloud_hint": None,
        "instagram_result": None,
        "soundcloud_result": None,
        "profile_picture": None,
        "bio": None,
        "city": None,
        "final_profile": None,
        "tool_calls": [],
    }

    with using_attributes(tags=["enrichment", "ra_entity"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.source_url", req.url)
        out = graph.invoke(state)

    final_profile = out.get("final_profile") or {}
    entity_name = out.get("entity_name") or "Unknown"
    entity_type = out.get("ra_entity_type") or "unknown"

    # Separate RA base fields from enriched fields
    enriched_field_names = ["biography", "profile_picture", "instagram", "soundcloud", "city"]
    enriched_fields = {}
    missing_fields = []

    for field_name, field_data in final_profile.items():
        if isinstance(field_data, dict) and field_data.get("value") is not None:
            enriched_fields[field_name] = EnrichedField(
                value=str(field_data["value"]),
                source=field_data.get("source", "unknown"),
                confidence=field_data.get("confidence", 0.0),
            )

    # Check which enrichment targets are still missing
    for field in enriched_field_names:
        if field not in enriched_fields:
            missing_fields.append(field)

    processing_time_ms = int((time.time() - start_time) * 1000)

    return EnrichResponse(
        entity_type=entity_type,
        entity_name=entity_name,
        source_url=req.url,
        enriched_fields=enriched_fields,
        missing_fields=missing_fields,
        processing_time_ms=processing_time_ms,
        agent_calls=out.get("tool_calls", []),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
