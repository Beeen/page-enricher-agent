# Electronic Music Data Enrichment Agent - Product Requirements Document

## Overview

A multi-agent AI system that enriches electronic music entity profiles by automatically discovering missing information from web sources. Given a URL to an artist, venue, promoter, or label page, the agent identifies missing data fields and retrieves them through web scraping and intelligent search.

**Primary Use Case**: Automate data enrichment for electronic music databases (e.g., Resident Advisor, Beatport, event platforms)  
**Target Users**: Music platform operators, event promoters, booking agencies, music data aggregators

---

## Problem Statement

Electronic music databases often have incomplete entity profiles. Manual data entry is:
- Time-consuming (5-15 min per entity)
- Error-prone and inconsistent
- Difficult to scale across thousands of entities

The enrichment agent reduces this to seconds per entity while maintaining data quality.

---

## Supported Entity Types

| Entity Type | Description | Example Sources |
|-------------|-------------|-----------------|
| **Artist** | DJ, producer, live act | Resident Advisor, Beatport, Discogs |
| **Venue** | Club, festival ground, event space | RA, Google Maps, venue websites |
| **Promoter** | Event organizer, collective | RA, Facebook, Eventbrite |
| **Label** | Record label, imprint | Beatport, Discogs, Bandcamp |

---

## Data Fields to Enrich

| Field | Artist | Venue | Promoter | Label | Source Priority |
|-------|--------|-------|----------|-------|-----------------|
| Profile Picture | ✅ | ✅ | ✅ | ✅ | Source page → Instagram → Facebook |
| Biography | ✅ | ✅ | ✅ | ✅ | Source page → Wikipedia → Press kit |
| Instagram Link | ✅ | ✅ | ✅ | ✅ | Source page → Web search |
| SoundCloud Link | ✅ | ✅ | ✅ | ✅ | Source page → Web search |
| Spotify Link | ✅ | ❌ | ❌ | ✅ | Source page → Spotify search |
| Website | ✅ | ✅ | ✅ | ✅ | Source page → Web search |
| Google Maps Link | ❌ | ✅ | ❌ | ❌ | Google Places API → Address search |

> **Note**: Venues often post live recordings from events on SoundCloud. Promoters frequently maintain podcast series or mix compilations showcasing their events.

---

## System Architecture

Adapted from the ai-trip-planner multi-agent pattern with parallel execution:

```
User Input (URL) → FastAPI → LangGraph Workflow
                      │
              ┌───────┴───────┐
              │               │
         Scraper Agent    Entity Classifier
              │               │
              └───────┬───────┘
                      │
          ┌───────────┼───────────┐
          │           │           │
    Social Agent  Media Agent  Location Agent
    (Instagram,   (SoundCloud, (Google Maps,
     Website)      Spotify)     Address)
          │           │           │
          └───────────┼───────────┘
                      │
              Enrichment Agent
              (Synthesize & Validate)
                      │
              Enriched Profile JSON
```

### Agent Responsibilities

| Agent | Purpose | Tools |
|-------|---------|-------|
| **Scraper Agent** | Extract existing data from input URL | `scrape_page`, `extract_metadata` |
| **Entity Classifier** | Detect entity type from page content | `classify_entity` (LLM-based) |
| **Social Agent** | Find social media & website links | `search_instagram`, `search_website`, `validate_url` |
| **Media Agent** | Discover streaming platform links | `search_soundcloud`, `search_spotify`, `search_bandcamp` |
| **Location Agent** | Find venue addresses & map links | `search_google_places`, `geocode_address` |
| **Enrichment Agent** | Synthesize, dedupe, validate all findings | `merge_results`, `validate_profile` |

---

## API Design

### Endpoint

```
POST /enrich
```

### Request Body

```json
{
  "url": "https://ra.co/dj/amelielens",
  "entity_type": "artist",           // Optional: auto-detected if omitted
  "priority_fields": ["instagram", "spotify"],  // Optional: focus enrichment
  "force_refresh": false             // Optional: ignore cached data
}
```

### Response

```json
{
  "entity_type": "artist",
  "entity_name": "Amelie Lens",
  "source_url": "https://ra.co/dj/amelielens",
  "enriched_fields": {
    "profile_picture": {
      "url": "https://example.com/photo.jpg",
      "source": "instagram",
      "confidence": 0.95
    },
    "biography": {
      "text": "Belgian DJ and producer known for...",
      "source": "resident_advisor",
      "confidence": 0.99
    },
    "instagram": {
      "url": "https://instagram.com/amelikiethus",
      "source": "web_search",
      "confidence": 0.92
    },
    "soundcloud": {
      "url": "https://soundcloud.com/amelielens",
      "source": "source_page",
      "confidence": 1.0
    },
    "spotify": {
      "url": "https://open.spotify.com/artist/...",
      "source": "spotify_api",
      "confidence": 0.98
    },
    "website": {
      "url": "https://amelielens.com",
      "source": "source_page",
      "confidence": 1.0
    }
  },
  "missing_fields": [],
  "processing_time_ms": 3200,
  "agent_calls": [...]
}
```

---

## State Management

```python
class EnrichmentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    enrich_request: Dict[str, Any]       # Original request
    scraped_data: Optional[Dict]          # Raw page extraction
    entity_type: Optional[str]            # Detected entity type
    entity_name: Optional[str]            # Detected entity name
    social_results: Optional[Dict]        # Social agent findings
    media_results: Optional[Dict]         # Media agent findings
    location_results: Optional[Dict]      # Location agent findings (venues)
    final_profile: Optional[Dict]         # Merged & validated output
    tool_calls: Annotated[List[Dict], operator.add]
```

---

## Tools Specification

### Web Scraping Tools

```python
@tool
def scrape_page(url: str) -> Dict:
    """Scrape page content and extract structured data.
    Uses Firecrawl/Playwright for JS-rendered pages."""

@tool
def extract_metadata(html: str, entity_type: str) -> Dict:
    """Extract entity-specific metadata using CSS selectors + LLM."""
```

### Social Discovery Tools

```python
@tool
def search_instagram(entity_name: str, entity_type: str) -> Optional[str]:
    """Search for Instagram profile via web search + validation."""

@tool
def search_website(entity_name: str, context: str) -> Optional[str]:
    """Find official website via web search."""

@tool
def validate_url(url: str) -> Dict:
    """Check if URL is valid, accessible, and matches expected entity."""
```

### Media Platform Tools

```python
@tool
def search_soundcloud(entity_name: str) -> Optional[str]:
    """Search SoundCloud for artist/label profile."""

@tool
def search_spotify(entity_name: str, entity_type: str) -> Optional[str]:
    """Search Spotify API for artist/label."""

@tool
def search_bandcamp(entity_name: str) -> Optional[str]:
    """Search Bandcamp for artist/label profile."""
```

### Location Tools (Venues Only)

```python
@tool
def search_google_places(venue_name: str, city: str) -> Optional[Dict]:
    """Search Google Places API for venue details."""

@tool
def geocode_address(address: str) -> Optional[str]:
    """Convert address to Google Maps link."""
```

---

## Environment Variables

```bash
# LLM Provider (required)
OPENAI_API_KEY=sk-...
# OR
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-4o-mini

# Web Scraping (required - choose one)
FIRECRAWL_API_KEY=...              # Recommended for JS pages
# OR
BROWSERLESS_API_KEY=...            # Alternative scraping

# Search API
SERPAPI_API_KEY=...

# Platform APIs (optional - enables direct search)
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
GOOGLE_PLACES_API_KEY=...          # Required for venue location

# Observability (optional)
ARIZE_SPACE_ID=...
ARIZE_API_KEY=...
```

---

## Graceful Degradation Strategy

Following the trip-planner pattern, every tool has a fallback:

```
API Search → Web Search (SerpAPI) → LLM Inference → Empty
```

Example:
```python
def search_spotify(entity_name: str, entity_type: str) -> Optional[str]:
    # 1. Try Spotify API directly
    if SPOTIFY_CONFIGURED:
        result = spotify_api.search(entity_name, type=entity_type)
        if result:
            return result
    
    # 2. Fallback to web search
    query = f"{entity_name} spotify {entity_type}"
    result = _serp_search_raw(query)
    if result:
        return _extract_spotify_url(result)
    
    # 3. LLM inference fallback
    return _llm_fallback(f"Find Spotify URL for {entity_type} {entity_name}")
```

---

## Execution Flow

1. **Input Validation** - Validate URL format and accessibility
2. **Page Scraping** - Extract raw content from source URL (Scraper Agent)
3. **Entity Classification** - Detect type (artist/venue/promoter/label) and name
4. **Parallel Enrichment** - Run Social, Media, and Location agents simultaneously
5. **Result Synthesis** - Merge findings, resolve conflicts, validate URLs
6. **Response** - Return enriched profile with confidence scores

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Field Coverage | >85% | % of applicable fields successfully enriched |
| URL Accuracy | >95% | % of returned URLs that are valid and correct |
| Processing Time | <5s | Average time per entity |
| Fallback Rate | <20% | % of requests requiring LLM fallback |

---

## MVP Scope (Phase 1)

**In Scope:**
- Single URL input
- Artist entity type (primary focus)
- Core fields: Instagram, SoundCloud, Spotify, Website, Biography
- Web search-based discovery
- Basic URL validation

**Out of Scope (Future):**
- Batch processing
- All entity types
- Direct platform API integrations
- Image similarity matching for profile pictures
- Caching layer

---

## Technical Dependencies

```txt
# Core
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
langgraph>=0.2.55
langchain>=0.3.7
langchain-openai>=0.2.10

# Web Scraping
httpx>=0.25.0
beautifulsoup4>=4.12.0
firecrawl-py>=0.0.1        # Optional: for JS-heavy pages

# Observability
arize-otel>=0.8.1
openinference-instrumentation-langchain>=0.1.19

# Utilities
pydantic>=2.0.0
python-dotenv>=1.0.0
validators>=0.22.0          # URL validation
```

---

## Project Structure

```
data-enrichment-agent/
├── backend/
│   ├── main.py                 # FastAPI app, agents, graph
│   ├── tools/
│   │   ├── scraping.py         # Web scraping tools
│   │   ├── social.py           # Social media discovery
│   │   ├── media.py            # Streaming platform discovery
│   │   └── location.py         # Venue location tools
│   ├── agents/
│   │   ├── scraper.py          # Scraper agent
│   │   ├── classifier.py       # Entity classifier
│   │   ├── social.py           # Social agent
│   │   ├── media.py            # Media agent
│   │   ├── location.py         # Location agent
│   │   └── enrichment.py       # Final synthesis agent
│   └── requirements.txt
├── frontend/
│   └── index.html              # Simple input UI
├── tests/
│   └── test_enrichment.py
└── render.yaml
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Rate limiting by platforms | High | Implement caching, respect robots.txt, use multiple API keys |
| Incorrect entity matching | Medium | Confidence scoring, human review for low-confidence results |
| Stale/broken URLs | Medium | Periodic re-validation, timestamp tracking |
| Platform structure changes | Medium | Modular scrapers with fallback chains |

---

## Open Questions

1. Should we store enriched data in a database or return ephemeral results?
2. What confidence threshold triggers a "needs review" flag?
3. Should we support batch enrichment via CSV upload in Phase 1?
4. Integration priority: which data sources to support first beyond RA?

---

*Document Version: 1.0*  
*Last Updated: January 28, 2026*
