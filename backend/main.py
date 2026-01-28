from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import time
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# URL validation
try:
    import validators
except ImportError:
    validators = None

# HTML parsing for scraping
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# Minimal observability via Arize/OpenInference (optional)
try:
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.instrumentation import using_prompt_template, using_metadata, using_attributes
    from opentelemetry import trace
    _TRACING = True
except Exception:
    def using_prompt_template(**kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_metadata(*args, **kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_attributes(*args, **kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    _TRACING = False

# LangGraph + LangChain
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
import httpx


class TripRequest(BaseModel):
    destination: str
    duration: str
    budget: Optional[str] = None
    interests: Optional[str] = None
    travel_style: Optional[str] = None
    # Optional fields for enhanced session tracking and observability
    user_input: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    turn_index: Optional[int] = None


class TripResponse(BaseModel):
    result: str
    tool_calls: List[Dict[str, Any]] = []


def _init_llm():
    # Simple, test-friendly LLM init
    class _Fake:
        def __init__(self):
            pass
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            class _Msg:
                content = "Test itinerary"
                tool_calls: List[Dict[str, Any]] = []
            return _Msg()

    if os.getenv("TEST_MODE"):
        return _Fake()
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1500)
    elif os.getenv("OPENROUTER_API_KEY"):
        # Use OpenRouter via OpenAI-compatible client
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.7,
        )
    else:
        # Require a key unless running tests
        raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env")


llm = _init_llm()


# Feature flag for optional RAG demo (opt-in for learning)
ENABLE_RAG = os.getenv("ENABLE_RAG", "0").lower() not in {"0", "false", "no"}


# RAG helper: Load curated local guides as LangChain documents
def _load_local_documents(path: Path) -> List[Document]:
    """Load local guides JSON and convert to LangChain Documents."""
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return []

    docs: List[Document] = []
    for row in raw:
        description = row.get("description")
        city = row.get("city")
        if not description or not city:
            continue
        interests = row.get("interests", []) or []
        metadata = {
            "city": city,
            "interests": interests,
            "source": row.get("source"),
        }
        # Prefix city + interests in content so embeddings capture location context
        interest_text = ", ".join(interests) if interests else "general travel"
        content = f"City: {city}\nInterests: {interest_text}\nGuide: {description}"
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


class LocalGuideRetriever:
    """Retrieves curated local experiences using vector similarity search.
    
    This class demonstrates production RAG patterns for students:
    - Vector embeddings for semantic search
    - Fallback to keyword matching when embeddings unavailable
    - Graceful degradation with feature flags
    """
    
    def __init__(self, data_path: Path):
        """Initialize retriever with local guides data.
        
        Args:
            data_path: Path to local_guides.json file
        """
        self._docs = _load_local_documents(data_path)
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._vectorstore: Optional[InMemoryVectorStore] = None
        
        # Only create embeddings when RAG is enabled and we have an API key
        if ENABLE_RAG and self._docs and not os.getenv("TEST_MODE"):
            try:
                model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
                self._embeddings = OpenAIEmbeddings(model=model)
                store = InMemoryVectorStore(embedding=self._embeddings)
                store.add_documents(self._docs)
                self._vectorstore = store
            except Exception:
                # Gracefully degrade to keyword search if embeddings fail
                self._embeddings = None
                self._vectorstore = None

    @property
    def is_empty(self) -> bool:
        """Check if any documents were loaded."""
        return not self._docs

    def retrieve(self, destination: str, interests: Optional[str], *, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant local guides for a destination.
        
        Args:
            destination: City or destination name
            interests: Comma-separated interests (e.g., "food, art")
            k: Number of results to return
            
        Returns:
            List of dicts with 'content', 'metadata', and 'score' keys
        """
        if not ENABLE_RAG or self.is_empty:
            return []

        # Use vector search if available, otherwise fall back to keywords
        if not self._vectorstore:
            return self._keyword_fallback(destination, interests, k=k)

        query = destination
        if interests:
            query = f"{destination} with interests {interests}"
        
        try:
            # LangChain retriever ensures embeddings + searches are traced
            retriever = self._vectorstore.as_retriever(search_kwargs={"k": max(k, 4)})
            docs = retriever.invoke(query)
        except Exception:
            return self._keyword_fallback(destination, interests, k=k)

        # Format results with metadata and scores
        top_docs = docs[:k]
        results = []
        for doc in top_docs:
            score_val: float = 0.0
            if isinstance(doc.metadata, dict):
                maybe_score = doc.metadata.get("score")
                if isinstance(maybe_score, (int, float)):
                    score_val = float(maybe_score)
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score_val,
            })

        if not results:
            return self._keyword_fallback(destination, interests, k=k)
        return results

    def _keyword_fallback(self, destination: str, interests: Optional[str], *, k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval when embeddings unavailable.
        
        This demonstrates graceful degradation for students learning about
        fallback strategies in production systems.
        """
        dest_lower = destination.lower()
        interest_terms = [part.strip().lower() for part in (interests or "").split(",") if part.strip()]

        def _score(doc: Document) -> int:
            score = 0
            city_match = doc.metadata.get("city", "").lower()
            # Match city name
            if dest_lower and dest_lower.split(",")[0] in city_match:
                score += 2
            # Match interests
            for term in interest_terms:
                if term and term in " ".join(doc.metadata.get("interests") or []).lower():
                    score += 1
                if term and term in doc.page_content.lower():
                    score += 1
            return score

        scored_docs = [(_score(doc), doc) for doc in self._docs]
        scored_docs.sort(key=lambda item: item[0], reverse=True)
        top_docs = scored_docs[:k]
        
        results = []
        for score, doc in top_docs:
            if score > 0:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                })
        return results


# Initialize retriever at module level (loads data once at startup)
_DATA_DIR = Path(__file__).parent / "data"
GUIDE_RETRIEVER = LocalGuideRetriever(_DATA_DIR / "local_guides.json")


# Search API configuration and helpers
SEARCH_TIMEOUT = 10.0  # seconds


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
    """Search the web using Tavily or SerpAPI if configured, return None otherwise.
    
    This demonstrates graceful degradation: tools work with or without API keys.
    Students can enable real search by adding TAVILY_API_KEY or SERPAPI_API_KEY.
    """
    query = query.strip()
    if not query:
        return None

    # Try Tavily first (recommended for AI apps)
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
            pass  # Fail gracefully, try next option

    # Try SerpAPI as fallback
    serp_key = os.getenv("SERPAPI_API_KEY")
    if serp_key:
        try:
            with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
                resp = client.get(
                    "https://serpapi.com/search",
                    params={
                        "api_key": serp_key,
                        "engine": "google",
                        "num": 5,
                        "q": query,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                organic = data.get("organic_results", [])
                snippets = [item.get("snippet", "") for item in organic]
                combined = " ".join(snippets).strip()
                if combined:
                    return _compact(combined)
        except Exception:
            pass  # Fail gracefully

    return None  # No search APIs configured


def _llm_fallback(instruction: str, context: Optional[str] = None) -> str:
    """Use the LLM to generate a response when search APIs aren't available.
    
    This ensures tools always return useful information, even without API keys.
    """
    prompt = "Respond with 200 characters or less.\n" + instruction.strip()
    if context:
        prompt += "\nContext:\n" + context.strip()
    response = llm.invoke([
        SystemMessage(content="You are a concise travel assistant."),
        HumanMessage(content=prompt),
    ])
    return _compact(response.content)


def _with_prefix(prefix: str, summary: str) -> str:
    """Add a prefix to a summary for clarity."""
    text = f"{prefix}: {summary}" if prefix else summary
    return _compact(text)


# Tools with real API calls + LLM fallback (graceful degradation pattern)
@tool
def essential_info(destination: str) -> str:
    """Return essential destination info like weather, sights, and etiquette."""
    query = f"{destination} travel essentials weather best time top attractions etiquette language currency safety"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} essentials", summary)
    
    # LLM fallback when no search API is configured
    instruction = f"Summarize the climate, best visit time, standout sights, customs, language, currency, and safety tips for {destination}."
    return _llm_fallback(instruction)


@tool
def budget_basics(destination: str, duration: str) -> str:
    """Return high-level budget categories for a given destination and duration."""
    query = f"{destination} travel budget average daily costs {duration}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} budget {duration}", summary)
    
    instruction = f"Outline lodging, meals, transport, activities, and extra costs for a {duration} trip to {destination}."
    return _llm_fallback(instruction)


@tool
def local_flavor(destination: str, interests: Optional[str] = None) -> str:
    """Suggest authentic local experiences matching optional interests."""
    focus = interests or "local culture"
    query = f"{destination} authentic local experiences {focus}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} {focus}", summary)
    
    instruction = f"Recommend authentic local experiences in {destination} that highlight {focus}."
    return _llm_fallback(instruction)


@tool
def day_plan(destination: str, day: int) -> str:
    """Return a simple day plan outline for a specific day number."""
    query = f"{destination} day {day} itinerary highlights"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"Day {day} in {destination}", summary)
    
    instruction = f"Outline key activities for day {day} in {destination}, covering morning, afternoon, and evening."
    return _llm_fallback(instruction)


# Additional simple tools per agent (to mirror original multi-tool behavior)
@tool
def weather_brief(destination: str) -> str:
    """Return a brief weather summary for planning purposes."""
    query = f"{destination} weather forecast travel season temperatures rainfall"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} weather", summary)
    
    instruction = f"Give a weather brief for {destination} noting season, temperatures, rainfall, humidity, and packing guidance."
    return _llm_fallback(instruction)


@tool
def visa_brief(destination: str) -> str:
    """Return a brief visa guidance for travel planning."""
    query = f"{destination} tourist visa requirements entry rules"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} visa", summary)
    
    instruction = f"Provide a visa guidance summary for visiting {destination}, including advice to confirm with the relevant embassy."
    return _llm_fallback(instruction)


@tool
def attraction_prices(destination: str, attractions: Optional[List[str]] = None) -> str:
    """Return pricing information for attractions."""
    items = attractions or ["popular attractions"]
    focus = ", ".join(items)
    query = f"{destination} attraction ticket prices {focus}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} attraction prices", summary)
    
    instruction = f"Share typical ticket prices and savings tips for attractions such as {focus} in {destination}."
    return _llm_fallback(instruction)


@tool
def local_customs(destination: str) -> str:
    """Return cultural etiquette and customs information."""
    query = f"{destination} cultural etiquette travel customs"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} customs", summary)
    
    instruction = f"Summarize key etiquette and cultural customs travelers should know before visiting {destination}."
    return _llm_fallback(instruction)


@tool
def hidden_gems(destination: str) -> str:
    """Return lesser-known attractions and experiences."""
    query = f"{destination} hidden gems local secrets lesser known spots"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} hidden gems", summary)
    
    instruction = f"List lesser-known attractions or experiences that feel like hidden gems in {destination}."
    return _llm_fallback(instruction)


@tool
def travel_time(from_location: str, to_location: str, mode: str = "public") -> str:
    """Return travel time estimates between locations."""
    query = f"travel time {from_location} to {to_location} by {mode}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{from_location}→{to_location} {mode}", summary)
    
    instruction = f"Estimate travel time from {from_location} to {to_location} by {mode} transport."
    return _llm_fallback(instruction)


@tool
def packing_list(destination: str, duration: str, activities: Optional[List[str]] = None) -> str:
    """Return packing recommendations for the trip."""
    acts = ", ".join(activities or ["sightseeing"])
    query = f"what to pack for {destination} {duration} {acts}"
    summary = _search_api(query)
    if summary:
        return _with_prefix(f"{destination} packing", summary)
    
    instruction = f"Suggest packing essentials for a {duration} trip to {destination} focused on {acts}."
    return _llm_fallback(instruction)


class TripState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trip_request: Dict[str, Any]
    research: Optional[str]
    budget: Optional[str]
    local: Optional[str]
    final: Optional[str]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]


# =============================================================================
# DATA ENRICHMENT AGENT - Phase 1 (Artists focus)
# =============================================================================

class EnrichRequest(BaseModel):
    """Request model for entity enrichment."""
    url: str = Field(..., description="URL of the entity page to enrich")
    entity_type: Optional[str] = Field(None, description="Entity type: artist, venue, promoter, label")
    priority_fields: Optional[List[str]] = Field(None, description="Fields to prioritize")
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


class EnrichmentState(TypedDict):
    """State for the enrichment agent graph."""
    messages: Annotated[List[BaseMessage], operator.add]
    enrich_request: Dict[str, Any]
    scraped_data: Optional[Dict[str, Any]]
    entity_type: Optional[str]
    entity_name: Optional[str]
    social_results: Optional[Dict[str, Any]]
    media_results: Optional[Dict[str, Any]]
    final_profile: Optional[Dict[str, Any]]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]


# --- Enrichment Tools ---

def _extract_url_from_text(text: str, platform: str) -> Optional[str]:
    """Extract a platform URL from text using regex patterns."""
    patterns = {
        "instagram": r'(?:https?://)?(?:www\.)?instagram\.com/([a-zA-Z0-9_\.]+)/?',
        "soundcloud": r'(?:https?://)?(?:www\.)?soundcloud\.com/([a-zA-Z0-9_-]+)/?',
        "spotify": r'(?:https?://)?open\.spotify\.com/artist/([a-zA-Z0-9]+)',
        "bandcamp": r'(?:https?://)?([a-zA-Z0-9-]+)\.bandcamp\.com/?',
        "website": r'(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})/?',
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
    # Basic validation fallback
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


@tool
def scrape_entity_page(url: str) -> Dict[str, Any]:
    """Scrape an entity page and extract available data.
    
    Returns structured data including name, bio, and any social links found on the page.
    """
    result = {
        "url": url,
        "name": None,
        "bio": None,
        "links": {},
        "raw_text": "",
        "success": False,
    }
    
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            html = resp.text
            result["success"] = True
            
            if BeautifulSoup:
                soup = BeautifulSoup(html, "html.parser")
                
                # Extract title/name
                title_tag = soup.find("title")
                if title_tag:
                    result["name"] = title_tag.get_text().split("|")[0].split("-")[0].strip()
                
                # Extract meta description as potential bio
                meta_desc = soup.find("meta", attrs={"name": "description"})
                if meta_desc and meta_desc.get("content"):
                    result["bio"] = meta_desc["content"][:500]
                
                # Extract all links and look for social profiles
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    if "instagram.com" in href and "instagram" not in result["links"]:
                        result["links"]["instagram"] = href
                    elif "soundcloud.com" in href and "soundcloud" not in result["links"]:
                        result["links"]["soundcloud"] = href
                    elif "spotify.com" in href and "spotify" not in result["links"]:
                        result["links"]["spotify"] = href
                    elif "bandcamp.com" in href and "bandcamp" not in result["links"]:
                        result["links"]["bandcamp"] = href
                
                # Get text content for further analysis
                result["raw_text"] = soup.get_text(separator=" ", strip=True)[:2000]
            else:
                result["raw_text"] = html[:2000]
                
    except Exception as e:
        result["error"] = str(e)
    
    return result


@tool
def search_instagram(entity_name: str, entity_type: str = "artist") -> Optional[str]:
    """Search for an entity's Instagram profile.
    
    Uses web search to find the Instagram profile, then validates the URL.
    """
    query = f"{entity_name} {entity_type} instagram official"
    summary = _search_api(query)
    
    if summary:
        # Try to extract Instagram URL from search results
        url = _extract_url_from_text(summary, "instagram")
        if url and _validate_url(url):
            return url
    
    # LLM fallback to infer likely Instagram handle
    instruction = f"What is the likely Instagram username for {entity_type} '{entity_name}'? Return ONLY the username, nothing else."
    response = llm.invoke([
        SystemMessage(content="You are a music industry data specialist. Be concise."),
        HumanMessage(content=instruction),
    ])
    handle = response.content.strip().replace("@", "").split()[0] if response.content else None
    if handle and len(handle) > 2:
        return f"https://instagram.com/{handle}"
    return None


@tool
def search_soundcloud(entity_name: str) -> Optional[str]:
    """Search for an entity's SoundCloud profile.
    
    Uses web search to find the SoundCloud profile.
    """
    query = f"{entity_name} soundcloud official"
    summary = _search_api(query)
    
    if summary:
        url = _extract_url_from_text(summary, "soundcloud")
        if url and _validate_url(url):
            return url
    
    # LLM fallback
    instruction = f"What is the likely SoundCloud URL for artist '{entity_name}'? Return ONLY the URL, nothing else."
    response = llm.invoke([
        SystemMessage(content="You are a music industry data specialist. Be concise."),
        HumanMessage(content=instruction),
    ])
    url = _extract_url_from_text(response.content, "soundcloud") if response.content else None
    return url if url and _validate_url(url) else None


@tool
def search_spotify(entity_name: str, entity_type: str = "artist") -> Optional[str]:
    """Search for an entity's Spotify profile.
    
    Uses web search to find the Spotify artist/label page.
    """
    query = f"{entity_name} spotify {entity_type}"
    summary = _search_api(query)
    
    if summary:
        url = _extract_url_from_text(summary, "spotify")
        if url and _validate_url(url):
            return url
    
    # LLM fallback
    instruction = f"What is the Spotify artist page URL for '{entity_name}'? Return ONLY the URL, nothing else."
    response = llm.invoke([
        SystemMessage(content="You are a music industry data specialist. Be concise."),
        HumanMessage(content=instruction),
    ])
    url = _extract_url_from_text(response.content, "spotify") if response.content else None
    return url if url and _validate_url(url) else None


@tool
def search_website(entity_name: str, entity_type: str = "artist") -> Optional[str]:
    """Search for an entity's official website.
    
    Uses web search to find the official website.
    """
    query = f"{entity_name} {entity_type} official website"
    summary = _search_api(query)
    
    if summary:
        # Look for likely website patterns
        patterns = [
            rf'(?:https?://)?(?:www\.)?{re.escape(entity_name.lower().replace(" ", ""))}\.com',
            rf'(?:https?://)?(?:www\.)?{re.escape(entity_name.lower().replace(" ", "-"))}\.com',
        ]
        for pattern in patterns:
            match = re.search(pattern, summary, re.IGNORECASE)
            if match:
                url = match.group(0)
                if not url.startswith("http"):
                    url = "https://" + url
                if _validate_url(url):
                    return url
    
    # LLM fallback
    instruction = f"What is the official website URL for {entity_type} '{entity_name}'? Return ONLY the URL or 'unknown' if you don't know."
    response = llm.invoke([
        SystemMessage(content="You are a music industry data specialist. Be concise."),
        HumanMessage(content=instruction),
    ])
    content = response.content.strip() if response.content else ""
    if content and content.lower() != "unknown" and "." in content:
        url = content if content.startswith("http") else f"https://{content}"
        return url if _validate_url(url) else None
    return None


@tool
def extract_biography(entity_name: str, context: str = "") -> str:
    """Extract or generate a biography for an entity.
    
    Uses scraped content or web search to find biographical information.
    """
    if context and len(context) > 100:
        # Use LLM to extract bio from scraped content
        instruction = f"Extract a concise biography (2-3 sentences) for '{entity_name}' from this content:\n{context[:1500]}"
        response = llm.invoke([
            SystemMessage(content="You are extracting artist biographies. Be factual and concise."),
            HumanMessage(content=instruction),
        ])
        return response.content.strip() if response.content else ""
    
    # Search for biography
    query = f"{entity_name} DJ producer biography"
    summary = _search_api(query)
    if summary:
        return _compact(summary, limit=500)
    
    # LLM fallback
    instruction = f"Write a brief 2-3 sentence biography for electronic music artist '{entity_name}'."
    response = llm.invoke([
        SystemMessage(content="You are a music journalist. Be factual and concise."),
        HumanMessage(content=instruction),
    ])
    return _compact(response.content, limit=500) if response.content else ""


# --- Enrichment Agents ---

def scraper_agent(state: EnrichmentState) -> EnrichmentState:
    """Scrape the source URL and extract initial data."""
    req = state["enrich_request"]
    url = req["url"]
    
    prompt_t = (
        "You are a web scraper agent.\n"
        "Scrape the page at {url} and extract: entity name, biography, and any social links.\n"
        "Use the scrape_entity_page tool."
    )
    vars_ = {"url": url}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [scrape_entity_page]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    scraped_data: Dict[str, Any] = {}
    entity_name = None
    entity_type = req.get("entity_type", "artist")
    
    with using_attributes(tags=["scraper", "enrichment"]):
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "scraper", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Extract scraped data from tool results
        for msg in tr["messages"]:
            if hasattr(msg, "content") and msg.content:
                try:
                    scraped_data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    entity_name = scraped_data.get("name")
                except (json.JSONDecodeError, TypeError):
                    scraped_data = {"raw_text": str(msg.content)}
    
    # If no name found, try to extract from URL
    if not entity_name:
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        if path_parts:
            entity_name = path_parts[-1].replace("-", " ").replace("_", " ").title()
    
    return {
        "messages": [SystemMessage(content=f"Scraped: {entity_name}")],
        "scraped_data": scraped_data,
        "entity_name": entity_name,
        "entity_type": entity_type,
        "tool_calls": calls,
    }


def social_agent(state: EnrichmentState) -> EnrichmentState:
    """Find social media profiles (Instagram, Website)."""
    entity_name = state.get("entity_name", "Unknown")
    entity_type = state.get("entity_type", "artist")
    scraped_data = state.get("scraped_data") or {}
    existing_links = scraped_data.get("links", {})
    
    prompt_t = (
        "You are a social media discovery agent.\n"
        "Find Instagram and website for {entity_type} '{entity_name}'.\n"
        "Use tools to search for these profiles."
    )
    vars_ = {"entity_name": entity_name, "entity_type": entity_type}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [search_instagram, search_website]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    social_results: Dict[str, Any] = {}
    
    # Use already-found links from scraping
    if existing_links.get("instagram"):
        social_results["instagram"] = {"value": existing_links["instagram"], "source": "source_page", "confidence": 1.0}
    
    with using_attributes(tags=["social", "enrichment"]):
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "social", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Process tool results
        for msg in tr["messages"]:
            if hasattr(msg, "content") and msg.content:
                content = str(msg.content)
                if "instagram.com" in content.lower() and "instagram" not in social_results:
                    url = _extract_url_from_text(content, "instagram") or content
                    social_results["instagram"] = {"value": url, "source": "web_search", "confidence": 0.85}
                elif "." in content and "instagram" not in content.lower() and "website" not in social_results:
                    social_results["website"] = {"value": content, "source": "web_search", "confidence": 0.8}
    
    return {
        "messages": [SystemMessage(content=f"Social found: {list(social_results.keys())}")],
        "social_results": social_results,
        "tool_calls": calls,
    }


def media_agent(state: EnrichmentState) -> EnrichmentState:
    """Find streaming platform profiles (SoundCloud, Spotify)."""
    entity_name = state.get("entity_name", "Unknown")
    entity_type = state.get("entity_type", "artist")
    scraped_data = state.get("scraped_data") or {}
    existing_links = scraped_data.get("links", {})
    
    prompt_t = (
        "You are a music platform discovery agent.\n"
        "Find SoundCloud and Spotify profiles for {entity_type} '{entity_name}'.\n"
        "Use tools to search for these profiles."
    )
    vars_ = {"entity_name": entity_name, "entity_type": entity_type}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [search_soundcloud, search_spotify]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    media_results: Dict[str, Any] = {}
    
    # Use already-found links from scraping
    if existing_links.get("soundcloud"):
        media_results["soundcloud"] = {"value": existing_links["soundcloud"], "source": "source_page", "confidence": 1.0}
    if existing_links.get("spotify"):
        media_results["spotify"] = {"value": existing_links["spotify"], "source": "source_page", "confidence": 1.0}
    
    with using_attributes(tags=["media", "enrichment"]):
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "media", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Process tool results
        for msg in tr["messages"]:
            if hasattr(msg, "content") and msg.content:
                content = str(msg.content)
                if "soundcloud.com" in content.lower() and "soundcloud" not in media_results:
                    url = _extract_url_from_text(content, "soundcloud") or content
                    media_results["soundcloud"] = {"value": url, "source": "web_search", "confidence": 0.85}
                elif "spotify.com" in content.lower() and "spotify" not in media_results:
                    url = _extract_url_from_text(content, "spotify") or content
                    media_results["spotify"] = {"value": url, "source": "web_search", "confidence": 0.85}
    
    return {
        "messages": [SystemMessage(content=f"Media found: {list(media_results.keys())}")],
        "media_results": media_results,
        "tool_calls": calls,
    }


def enrichment_agent(state: EnrichmentState) -> EnrichmentState:
    """Synthesize all results and create final enriched profile."""
    entity_name = state.get("entity_name", "Unknown")
    entity_type = state.get("entity_type", "artist")
    scraped_data = state.get("scraped_data") or {}
    social_results = state.get("social_results") or {}
    media_results = state.get("media_results") or {}
    
    # Extract biography
    bio_context = scraped_data.get("bio") or scraped_data.get("raw_text", "")
    tools = [extract_biography]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    bio = ""
    
    prompt_t = "Extract a biography for '{entity_name}' from the available context."
    vars_ = {"entity_name": entity_name}
    
    messages = [
        SystemMessage(content=prompt_t.format(**vars_)),
        HumanMessage(content=f"Context: {bio_context[:1000]}" if bio_context else "No context available."),
    ]
    
    with using_attributes(tags=["enrichment", "synthesis"]):
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "enrichment", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        for msg in tr["messages"]:
            if hasattr(msg, "content") and msg.content:
                bio = str(msg.content)
                break
    
    if not bio and bio_context:
        bio = _compact(bio_context, limit=300)
    
    # Merge all results into final profile
    final_profile: Dict[str, Any] = {
        "biography": {
            "value": bio,
            "source": "source_page" if scraped_data.get("bio") else "llm_generated",
            "confidence": 0.9 if scraped_data.get("bio") else 0.7,
        }
    }
    
    # Add social results
    for key, val in social_results.items():
        if isinstance(val, dict):
            final_profile[key] = val
        else:
            final_profile[key] = {"value": val, "source": "web_search", "confidence": 0.8}
    
    # Add media results
    for key, val in media_results.items():
        if isinstance(val, dict):
            final_profile[key] = val
        else:
            final_profile[key] = {"value": val, "source": "web_search", "confidence": 0.8}
    
    return {
        "messages": [SystemMessage(content=f"Enrichment complete: {len(final_profile)} fields")],
        "final_profile": final_profile,
        "tool_calls": calls,
    }


def build_enrichment_graph():
    """Build the LangGraph workflow for entity enrichment."""
    g = StateGraph(EnrichmentState)
    g.add_node("scraper_node", scraper_agent)
    g.add_node("social_node", social_agent)
    g.add_node("media_node", media_agent)
    g.add_node("enrichment_node", enrichment_agent)
    
    # Scraper runs first
    g.add_edge(START, "scraper_node")
    
    # Social and Media agents run in parallel after scraper
    g.add_edge("scraper_node", "social_node")
    g.add_edge("scraper_node", "media_node")
    
    # Both feed into enrichment agent
    g.add_edge("social_node", "enrichment_node")
    g.add_edge("media_node", "enrichment_node")
    
    g.add_edge("enrichment_node", END)
    
    return g.compile()


def research_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    prompt_t = (
        "You are a research assistant.\n"
        "Gather essential information about {destination}.\n"
        "Use tools to get weather, visa, and essential info, then summarize."
    )
    vars_ = {"destination": destination}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [essential_info, weather_brief, visa_brief]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    tool_results = []
    
    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["research", "info_gathering"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "research")
                current_span.set_attribute("metadata.agent_node", "research_agent")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    # Collect tool calls and execute them
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "research", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        tool_results = tr["messages"]
        
        # Add tool results to conversation and ask LLM to synthesize
        messages.append(res)
        messages.extend(tool_results)
        
        synthesis_prompt = "Based on the above information, provide a comprehensive summary for the traveler."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        # Instrument synthesis LLM call with its own prompt template
        synthesis_vars = {"destination": destination, "context": "tool_results"}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "research": out, "tool_calls": calls}


def budget_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination, duration = req["destination"], req["duration"]
    budget = req.get("budget", "moderate")
    prompt_t = (
        "You are a budget analyst.\n"
        "Analyze costs for {destination} over {duration} with budget: {budget}.\n"
        "Use tools to get pricing information, then provide a detailed breakdown."
    )
    vars_ = {"destination": destination, "duration": duration, "budget": budget}
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [budget_basics, attraction_prices]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["budget", "cost_analysis"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "budget")
                current_span.set_attribute("metadata.agent_node", "budget_agent")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "budget", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tr["messages"])
        
        synthesis_prompt = f"Create a detailed budget breakdown for {duration} in {destination} with a {budget} budget."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        # Instrument synthesis LLM call
        synthesis_vars = {"duration": duration, "destination": destination, "budget": budget}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "budget": out, "tool_calls": calls}


def local_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    interests = req.get("interests", "local culture")
    travel_style = req.get("travel_style", "standard")
    
    # RAG: Retrieve curated local guides if enabled
    context_lines = []
    if ENABLE_RAG:
        retrieved = GUIDE_RETRIEVER.retrieve(destination, interests, k=3)
        if retrieved:
            context_lines.append("=== Curated Local Guides (from database) ===")
            for idx, item in enumerate(retrieved, 1):
                content = item["content"]
                source = item["metadata"].get("source", "Unknown")
                context_lines.append(f"{idx}. {content}")
                context_lines.append(f"   Source: {source}")
            context_lines.append("=== End of Curated Guides ===\n")
    
    context_text = "\n".join(context_lines) if context_lines else ""
    
    prompt_t = (
        "You are a local guide.\n"
        "Find authentic experiences in {destination} for someone interested in: {interests}.\n"
        "Travel style: {travel_style}. Use tools to gather local insights.\n"
    )
    
    # Add retrieved context to prompt if available
    if context_text:
        prompt_t += "\nRelevant curated experiences from our database:\n{context}\n"
    
    vars_ = {
        "destination": destination,
        "interests": interests,
        "travel_style": travel_style,
        "context": context_text if context_text else "No curated context available.",
    }
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [local_flavor, local_customs, hidden_gems]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["local", "local_experiences"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "local")
                current_span.set_attribute("metadata.agent_node", "local_agent")
                if ENABLE_RAG and context_text:
                    current_span.set_attribute("metadata.rag_enabled", "true")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = agent.invoke(messages)
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "local", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        # Add tool results and ask for synthesis
        messages.append(res)
        messages.extend(tr["messages"])
        
        synthesis_prompt = f"Create a curated list of authentic experiences for someone interested in {interests} with a {travel_style} approach."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        # Instrument synthesis LLM call
        synthesis_vars = {"interests": interests, "travel_style": travel_style, "destination": destination}
        with using_prompt_template(template=synthesis_prompt, variables=synthesis_vars, version="v1-synthesis"):
            final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content

    return {"messages": [SystemMessage(content=out)], "local": out, "tool_calls": calls}


def itinerary_agent(state: TripState) -> TripState:
    req = state["trip_request"]
    destination = req["destination"]
    duration = req["duration"]
    travel_style = req.get("travel_style", "standard")
    user_input = (req.get("user_input") or "").strip()
    
    prompt_parts = [
        "Create a {duration} itinerary for {destination} ({travel_style}).",
        "",
        "Inputs:",
        "Research: {research}",
        "Budget: {budget}",
        "Local: {local}",
    ]
    if user_input:
        prompt_parts.append("User input: {user_input}")
    
    prompt_t = "\n".join(prompt_parts)
    vars_ = {
        "duration": duration,
        "destination": destination,
        "travel_style": travel_style,
        "research": (state.get("research") or "")[:400],
        "budget": (state.get("budget") or "")[:400],
        "local": (state.get("local") or "")[:400],
        "user_input": user_input,
    }
    
    # Add span attributes for better observability in Arize
    # NOTE: using_attributes must be OUTER context for proper propagation
    with using_attributes(tags=["itinerary", "final_agent"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.itinerary", "true")
                current_span.set_attribute("metadata.agent_type", "itinerary")
                current_span.set_attribute("metadata.agent_node", "itinerary_agent")
                if user_input:
                    current_span.set_attribute("metadata.user_input", user_input)
        
        # Prompt template wrapper for Arize Playground integration
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1"):
            res = llm.invoke([SystemMessage(content=prompt_t.format(**vars_))])
    
    return {"messages": [SystemMessage(content=res.content)], "final": res.content}


def build_graph():
    g = StateGraph(TripState)
    g.add_node("research_node", research_agent)
    g.add_node("budget_node", budget_agent)
    g.add_node("local_node", local_agent)
    g.add_node("itinerary_node", itinerary_agent)

    # Run research, budget, and local agents in parallel
    g.add_edge(START, "research_node")
    g.add_edge(START, "budget_node")
    g.add_edge(START, "local_node")
    
    # All three agents feed into the itinerary agent
    g.add_edge("research_node", "itinerary_node")
    g.add_edge("budget_node", "itinerary_node")
    g.add_edge("local_node", "itinerary_node")
    
    g.add_edge("itinerary_node", END)

    # Compile without checkpointer to avoid state persistence issues
    return g.compile()


app = FastAPI(title="AI Trip Planner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# Initialize tracing once at startup, not per request
if _TRACING:
    try:
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        if space_id and api_key:
            tp = register(space_id=space_id, api_key=api_key, project_name="ai-trip-planner")
            LangChainInstrumentor().instrument(tracer_provider=tp, include_chains=True, include_agents=True, include_tools=True)
            LiteLLMInstrumentor().instrument(tracer_provider=tp, skip_dep_check=True)
    except Exception:
        pass

@app.post("/plan-trip", response_model=TripResponse)
def plan_trip(req: TripRequest):
    graph = build_graph()
    
    # Only include necessary fields in initial state
    # Agent outputs (research, budget, local, final) will be added during execution
    state = {
        "messages": [],
        "trip_request": req.model_dump(),
        "tool_calls": [],
    }
    
    # Add session and user tracking attributes to the trace
    session_id = req.session_id
    user_id = req.user_id
    turn_idx = req.turn_index
    
    # Build attributes for session and user tracking
    attrs_kwargs = {}
    if session_id:
        attrs_kwargs["session_id"] = session_id
    if user_id:
        attrs_kwargs["user_id"] = user_id
    
    # Add turn_index as a custom span attribute if provided
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


# =============================================================================
# DATA ENRICHMENT ENDPOINT
# =============================================================================

@app.post("/enrich", response_model=EnrichResponse)
def enrich_entity(req: EnrichRequest):
    """Enrich an electronic music entity with missing data fields.
    
    Accepts a URL to an artist, venue, promoter, or label page and
    discovers missing information like social links, biography, etc.
    """
    start_time = time.time()
    
    # Validate URL
    if not _validate_url(req.url):
        raise HTTPException(status_code=400, detail="Invalid URL provided")
    
    graph = build_enrichment_graph()
    
    # Initialize state
    state = {
        "messages": [],
        "enrich_request": req.model_dump(),
        "scraped_data": None,
        "entity_type": req.entity_type or "artist",
        "entity_name": None,
        "social_results": None,
        "media_results": None,
        "final_profile": None,
        "tool_calls": [],
    }
    
    # Run the enrichment graph
    with using_attributes(tags=["enrichment", "music_data"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.source_url", req.url)
                current_span.set_attribute("metadata.entity_type", req.entity_type or "artist")
        out = graph.invoke(state)
    
    # Build response
    final_profile = out.get("final_profile") or {}
    entity_name = out.get("entity_name") or "Unknown"
    entity_type = out.get("entity_type") or req.entity_type or "artist"
    
    # Convert to response format
    enriched_fields = {}
    expected_fields = ["biography", "instagram", "soundcloud", "spotify", "website"]
    missing_fields = []
    
    for field in expected_fields:
        if field in final_profile and final_profile[field].get("value"):
            enriched_fields[field] = EnrichedField(
                value=final_profile[field].get("value"),
                source=final_profile[field].get("source", "unknown"),
                confidence=final_profile[field].get("confidence", 0.0),
            )
        else:
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
