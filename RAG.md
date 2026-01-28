# Local Guide RAG Demo

This AI Trip Planner includes an optional Retrieval-Augmented Generation (RAG) feature that powers the `local_agent` with curated, real-world local experiences. RAG stays dormant until you opt in, making it perfect for learning step-by-step.

## What is RAG?

RAG (Retrieval-Augmented Generation) combines:
1. **Retrieval**: Search a database for relevant information
2. **Augmentation**: Add that information to the LLM's context
3. **Generation**: LLM generates responses using both its knowledge and the retrieved data

This pattern is fundamental in production AI systems because it:
- Grounds responses in real, curated data
- Provides citations and sources
- Reduces hallucinations
- Allows updating knowledge without retraining models

## How to Enable RAG

### 1. Set the Feature Flag

Copy `backend/.env.example` to `backend/.env` if you haven't already, then:

```bash
# Enable RAG feature
ENABLE_RAG=1

# Provide your OpenAI API key (needed for embeddings)
OPENAI_API_KEY=sk-...

# Optional: override the embeddings model (defaults to text-embedding-3-small)
OPENAI_EMBED_MODEL=text-embedding-3-small
```

### 2. Restart the Server

```bash
cd backend
uvicorn main:app --reload --port 8000
```

On startup, the app will:
- Load 540+ curated local experiences from `backend/data/local_guides.json`
- Create vector embeddings for semantic search
- Index them into an in-memory vector store

### 3. Test It Out

Make a request with specific interests:

```bash
curl -X POST http://localhost:8000/plan-trip \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Tokyo",
    "duration": "5 days",
    "interests": "food, anime, technology"
  }'
```

The `local_agent` will now retrieve the most relevant local experiences from the database and incorporate them into its recommendations!

## What Happens Behind the Scenes

### With ENABLE_RAG=1 (Semantic Search)

```
User Request: "Tokyo with food, anime interests"
       ↓
1. Create query embedding (OpenAI text-embedding-3-small)
       ↓
2. Search vector store for top 3 similar experiences
       ↓
3. Retrieve: "Tsukiji Market tour", "Studio Ghibli Museum", "Akihabara gaming"
       ↓
4. Inject retrieved context into local_agent prompt
       ↓
5. LLM generates response using curated data + its knowledge
```

### With ENABLE_RAG=0 (Default Behavior)

The `local_agent` falls back to its original heuristic responses using the mock `local_flavor`, `local_customs`, and `hidden_gems` tools.

## The Local Guides Database

Located at `backend/data/local_guides.json`, this file contains 540+ curated experiences across 20 cities:

```json
[
  {
    "city": "Tokyo",
    "interests": ["food", "sushi"],
    "description": "Join a former Tsukiji auctioneer at Toyosu Market for tuna tastings...",
    "source": "https://www.tsukiji.or.jp"
  },
  {
    "city": "Prague",
    "interests": ["history", "architecture"],
    "description": "Join a historian for a dawn walk along the Royal Route...",
    "source": "https://www.prague.eu/en"
  }
]
```

Each entry includes:
- **city**: Destination name
- **interests**: List of relevant topics (food, art, history, etc.)
- **description**: Detailed experience description
- **source**: Citation URL for verification

## Graceful Fallback Strategy

The RAG implementation demonstrates production-ready error handling:

### Scenario 1: No OpenAI API Key
→ Falls back to **keyword matching** (simple text search)

### Scenario 2: OpenAI API Error
→ Falls back to **keyword matching**

### Scenario 3: No Matching Results
→ Falls back to **keyword matching**, then to **empty results**

### Scenario 4: ENABLE_RAG=0
→ Returns **empty results**, local_agent uses its mock tools

This teaches students that production systems need multiple fallback layers!

## Observability in Arize

When RAG is enabled and Arize tracing is configured, you'll see:

1. **Embedding Spans**: Shows the embedding model and token count
2. **Retrieval Spans**: Shows the query and number of documents retrieved
3. **Retrieved Documents**: The actual content passed to the LLM
4. **Similarity Scores**: How well each document matched the query
5. **Metadata**: City, interests, and source URLs for each result

This makes debugging RAG systems much easier!

## How Students Can Extend This

### Add More Cities

Edit `backend/data/local_guides.json`:

```json
{
  "city": "Paris",
  "interests": ["food", "wine"],
  "description": "Wine tasting tour in Montmartre...",
  "source": "https://example.com"
}
```

Restart the server - embeddings will be regenerated automatically!

### Experiment with Different Embeddings

Try different models in `.env`:

```bash
# Smaller, faster (default)
OPENAI_EMBED_MODEL=text-embedding-3-small

# Larger, more accurate
OPENAI_EMBED_MODEL=text-embedding-3-large

# Legacy model
OPENAI_EMBED_MODEL=text-embedding-ada-002
```

### Adjust Retrieval Parameters

In `backend/main.py`, modify the `local_agent`:

```python
# Retrieve more results
retrieved = GUIDE_RETRIEVER.retrieve(destination, interests, k=5)  # was k=3

# Use different search parameters
retriever = self._vectorstore.as_retriever(
    search_kwargs={"k": 10, "score_threshold": 0.7}
)
```

### Add Metadata Filtering

Enhance the retriever to filter by city or interests before searching:

```python
# In LocalGuideRetriever.retrieve()
docs = retriever.invoke(
    query,
    filter={"city": destination}  # Only search this city
)
```

## Common Issues & Solutions

### "No embeddings created"
**Cause**: ENABLE_RAG=1 but no OPENAI_API_KEY  
**Solution**: Add your OpenAI API key to `.env`

### "Empty retrieval results"
**Cause**: City not in database OR interests don't match  
**Solution**: Check `local_guides.json` for your destination, or add entries

### "Rate limit errors"
**Cause**: Too many embedding requests during startup  
**Solution**: The embeddings are cached in memory. Restart less frequently, or use a smaller dataset for development.

### "Tracing not showing retrieval"
**Cause**: LangChain instrumentation not configured  
**Solution**: Ensure `LangChainInstrumentor().instrument()` is called at startup

## Learning Resources

- **LangChain Retrievers**: https://python.langchain.com/docs/modules/data_connection/retrievers/
- **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings
- **Vector Stores**: https://python.langchain.com/docs/integrations/vectorstores/
- **RAG Pattern**: https://www.pinecone.io/learn/retrieval-augmented-generation/

## Disabling RAG

To turn off RAG and return to the original behavior:

```bash
# In .env
ENABLE_RAG=0
```

Restart the server. The local_agent will use its original mock tools.

---

**Next Steps**: Try enabling RAG, make some test requests, and view the traces in Arize to see how retrieval augments the LLM's responses!

