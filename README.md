# AI Trip Planner

A **production-ready multi-agent system** built for learning and customization. This repo demonstrates three essential AI engineering patterns that students can study, modify, and adapt for their own use cases.

## What You'll Learn

- 🤖 **Multi-Agent Orchestration**: 4 specialized agents running in parallel using LangGraph
- 🔍 **RAG (Retrieval-Augmented Generation)**: Vector search over curated data with fallback strategies
- 🌐 **API Integration**: Real-time web search with graceful degradation (LLM fallback)
- 📊 **Observability**: Production tracing with Arize for debugging and evaluation
- 🛠️ **Composable Architecture**: Easily adapt from "trip planner" to your own agent system

**Perfect for:** Students learning to build, evaluate, and deploy agentic AI systems.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Request                             │
│                    (destination, duration, interests)            │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   FastAPI Endpoint      │
                    │   + Session Tracking    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   LangGraph Workflow    │
                    │   (Parallel Execution)  │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
   ┌────▼─────┐           ┌─────▼──────┐         ┌──────▼─────┐
   │ Research │           │   Budget   │         │   Local    │
   │  Agent   │           │   Agent    │         │   Agent    │
   └────┬─────┘           └─────┬──────┘         └──────┬─────┘
        │                        │                        │
        │ Tools:                 │ Tools:                 │ Tools + RAG:
        │ • essential_info       │ • budget_basics        │ • local_flavor
        │ • weather_brief        │ • attraction_prices    │ • hidden_gems
        │ • visa_brief           │                        │ • Vector search
        │                        │                        │   (90+ guides)
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 │
                            ┌────▼─────┐
                            │Itinerary │
                            │  Agent   │
                            │(Synthesis)│
                            └────┬─────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Final Itinerary       │
                    │   + Tool Call Metadata  │
                    └─────────────────────────┘

All agents, tools, and LLM calls → Arize Observability Platform
```

## Learning Paths

### 🎓 Beginner Path
1. **Setup & Run** (15 min)
   - Clone repo, configure `.env` with OpenAI key
   - Start server: `./start.sh`
   - Test API: `python "test scripts/test_api.py"`

2. **Observe & Understand** (30 min)
   - Make a few trip planning requests
   - View traces in Arize dashboard
   - Understand agent execution flow and tool calls

3. **Experiment with Prompts** (30 min)
   - Modify agent prompts in `backend/main.py`
   - Change tool descriptions
   - See how it affects outputs

### 🚀 Intermediate Path
1. **Enable Advanced Features** (20 min)
   - Set `ENABLE_RAG=1` to use vector search
   - Add `SERPAPI_API_KEY` for real-time web search
   - Compare results with/without these features

2. **Add Custom Data** (45 min)
   - Add your own city to `backend/data/local_guides.json`
   - Test RAG retrieval with your data
   - Understand fallback strategies

3. **Create a New Tool** (1 hour)
   - Add a new tool (e.g., `restaurant_finder`)
   - Integrate it into an agent
   - Test and trace the new tool calls

### 💪 Advanced Path
1. **Change the Domain** (2-3 hours)
   - Use Cursor AI to help transform the system
   - Example: Change from "trip planner" to "PRD generator"
   - Modify state, agents, and tools for your use case

2. **Add a New Agent** (2 hours)
   - Create a 5th agent (e.g., "activities planner")
   - Update the LangGraph workflow
   - Test parallel vs sequential execution

3. **Implement Evaluations** (2 hours)
   - Use `test scripts/synthetic_data_gen.py` as a base
   - Create evaluation criteria for your domain
   - Set up automated evals in Arize

## Common Use Cases (Built by Students)

Students have successfully adapted this codebase for:

- **📝 PR Description Generator**
  - Agents: Code Analyzer, Context Gatherer, Description Writer
  - Replaces travel tools with GitHub API calls
  - Used by tech leads to auto-generate PR descriptions

- **🎯 Customer Support Analyst**
  - Agents: Ticket Classifier, Knowledge Base Search, Response Generator
  - RAG over support docs instead of local guides
  - Routes tickets and drafts responses

- **🔬 Research Assistant**
  - Agents: Web Searcher, Academic Search, Citation Manager, Synthesizer
  - Web search for papers + RAG over personal library
  - Generates research summaries with citations

- **📱 Content Planning System**
  - Agents: SEO Researcher, Social Media Planner, Blog Scheduler
  - Tools for keyword research, trend analysis
  - Creates cross-platform content calendars

- **🏗️ Architecture Review Agent**
  - Agents: Code Scanner, Pattern Detector, Best Practices Checker
  - RAG over architecture docs
  - Reviews PRs for architectural concerns

**💡 Your Turn**: Use Cursor AI to help you adapt this system for your domain!

## Quickstart

1) Requirements
- Python 3.10+ (Docker optional)

2) Configure environment
- Copy `backend/.env.example` to `backend/.env`.
- Set one LLM key: `OPENAI_API_KEY=...` or `OPENROUTER_API_KEY=...`.
- Optional: `ARIZE_SPACE_ID` and `ARIZE_API_KEY` for tracing.

3) Install dependencies
```bash
cd backend
uv pip install -r requirements.txt   # faster, deterministic installs
# If uv is not installed: curl -LsSf https://astral.sh/uv/install.sh | sh
# Fallback: pip install -r requirements.txt
```

4) Run
```bash
# make sure you are back in the root directory of ai-trip-planner
cd ..
./start.sh                      # starts backend on 8000; serves minimal UI at '/'
# or
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

5) Open
- Frontend: http://localhost:3000
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
 - Minimal UI: http://localhost:8000/

Docker (optional)
```bash
docker-compose up --build
```

## Project Structure
- `backend/`: FastAPI app (`main.py`), LangGraph agents, tracing hooks.
- `frontend/index.html`: Minimal static UI served by backend at `/`.
- `optional/airtable/`: Airtable integration (optional, not on critical path).
- `test scripts/`: `test_api.py`, `synthetic_data_gen.py` for quick checks/evals.
- Root: `start.sh`, `docker-compose.yml`, `README.md`.

## Development Commands
- Backend (dev): `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
- API smoke test: `python "test scripts"/test_api.py`
- Synthetic evals: `python "test scripts"/synthetic_data_gen.py --base-url http://localhost:8000 --count 12`

## API
- POST `/plan-trip` → returns a generated itinerary.
  Example body:
  ```json
  {"destination":"Tokyo, Japan","duration":"7 days","budget":"$2000","interests":"food, culture"}
  ```
- GET `/health` → simple status.

## Notes on Tracing (Optional)
- If `ARIZE_SPACE_ID` and `ARIZE_API_KEY` are set, OpenInference exports spans for agents/tools/LLM calls. View at https://app.arize.com.

## Optional Features

### RAG: Vector Search for Local Guides

The local agent can use vector search to retrieve curated local experiences from a database of 90+ real-world recommendations:

- **Enable**: Set `ENABLE_RAG=1` in your `.env` file
- **Requirements**: Requires `OPENAI_API_KEY` for embeddings
- **Data**: Uses curated experiences from `backend/data/local_guides.json`
- **Benefits**: Provides grounded, cited recommendations with sources
- **Learning**: Great example of production RAG patterns with fallback strategies

When disabled (default), the local agent uses LLM-generated responses.

See `RAG.md` for detailed documentation.

### Web Search: Real-Time Tool Data

Tools can call real web search APIs (SerpAPI) for up-to-date information:

- **Enable**: Add `SERPAPI_API_KEY` to your `.env` file
- **Benefits**: Real-time data for artist profiles, social links, etc.
- **Fallback**: Without API keys, tools automatically fall back to LLM-generated responses

## Next Steps

1. **🎯 Start Simple**: Get it running, make some requests, view traces
2. **🔍 Explore Code**: Read through `backend/main.py` to understand patterns
3. **🛠️ Modify Prompts**: Change agent behaviors to see what happens
4. **🚀 Enable Features**: Try RAG and web search
5. **💡 Build Your Own**: Use Cursor to transform it into your agent system

## Troubleshooting

- **401/empty results**: Verify `OPENAI_API_KEY` or `OPENROUTER_API_KEY` in `backend/.env`
- **No traces**: Ensure Arize credentials are set and reachable
- **Port conflicts**: Stop existing services on 3000/8000 or change ports
- **RAG not working**: Check `ENABLE_RAG=1` and `OPENAI_API_KEY` are both set
- **Slow responses**: Web search APIs may timeout; LLM fallback will handle it

## Deploy on Render
- This repo includes `render.yaml`. Connect your GitHub repo in Render and deploy as a Web Service.
- Render will run: `pip install -r backend/requirements.txt` and `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Set `OPENAI_API_KEY` (or `OPENROUTER_API_KEY`) and optional Arize vars in the Render dashboard.
