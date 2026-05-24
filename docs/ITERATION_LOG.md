# IndiaVC — 6-Week Iteration Log

A record of architectural decisions, test counts, and key changes across each week of the build.

---

## Week 1 — Foundation

**Goal:** Minimal working API that can answer a startup research question.

**What was built:**
- FastAPI scaffold with `/health` and `/research` endpoints
- Single `ResearcherAgent` using OpenAI function-calling
- MCP client (stdio transport) connecting to `mcp-duckduckgo` via `npx`
- Pydantic Settings v2 for config (`OPENAI_API_KEY`, `LOG_LEVEL`)
- structlog JSON logging

**Tests:** 7

**Key decisions:**
- Used `mcp` Python SDK for stdio transport rather than calling DuckDuckGo directly — sets up the MCP pattern for the custom server in Week 5.
- Fail-open on tool errors: agent continues with partial results rather than hard-failing.

**Known gap:** MCP stdio transport is unreliable on Windows; deferred fix to Week 2.

---

## Week 2 — Multi-Agent Orchestration

**Goal:** Replace single agent with a pipeline of specialized agents.

**What was built:**
- `Orchestrator` class coordinating 5 agents
- `WebResearcherAgent`, `NewsAnalyzerAgent`, `CompetitorAnalyzerAgent` running in parallel (Phase 1)
- `CriticAgent` reviewing Phase 1 outputs (Phase 2)
- `WriterAgent` producing the final Markdown investor brief
- HTTP-based web search fallback (DuckDuckGo DDGS library) replacing unreliable stdio MCP on Windows

**Tests:** 18 (added 11 agent/orchestrator tests)

**Key decisions:**
- `asyncio.gather` for Phase 1 parallelism — cut research time from ~3× sequential to ~1×.
- Orchestrator uses a `_task_for(agent, question)` helper to construct per-agent prompts; keeps orchestration logic separate from agent logic.
- Critic receives all three Phase 1 outputs as context, not just a concatenated string.

---

## Week 3 — LLM Router + Guardrails + HITL

**Goal:** Add safety checks, tiered model routing, and human approval before final brief.

**What was built:**
- `Guardrails` class: gpt-4o-mini safety classifier, fail-open on error
- LLM router: `HEAVY` (gpt-4o) for Critic/Orchestrator; `MEDIUM`/`LIGHT` (gpt-4o-mini) for workers
- HITL flow: `/research/stream` SSE endpoint pauses at `awaiting_approval`; `/research/approve` and `/research/reject` endpoints
- In-memory session store keyed by UUID

**Tests:** 25 (added 7 guardrails + stream tests)

**Key decisions:**
- Fail-open on guardrails error — availability preferred over false refusals. The LLM check is a UX gate, not a hard security boundary.
- HITL pause is after the Critic, not after Phase 1 — gives the human the critic's assessment to review, not raw research dumps.
- SSE `event: awaiting_approval` carries `session_id` so the frontend can wire approval directly.

---

## Week 4 — Frontend Dashboard

**Goal:** Build a Next.js UI so non-technical users can run research without cURL.

**What was built:**
- Next.js 14 app in `ui/` directory
- Research input form with streaming SSE progress display
- Agent progress cards (started → completed with output preview)
- Critic review display with approve/reject buttons
- Final brief rendered as Markdown

**Tests:** 25 (no new backend tests; frontend uses manual browser testing)

**Key decisions:**
- `ui/` kept separate from backend root so HF Spaces deployment only needs the Python backend.
- `NEXT_PUBLIC_API_URL` env var for backend URL — defaults to `localhost:8000` for local dev, set to HF Space URL for production.

---

## Week 5 — Custom MCP Server + Evaluation Framework

**Goal:** Add proprietary data source and measure brief quality systematically.

**What was built:**
- `StartupDataStore`: 25 Indian startups with sector, founders, description, funding rounds, competitors, controversies
- Custom MCP server exposing `lookup_company`, `lookup_funding`, `lookup_competitors` tools
- `MCPClient` hybrid backend: in-process custom tools + DuckDuckGo search
- Windows workaround: stdio transport fails on Windows; used in-process calls to `StartupDataStore` instead of spawning subprocess
- `scripts/evaluate_agents.py`: gpt-4o-mini as judge, 5 scoring dimensions
- `data/eval/v1_companies.json`: 10 test cases (well-known, comparison, lesser-known, harmful, off-topic)

**Tests:** 51 (added 18 MCP tests + 8 eval framework tests)

**Key decisions:**
- In-process MCP calls avoid the `mcp` stdio subprocess issue on Windows while preserving the MCP tool-calling interface for when stdio is available on Linux/Docker.
- gpt-4o-mini as judge is cheaper than gpt-4o for bulk evaluation; scores correlate well with manual review.
- Evaluation dataset includes a `harmful_refused` category to verify guardrails correctness end-to-end.

**Dependency issue resolved:** `mcp>=1.0.0` installs `sse-starlette 3.4.4` which conflicts with FastAPI's `starlette<0.47.0`. Fixed by pinning `starlette==0.46.2` after install.

---

## Week 6 — Production Hardening + Deployment

**Goal:** Fix guardrails false positive, deploy to Hugging Face Spaces + Vercel, polish documentation.

**What was built:**
- Guardrails prompt broadened: explicitly includes gaming, fantasy sports, all Indian business sectors (was too narrow; refused Dream11)
- Dockerfile port changed from 8000 → 7860 (HF Spaces requirement)
- CORS updated to include Vercel domain + regex for preview deployments
- `.dockerignore` to keep Docker image lean
- `deploy-clean` branch: excludes `ui/` so HF Space only receives the Python backend
- HF YAML front-matter added to README.md (`sdk: docker, app_port: 7860`)
- Full README rewrite: eval results table, architecture diagram, 17 agentic concepts
- `ui/vercel.json` for Vercel build config
- `.env.example` updated with `MCP_BACKEND=hybrid`

**Tests:** 52 (added `test_guardrails_allows_gaming_company_research`)

**Key decisions:**
- `deploy-clean` branch (not a separate repo) — keeps deployment history traceable to main branch commits.
- Fail-open guardrails confirmed correct: gaming/fantasy sports companies (Dream11, MPL) are valid Indian businesses and should never be refused.
- README YAML front-matter is required by HF Spaces to recognize the Space type; without it, HF treats the repo as a static app.

---

## Summary

| Week | Focus | Tests |
|---|---|---|
| 1 | FastAPI + MCP + single agent | 7 |
| 2 | Multi-agent orchestration | 18 |
| 3 | Guardrails + HITL + LLM router | 25 |
| 4 | Next.js frontend | 25 |
| 5 | Custom MCP server + eval framework | 51 |
| 6 | Deployment + polish | 52 |
