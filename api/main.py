"""FastAPI application entry point for IndiaVC research system."""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from api.schemas import (
    AgentOutput,
    ApproveRequest,
    RejectRequest,
    ResearchRequest,
    ResearchResponse,
)
from config.settings import settings
from src.agents.competitor_analyzer import CompetitorAnalyzerAgent
from src.agents.critic import CriticAgent
from src.agents.news_analyzer import NewsAnalyzerAgent
from src.agents.orchestrator import Orchestrator
from src.agents.web_researcher import WebResearcherAgent
from src.agents.writer import WriterAgent
from src.guardrails import Guardrails
from src.mcp.client import MCPClientError, make_duckduckgo_client
from src.utils.logger import configure_logging, get_logger

configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("startup_begin")
    mcp_client = make_duckduckgo_client()
    try:
        await mcp_client.connect()
    except MCPClientError as exc:
        logger.error("mcp_connect_failed", error=str(exc))
        raise RuntimeError(f"MCP startup failed: {exc}") from exc

    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    guardrails = Guardrails(openai_client)

    orchestrator = Orchestrator(
        agents={
            "web_researcher": WebResearcherAgent(openai_client, mcp_client=mcp_client),
            "news_analyzer": NewsAnalyzerAgent(openai_client, mcp_client=mcp_client),
            "competitor_analyzer": CompetitorAnalyzerAgent(openai_client, mcp_client=mcp_client),
            "critic": CriticAgent(openai_client),
            "writer": WriterAgent(openai_client),
        }
    )

    app.state.mcp_client = mcp_client
    app.state.openai_client = openai_client
    app.state.guardrails = guardrails
    app.state.orchestrator = orchestrator
    app.state.session_store = {}  # in-memory; Redis in Week 6
    logger.info("startup_complete", tools=mcp_client.list_tools())

    yield

    logger.info("shutdown_begin")
    await mcp_client.disconnect()
    logger.info("shutdown_complete")


app = FastAPI(
    title="IndiaVC Research API",
    description="Multi-agent AI system for Indian startup due diligence",
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.4.0"}


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest, app_request: Request) -> ResearchResponse:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    guardrails: Guardrails = app_request.app.state.guardrails
    orchestrator: Orchestrator = app_request.app.state.orchestrator

    check = await guardrails.check(request.question)
    if check["decision"] == "refuse":
        raise HTTPException(
            status_code=400,
            detail={"error": "Query refused by safety policy", "reason": check["reason"]},
        )

    try:
        result = await orchestrator.run_research(request.question)
    except Exception as exc:
        logger.error("research_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ResearchResponse(
        brief=result["final_brief"],
        agent_outputs={
            name: AgentOutput(**out)
            for name, out in result["agent_outputs"].items()
        },
        total_tool_calls=result["total_tool_calls"],
        execution_time_seconds=result["execution_time_seconds"],
    )


@app.post("/research/stream")
async def research_stream(request: ResearchRequest, app_request: Request) -> StreamingResponse:
    """
    Streams agent progress as Server-Sent Events.
    HITL flow: research agents → critic → awaiting_approval pause.
    Client then POSTs to /research/approve or /research/reject.
    """
    guardrails: Guardrails = app_request.app.state.guardrails
    orchestrator: Orchestrator = app_request.app.state.orchestrator
    session_store: dict = app_request.app.state.session_store

    check = await guardrails.check(request.question)
    if check["decision"] == "refuse":
        async def refuse_stream():
            yield _sse({"event": "refused", "reason": check["reason"]})
        return StreamingResponse(refuse_stream(), media_type="text/event-stream")

    session_id = str(uuid.uuid4())

    async def event_stream():
        yield _sse({"event": "started", "session_id": session_id, "question": request.question})

        # Phase 1 — parallel research, emit start events immediately
        agent_names = ["web_researcher", "news_analyzer", "competitor_analyzer"]
        for name in agent_names:
            yield _sse({"event": "agent_started", "agent": name})

        results = await asyncio.gather(
            *[orchestrator._agents[name].execute(orchestrator._task_for(name, request.question))
              for name in agent_names],
            return_exceptions=True,
        )

        agent_outputs: dict[str, Any] = {}
        for name, result in zip(agent_names, results):
            if isinstance(result, Exception):
                logger.error("stream_agent_failed", agent=name, error=str(result))
                agent_outputs[name] = {"output": str(result), "sources": [], "tool_calls": 0}
                yield _sse({"event": "agent_failed", "agent": name, "error": str(result)})
            else:
                agent_outputs[name] = result
                yield _sse({
                    "event": "agent_completed",
                    "agent": name,
                    "output_preview": result["output"][:300],
                    "tool_calls": result["tool_calls"],
                })

        # Phase 2 — critic
        yield _sse({"event": "agent_started", "agent": "critic"})
        critic_result = await orchestrator._agents["critic"].execute(
            task=f"Critically review research findings for: {request.question}",
            context=agent_outputs,
        )
        agent_outputs["critic"] = critic_result
        yield _sse({"event": "critic_completed", "output": critic_result["output"]})

        # HITL pause
        session_store[session_id] = {
            "question": request.question,
            "agent_outputs": agent_outputs,
            "status": "awaiting_approval",
        }
        yield _sse({
            "event": "awaiting_approval",
            "session_id": session_id,
            "message": "Review critic findings and approve to generate final brief",
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/research/approve")
async def research_approve(request: ApproveRequest, app_request: Request) -> dict:
    """Continue after HITL approval — runs writer and returns final brief."""
    session_store: dict = app_request.app.state.session_store
    orchestrator: Orchestrator = app_request.app.state.orchestrator

    session = session_store.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if session["status"] != "awaiting_approval":
        raise HTTPException(status_code=400, detail=f"Session status: {session['status']}")

    writer_result = await orchestrator._agents["writer"].execute(
        task=f"Produce an investor brief for: {session['question']}",
        context=session["agent_outputs"],
    )

    session["status"] = "completed"
    session["final_brief"] = writer_result["output"]

    return {
        "brief": writer_result["output"],
        "agent_outputs": session["agent_outputs"],
    }


@app.post("/research/reject")
async def research_reject(request: RejectRequest, app_request: Request) -> dict:
    """Reject research and discard session."""
    session_store: dict = app_request.app.state.session_store
    session_store.pop(request.session_id, None)
    return {"status": "rejected"}
