"""FastAPI application entry point for IndiaVC research system."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI

from api.schemas import AgentOutput, ResearchRequest, ResearchResponse
from config.settings import settings
from src.agents.competitor_analyzer import CompetitorAnalyzerAgent
from src.agents.news_analyzer import NewsAnalyzerAgent
from src.agents.orchestrator import Orchestrator
from src.agents.web_researcher import WebResearcherAgent
from src.agents.writer import WriterAgent
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
    model = settings.worker_model

    orchestrator = Orchestrator(
        agents={
            "web_researcher": WebResearcherAgent(openai_client, model, mcp_client),
            "news_analyzer": NewsAnalyzerAgent(openai_client, model, mcp_client),
            "competitor_analyzer": CompetitorAnalyzerAgent(openai_client, model, mcp_client),
            "writer": WriterAgent(openai_client, model),
        }
    )

    app.state.mcp_client = mcp_client
    app.state.openai_client = openai_client
    app.state.orchestrator = orchestrator
    logger.info("startup_complete", tools=mcp_client.list_tools())

    yield

    logger.info("shutdown_begin")
    await mcp_client.disconnect()
    logger.info("shutdown_complete")


app = FastAPI(
    title="IndiaVC Research API",
    description="Multi-agent AI system for Indian startup due diligence",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.2.0"}


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest, app_request: Request) -> ResearchResponse:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    orchestrator: Orchestrator = app_request.app.state.orchestrator
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
