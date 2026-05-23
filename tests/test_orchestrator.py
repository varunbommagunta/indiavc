"""Unit tests for the Orchestrator."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.orchestrator import Orchestrator


def _make_agent(name: str, output: dict) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.execute = AsyncMock(return_value=output)
    return agent


# ── test 1: 3 research agents run in parallel ─────────────────────────────────

@pytest.mark.asyncio
async def test_orchestrator_runs_3_research_agents_in_parallel() -> None:
    import asyncio

    call_times: list[float] = []

    async def slow_execute(task: str, context=None) -> dict:
        import time
        call_times.append(time.monotonic())
        await asyncio.sleep(0.05)
        return {"output": "done", "sources": [], "tool_calls": 1}

    web = MagicMock()
    web.execute = slow_execute
    news = MagicMock()
    news.execute = slow_execute
    comp = MagicMock()
    comp.execute = slow_execute
    writer = _make_agent("writer", {"output": "brief", "sources": [], "tool_calls": 0})

    orch = Orchestrator({"web_researcher": web, "news_analyzer": news,
                         "competitor_analyzer": comp, "writer": writer})

    import time
    t0 = time.monotonic()
    await orch.run_research("test question")
    elapsed = time.monotonic() - t0

    # 3 agents sleeping 50ms each; if serial that's 150ms+, parallel should be ~50ms
    assert elapsed < 0.12, f"Agents appear to have run serially (took {elapsed:.2f}s)"
    assert len(call_times) == 3


# ── test 2: writer receives all research outputs as context ───────────────────

@pytest.mark.asyncio
async def test_orchestrator_passes_research_outputs_to_writer() -> None:
    web = _make_agent("web_researcher", {"output": "web output", "sources": ["https://a.com"], "tool_calls": 2})
    news = _make_agent("news_analyzer", {"output": "news output", "sources": [], "tool_calls": 1})
    comp = _make_agent("competitor_analyzer", {"output": "comp output", "sources": [], "tool_calls": 1})
    writer = _make_agent("writer", {"output": "final brief", "sources": [], "tool_calls": 0})

    orch = Orchestrator({"web_researcher": web, "news_analyzer": news,
                         "competitor_analyzer": comp, "writer": writer})
    await orch.run_research("Razorpay")

    # Writer must have been called with context containing all 3 agent outputs
    call_kwargs = writer.execute.call_args
    context_passed = call_kwargs.kwargs.get("context") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else None
    if context_passed is None and call_kwargs.kwargs:
        context_passed = call_kwargs.kwargs.get("context")

    assert context_passed is not None, "Writer was not called with a context argument"
    assert "web_researcher" in context_passed
    assert "news_analyzer" in context_passed
    assert "competitor_analyzer" in context_passed


# ── test 3: failed agent doesn't break the others ────────────────────────────

@pytest.mark.asyncio
async def test_orchestrator_handles_failed_agent_gracefully() -> None:
    web = _make_agent("web_researcher", {"output": "web output", "sources": [], "tool_calls": 1})

    # news_analyzer raises
    news = MagicMock()
    news.name = "news_analyzer"
    news.execute = AsyncMock(side_effect=RuntimeError("news API down"))

    comp = _make_agent("competitor_analyzer", {"output": "comp output", "sources": [], "tool_calls": 1})
    writer = _make_agent("writer", {"output": "brief despite failure", "sources": [], "tool_calls": 0})

    orch = Orchestrator({"web_researcher": web, "news_analyzer": news,
                         "competitor_analyzer": comp, "writer": writer})
    result = await orch.run_research("Razorpay")

    # Should still complete
    assert result["final_brief"] == "brief despite failure"
    # Failed agent output should be a stub
    assert "Agent failed" in result["agent_outputs"]["news_analyzer"]["output"]
    # Other agents should have their real outputs
    assert result["agent_outputs"]["web_researcher"]["output"] == "web output"
    assert result["agent_outputs"]["competitor_analyzer"]["output"] == "comp output"


# ── test 4: structured result shape ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_orchestrator_returns_structured_result() -> None:
    web = _make_agent("web_researcher", {"output": "web", "sources": ["https://x.com"], "tool_calls": 2})
    news = _make_agent("news_analyzer", {"output": "news", "sources": [], "tool_calls": 1})
    comp = _make_agent("competitor_analyzer", {"output": "comp", "sources": [], "tool_calls": 1})
    writer = _make_agent("writer", {"output": "the brief", "sources": [], "tool_calls": 0})

    orch = Orchestrator({"web_researcher": web, "news_analyzer": news,
                         "competitor_analyzer": comp, "writer": writer})
    result = await orch.run_research("Razorpay")

    assert "final_brief" in result
    assert "agent_outputs" in result
    assert "total_tool_calls" in result
    assert "execution_time_seconds" in result
    assert result["final_brief"] == "the brief"
    assert result["total_tool_calls"] == 4  # 2+1+1
    assert isinstance(result["execution_time_seconds"], float)
    assert set(result["agent_outputs"].keys()) == {"web_researcher", "news_analyzer", "competitor_analyzer"}
