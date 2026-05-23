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


def _mock_critic() -> MagicMock:
    return _make_agent("critic", {"output": "CRITIC ASSESSMENT\nConfidence Score: 7/10", "sources": [], "tool_calls": 0})


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
    critic = _mock_critic()
    writer = _make_agent("writer", {"output": "brief", "sources": [], "tool_calls": 0})

    orch = Orchestrator({"web_researcher": web, "news_analyzer": news,
                         "competitor_analyzer": comp, "critic": critic, "writer": writer})

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
    critic = _mock_critic()
    writer = _make_agent("writer", {"output": "final brief", "sources": [], "tool_calls": 0})

    orch = Orchestrator({"web_researcher": web, "news_analyzer": news,
                         "competitor_analyzer": comp, "critic": critic, "writer": writer})
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
    critic = _mock_critic()
    writer = _make_agent("writer", {"output": "brief despite failure", "sources": [], "tool_calls": 0})

    orch = Orchestrator({"web_researcher": web, "news_analyzer": news,
                         "competitor_analyzer": comp, "critic": critic, "writer": writer})
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
    critic = _mock_critic()
    writer = _make_agent("writer", {"output": "the brief", "sources": [], "tool_calls": 0})

    orch = Orchestrator({"web_researcher": web, "news_analyzer": news,
                         "competitor_analyzer": comp, "critic": critic, "writer": writer})
    result = await orch.run_research("Razorpay")

    assert "final_brief" in result
    assert "agent_outputs" in result
    assert "total_tool_calls" in result
    assert "execution_time_seconds" in result
    assert result["final_brief"] == "the brief"
    assert result["total_tool_calls"] == 4  # 2+1+1+0(critic)
    assert isinstance(result["execution_time_seconds"], float)
    # critic is now part of agent_outputs (Phase 2 output visible in the dict)
    assert set(result["agent_outputs"].keys()) == {
        "web_researcher", "news_analyzer", "competitor_analyzer", "critic"
    }


# ── test 5: critic is called between research and writer ─────────────────────

@pytest.mark.asyncio
async def test_orchestrator_includes_critic_in_workflow() -> None:
    web = _make_agent("web_researcher", {"output": "web output", "sources": [], "tool_calls": 1})
    news = _make_agent("news_analyzer", {"output": "news output", "sources": [], "tool_calls": 1})
    comp = _make_agent("competitor_analyzer", {"output": "comp output", "sources": [], "tool_calls": 1})
    critic = _mock_critic()
    writer = _make_agent("writer", {"output": "final brief", "sources": [], "tool_calls": 0})

    orch = Orchestrator({"web_researcher": web, "news_analyzer": news,
                         "competitor_analyzer": comp, "critic": critic, "writer": writer})
    await orch.run_research("Razorpay")

    # Critic must have been called with the Phase 1 research outputs as context
    critic.execute.assert_called_once()
    critic_call = critic.execute.call_args
    critic_context = critic_call.kwargs.get("context") or (critic_call.args[1] if len(critic_call.args) > 1 else None)
    assert critic_context is not None, "Critic was not called with a context argument"
    assert "web_researcher" in critic_context
    assert "news_analyzer" in critic_context
    assert "competitor_analyzer" in critic_context

    # Writer must have been called with critic's output included in context
    writer_call = writer.execute.call_args
    writer_context = writer_call.kwargs.get("context") or (writer_call.args[1] if len(writer_call.args) > 1 else None)
    assert writer_context is not None
    assert "critic" in writer_context
