"""Unit tests for individual agents."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _fake_completion(content: str) -> MagicMock:
    """Build a mock OpenAI chat completion that returns content with no tool calls."""
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = content
    msg.model_dump = MagicMock(return_value={"role": "assistant", "content": content})
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _fake_tool_completion(tool_name: str, args: dict, call_id: str = "call_1") -> MagicMock:
    """Build a mock completion that requests a single tool call."""
    import json

    tc = MagicMock()
    tc.id = call_id
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(args)

    msg = MagicMock()
    msg.tool_calls = [tc]
    msg.content = None
    msg.model_dump = MagicMock(return_value={"role": "assistant", "tool_calls": []})

    choice = MagicMock()
    choice.message = msg

    completion = MagicMock()
    completion.choices = [choice]
    return completion


# ── test 1: writer synthesizes from context ───────────────────────────────────

@pytest.mark.asyncio
async def test_writer_agent_synthesizes_from_context() -> None:
    from src.agents.writer import WriterAgent

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_fake_completion(
            "INVESTOR BRIEF — Razorpay\n## Company Overview\nRazorpay is a fintech. https://razorpay.com"
        )
    )

    agent = WriterAgent(openai_client=mock_openai, model="gpt-4o-mini")
    context = {
        "web_researcher": {
            "output": "Razorpay founded 2014, raised $740M.",
            "sources": ["https://crunchbase.com/razorpay"],
            "tool_calls": 2,
        },
        "news_analyzer": {
            "output": "No major issues found.",
            "sources": [],
            "tool_calls": 1,
        },
        "competitor_analyzer": {
            "output": "Competitors: PayU, Cashfree.",
            "sources": ["https://payu.in"],
            "tool_calls": 1,
        },
    }

    result = await agent.execute(task="Produce investor brief for Razorpay", context=context)

    assert "output" in result
    assert result["tool_calls"] == 0  # writer never calls tools
    assert isinstance(result["sources"], list)
    # Sources from context agents should be aggregated
    assert "https://crunchbase.com/razorpay" in result["sources"]
    assert "https://payu.in" in result["sources"]
    mock_openai.chat.completions.create.assert_called_once()


# ── test 2: web_researcher calls MCP tool ─────────────────────────────────────

@pytest.mark.asyncio
async def test_web_researcher_uses_mcp_tool() -> None:
    from src.agents.web_researcher import WebResearcherAgent

    mock_mcp = MagicMock()
    mock_mcp.list_tools = MagicMock(
        return_value=[
            {
                "name": "duckduckgo_search",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
    )
    mock_mcp.call_tool = AsyncMock(return_value="Title: Razorpay\nURL: https://razorpay.com\nBody: fintech\n")

    mock_openai = MagicMock()
    # First call returns a tool call; second returns the final answer
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[
            _fake_tool_completion("duckduckgo_search", {"query": "Razorpay funding"}),
            _fake_completion("Razorpay raised $740M. https://razorpay.com"),
        ]
    )

    agent = WebResearcherAgent(openai_client=mock_openai, model="gpt-4o-mini", mcp_client=mock_mcp)
    result = await agent.execute("Research Razorpay")

    assert result["tool_calls"] == 1
    mock_mcp.call_tool.assert_called_once_with("duckduckgo_search", {"query": "Razorpay funding"})
    assert "output" in result


# ── test 3: news_analyzer calls MCP tool ─────────────────────────────────────

@pytest.mark.asyncio
async def test_news_analyzer_uses_mcp_tool() -> None:
    from src.agents.news_analyzer import NewsAnalyzerAgent

    mock_mcp = MagicMock()
    mock_mcp.list_tools = MagicMock(
        return_value=[
            {
                "name": "duckduckgo_search",
                "description": "Search",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ]
    )
    mock_mcp.call_tool = AsyncMock(return_value="Title: Razorpay layoffs\nURL: https://news.com\nBody: some news\n")

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[
            _fake_tool_completion("duckduckgo_search", {"query": "Razorpay controversies"}),
            _fake_completion("No significant controversies. https://news.com"),
        ]
    )

    agent = NewsAnalyzerAgent(openai_client=mock_openai, model="gpt-4o-mini", mcp_client=mock_mcp)
    result = await agent.execute("Investigate Razorpay reputation")

    assert result["tool_calls"] == 1
    mock_mcp.call_tool.assert_called_once()
    assert "output" in result


# ── test 4: competitor_analyzer calls MCP tool ───────────────────────────────

@pytest.mark.asyncio
async def test_competitor_analyzer_uses_mcp_tool() -> None:
    from src.agents.competitor_analyzer import CompetitorAnalyzerAgent

    mock_mcp = MagicMock()
    mock_mcp.list_tools = MagicMock(
        return_value=[
            {
                "name": "duckduckgo_search",
                "description": "Search",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ]
    )
    mock_mcp.call_tool = AsyncMock(return_value="Title: PayU India\nURL: https://payu.in\nBody: payment gateway\n")

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[
            _fake_tool_completion("duckduckgo_search", {"query": "Razorpay competitors"}),
            _fake_completion("Main competitors: PayU, Cashfree. https://payu.in"),
        ]
    )

    agent = CompetitorAnalyzerAgent(openai_client=mock_openai, model="gpt-4o-mini", mcp_client=mock_mcp)
    result = await agent.execute("Analyze competitors of Razorpay")

    assert result["tool_calls"] == 1
    mock_mcp.call_tool.assert_called_once()
    assert "output" in result
