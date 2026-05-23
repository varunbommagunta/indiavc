"""Tests for the Critic agent."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _mock_openai_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


@pytest.mark.asyncio
async def test_critic_synthesizes_from_research_context() -> None:
    from src.agents.critic import CriticAgent

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response(
            "CRITIC ASSESSMENT\n\nConfidence Score: 7/10\nSolid research with minor gaps.\n\n"
            "Red Flags Identified\n- Customer service issues reported\n\n"
            "Bull Case\n1. Strong valuation growth\n2. IPO potential\n3. Diverse product suite\n\n"
            "Bear Case\n1. Customer complaints\n2. Leadership changes\n3. Regulatory scrutiny"
        )
    )

    critic = CriticAgent(mock_client)
    context = {
        "web_researcher": {"output": "Razorpay raised $375M in Series F", "sources": [], "tool_calls": 2},
        "news_analyzer": {"output": "Recent layoffs reported at Razorpay", "sources": [], "tool_calls": 1},
        "competitor_analyzer": {"output": "Main competitors: PayU, Cashfree", "sources": [], "tool_calls": 1},
    }

    result = await critic.execute("Research Razorpay", context=context)

    assert "CRITIC ASSESSMENT" in result["output"]
    assert result["tool_calls"] == 0
    assert isinstance(result["sources"], list)
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_critic_handles_empty_context() -> None:
    from src.agents.critic import CriticAgent

    mock_client = MagicMock()
    critic = CriticAgent(mock_client)

    result = await critic.execute("test question", context=None)

    assert "No research findings" in result["output"]
    assert result["tool_calls"] == 0
    # LLM should never be called when there's no context to review
    mock_client.chat.completions.create.assert_not_called()
