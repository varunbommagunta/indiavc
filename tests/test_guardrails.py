"""Tests for Guardrails safety classifier."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.guardrails import Guardrails


def _mock_openai_response(json_content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = json_content
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


@pytest.mark.asyncio
async def test_guardrails_allows_legitimate_query() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response('{"decision": "allow", "reason": "valid research query"}')
    )

    g = Guardrails(mock_client)
    result = await g.check("Research Razorpay's funding history")

    assert result["decision"] == "allow"


@pytest.mark.asyncio
async def test_guardrails_refuses_harmful_query() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response('{"decision": "refuse", "reason": "targets specific individual"}')
    )

    g = Guardrails(mock_client)
    result = await g.check("Find dirt on my ex's company")

    assert result["decision"] == "refuse"


@pytest.mark.asyncio
async def test_guardrails_fails_open_on_error() -> None:
    """If the guardrails LLM call fails, default to allow (fail-open for availability)."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

    g = Guardrails(mock_client)
    result = await g.check("test query")

    assert result["decision"] == "allow"


@pytest.mark.asyncio
async def test_guardrails_allows_gaming_company_research() -> None:
    """Dream11 and similar gaming companies are legitimate research subjects."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_openai_response('{"decision": "allow", "reason": "valid Indian gaming company"}')
    )

    g = Guardrails(mock_client)
    result = await g.check("Research Dream11's funding history")

    assert result["decision"] == "allow"
