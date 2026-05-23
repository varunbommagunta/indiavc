"""Basic sanity tests for IndiaVC Week 1 scaffold."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


# ── test 1: settings ──────────────────────────────────────────────────────────

def test_settings_loads() -> None:
    from config.settings import settings

    assert settings.openai_api_key.startswith("sk-"), "API key should start with sk-"
    assert settings.worker_model == "gpt-4o-mini"
    assert 1 <= settings.max_agent_iterations <= 50


# ── test 2: /health endpoint ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    from api.main import app

    # ASGITransport does not trigger the lifespan scope, so /health
    # (which touches no state) works without any setup.
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── test 3: /research with mocked MCP + OpenAI ───────────────────────────────

@pytest.mark.asyncio
async def test_research_endpoint_with_mock() -> None:
    # Mock MCP client
    mock_mcp = MagicMock()
    mock_mcp.connect = AsyncMock()
    mock_mcp.disconnect = AsyncMock()
    mock_mcp.list_tools = MagicMock(return_value=[])

    # Fake OpenAI completion — returns a final answer immediately (no tool calls)
    fake_message = MagicMock()
    fake_message.tool_calls = None
    fake_message.content = (
        "Razorpay was founded in 2014 and has raised over $740M. https://razorpay.com"
    )
    fake_message.model_dump = MagicMock(
        return_value={"role": "assistant", "content": fake_message.content}
    )
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_completion = MagicMock()
    fake_completion.choices = [fake_choice]

    mock_openai_instance = MagicMock()
    mock_openai_instance.chat.completions.create = AsyncMock(return_value=fake_completion)

    from api.main import app

    # ASGITransport skips the ASGI lifespan scope, so we wire app.state directly.
    app.state.mcp_client = mock_mcp
    app.state.openai_client = mock_openai_instance

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/research",
                json={"question": "What is the funding history of Razorpay?"},
            )
    finally:
        # Clean up so other tests start fresh
        del app.state.mcp_client
        del app.state.openai_client

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["sources"], list)
    assert isinstance(data["tool_calls_made"], list)
    assert len(data["answer"]) > 0
