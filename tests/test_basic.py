"""Basic sanity tests for IndiaVC Week 1/2 scaffold."""
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

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── test 3: /research with mocked orchestrator ────────────────────────────────

@pytest.mark.asyncio
async def test_research_endpoint_with_mock() -> None:
    from api.main import app

    mock_orchestrator = MagicMock()
    mock_orchestrator.run_research = AsyncMock(
        return_value={
            "final_brief": "Razorpay was founded in 2014. https://razorpay.com",
            "agent_outputs": {
                "web_researcher": {
                    "output": "Razorpay: fintech founded 2014.",
                    "sources": ["https://razorpay.com"],
                    "tool_calls": 1,
                },
                "news_analyzer": {
                    "output": "No major controversies found.",
                    "sources": [],
                    "tool_calls": 1,
                },
                "competitor_analyzer": {
                    "output": "Competitors: PayU, Cashfree.",
                    "sources": [],
                    "tool_calls": 1,
                },
            },
            "total_tool_calls": 3,
            "execution_time_seconds": 5.0,
        }
    )

    # ASGITransport skips the ASGI lifespan — wire state directly.
    app.state.orchestrator = mock_orchestrator

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/research",
                json={"question": "What is the funding history of Razorpay?"},
            )
    finally:
        del app.state.orchestrator

    assert response.status_code == 200
    data = response.json()
    assert "brief" in data
    assert "agent_outputs" in data
    assert "total_tool_calls" in data
    assert "execution_time_seconds" in data
    assert len(data["brief"]) > 0
