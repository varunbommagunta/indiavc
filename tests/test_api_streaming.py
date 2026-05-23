"""Tests for streaming and HITL endpoints."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


def _wire_state(app, *, guardrails=None, orchestrator=None, session_store=None):
    if guardrails is not None:
        app.state.guardrails = guardrails
    if orchestrator is not None:
        app.state.orchestrator = orchestrator
    if session_store is not None:
        app.state.session_store = session_store


def _allow_guardrails():
    g = MagicMock()
    g.check = AsyncMock(return_value={"decision": "allow", "reason": "ok"})
    return g


def _refuse_guardrails():
    g = MagicMock()
    g.check = AsyncMock(return_value={"decision": "refuse", "reason": "harmful query"})
    return g


# ── approve endpoint ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_approve_endpoint_requires_valid_session() -> None:
    from api.main import app

    mock_orchestrator = MagicMock()
    app.state.session_store = {}
    app.state.orchestrator = mock_orchestrator

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/research/approve", json={"session_id": "nonexistent-session-id"}
        )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_approve_endpoint_returns_brief() -> None:
    from api.main import app

    writer_mock = MagicMock()
    writer_mock.execute = AsyncMock(
        return_value={"output": "Final brief text.", "sources": [], "tool_calls": 0}
    )
    mock_orchestrator = MagicMock()
    mock_orchestrator._agents = {"writer": writer_mock}

    session_id = "test-session-123"
    session_store = {
        session_id: {
            "question": "Razorpay",
            "agent_outputs": {
                "web_researcher": {"output": "web", "sources": [], "tool_calls": 1},
                "critic": {"output": "CRITIC ASSESSMENT", "sources": [], "tool_calls": 0},
            },
            "status": "awaiting_approval",
        }
    }

    app.state.session_store = session_store
    app.state.orchestrator = mock_orchestrator

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/research/approve", json={"session_id": session_id})

    assert response.status_code == 200
    data = response.json()
    assert data["brief"] == "Final brief text."
    assert session_store[session_id]["status"] == "completed"


# ── reject endpoint ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reject_endpoint_clears_session() -> None:
    from api.main import app

    session_id = "to-be-deleted"
    session_store = {
        session_id: {"question": "test", "agent_outputs": {}, "status": "awaiting_approval"}
    }
    app.state.session_store = session_store

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/research/reject", json={"session_id": session_id})

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert session_id not in session_store


@pytest.mark.asyncio
async def test_reject_endpoint_tolerates_missing_session() -> None:
    from api.main import app

    app.state.session_store = {}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/research/reject", json={"session_id": "ghost-session"}
        )

    assert response.status_code == 200
