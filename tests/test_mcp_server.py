"""Tests for the custom MCP server data store."""
from __future__ import annotations

import pytest

from src.mcp.server import StartupDataStore


@pytest.fixture
def store() -> StartupDataStore:
    return StartupDataStore()


def test_data_store_loads(store: StartupDataStore) -> None:
    assert len(store._companies) >= 20


def test_find_company_exact_match(store: StartupDataStore) -> None:
    company = store.find_company("Razorpay")
    assert company is not None
    assert company["name"] == "Razorpay"


def test_find_company_case_insensitive(store: StartupDataStore) -> None:
    company = store.find_company("razorpay")
    assert company is not None
    assert company["name"] == "Razorpay"


def test_find_company_alias(store: StartupDataStore) -> None:
    company = store.find_company("razor pay")
    assert company is not None
    assert company["name"] == "Razorpay"


def test_find_company_not_found(store: StartupDataStore) -> None:
    company = store.find_company("NonexistentCompany12345")
    assert company is None


def test_find_company_empty_string(store: StartupDataStore) -> None:
    company = store.find_company("")
    assert company is None


def test_find_competitors_returns_same_sector(store: StartupDataStore) -> None:
    comps = store.find_competitors("fintech", exclude="Razorpay")
    assert len(comps) > 0
    for c in comps:
        assert c["sector"] == "fintech"
        assert c["name"] != "Razorpay"


def test_find_competitors_excludes_target(store: StartupDataStore) -> None:
    comps = store.find_competitors("edtech", exclude="Byju's")
    names = [c["name"] for c in comps]
    assert "Byju's" not in names


def test_find_competitors_unknown_sector(store: StartupDataStore) -> None:
    comps = store.find_competitors("unknownsector999")
    assert comps == []


def test_all_companies_have_required_fields(store: StartupDataStore) -> None:
    required = {"name", "sector", "description", "founders"}
    for company in store._companies:
        for field in required:
            assert field in company, f"{company.get('name')} missing field: {field}"


def test_find_company_partial_match(store: StartupDataStore) -> None:
    company = store.find_company("PharmEasy")
    assert company is not None
    assert "health" in company["sector"]


def test_list_companies_returns_all(store: StartupDataStore) -> None:
    listing = store.list_companies()
    assert len(listing) == len(store._companies)
    for entry in listing:
        assert "name" in entry
        assert "sector" in entry


# ── MCPClient custom/hybrid backend integration ───────────────────────────────

@pytest.mark.asyncio
async def test_mcp_client_custom_lookup_company() -> None:
    from src.mcp.client import MCPClient
    import json

    client = MCPClient(backend="custom")
    await client.connect()
    result = await client.call_tool("lookup_company", {"company_name": "Razorpay"})
    data = json.loads(result)
    assert data["name"] == "Razorpay"
    assert "founders" in data
    await client.disconnect()


@pytest.mark.asyncio
async def test_mcp_client_custom_lookup_funding() -> None:
    from src.mcp.client import MCPClient
    import json

    client = MCPClient(backend="custom")
    await client.connect()
    result = await client.call_tool("lookup_funding", {"company_name": "Zomato"})
    data = json.loads(result)
    assert data["company"] == "Zomato"
    assert "rounds" in data
    await client.disconnect()


@pytest.mark.asyncio
async def test_mcp_client_custom_lookup_competitors() -> None:
    from src.mcp.client import MCPClient
    import json

    client = MCPClient(backend="custom")
    await client.connect()
    result = await client.call_tool("lookup_competitors", {"company_name": "Razorpay"})
    data = json.loads(result)
    assert data["company"] == "Razorpay"
    assert "known_competitors" in data
    assert "other_companies_in_sector" in data
    await client.disconnect()


@pytest.mark.asyncio
async def test_mcp_client_custom_not_found() -> None:
    from src.mcp.client import MCPClient
    import json

    client = MCPClient(backend="custom")
    await client.connect()
    result = await client.call_tool("lookup_company", {"company_name": "NonexistentXYZ"})
    data = json.loads(result)
    assert "error" in data
    await client.disconnect()


def test_mcp_client_hybrid_lists_all_tools() -> None:
    from src.mcp.client import MCPClient

    client = MCPClient(backend="hybrid")
    tools = client.list_tools()
    names = {t["name"] for t in tools}
    assert "duckduckgo_search" in names
    assert "lookup_company" in names
    assert "lookup_competitors" in names


def test_mcp_client_ddgs_lists_only_search() -> None:
    from src.mcp.client import MCPClient

    client = MCPClient(backend="ddgs")
    tools = client.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "duckduckgo_search"
