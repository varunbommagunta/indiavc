"""
MCPClient with multiple backends:

  'ddgs'   — DuckDuckGo HTTP search (original Week 1 fallback)
  'custom' — IndiaVC curated startup data (in-process, Windows stdio workaround)
  'hybrid' — Both: structured lookup first, web search for current news (default)

Windows note: MCP stdio subprocess transport fails on this host (process closes
before initialize() completes). The 'custom' and 'hybrid' backends therefore
call StartupDataStore directly instead of spawning a subprocess.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from ddgs import DDGS

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── tool descriptors ──────────────────────────────────────────────────────────
# All use `input_schema` key to match BaseAgent._build_tools() expectations.

_SEARCH_TOOL: dict[str, Any] = {
    "name": "duckduckgo_search",
    "description": (
        "Search the web using DuckDuckGo. "
        "Returns a list of results with title, URL, and snippet."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

_LOOKUP_COMPANY_TOOL: dict[str, Any] = {
    "name": "lookup_company",
    "description": (
        "Look up structured data about an Indian startup by name. "
        "Returns founders, founding year, headquarters, valuation, funding history, "
        "and notable facts from a curated dataset. Use this FIRST before web search."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "company_name": {
                "type": "string",
                "description": "The company name, e.g. 'Razorpay' or 'PhonePe'",
            }
        },
        "required": ["company_name"],
    },
}

_LOOKUP_FUNDING_TOOL: dict[str, Any] = {
    "name": "lookup_funding",
    "description": (
        "Get detailed funding round history and investor information for an Indian startup "
        "from a curated dataset."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "company_name": {"type": "string", "description": "Company name"}
        },
        "required": ["company_name"],
    },
}

_LOOKUP_COMPETITORS_TOOL: dict[str, Any] = {
    "name": "lookup_competitors",
    "description": (
        "Find competitors of an Indian startup based on sector. "
        "Returns known competitors and other companies in the same sector. "
        "Use this FIRST for competitor analysis."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "company_name": {
                "type": "string",
                "description": "Company name to find competitors for",
            }
        },
        "required": ["company_name"],
    },
}

_DDGS_TOOLS = [_SEARCH_TOOL]
_CUSTOM_TOOLS = [_LOOKUP_COMPANY_TOOL, _LOOKUP_FUNDING_TOOL, _LOOKUP_COMPETITORS_TOOL]
_HYBRID_TOOLS = [_LOOKUP_COMPANY_TOOL, _LOOKUP_FUNDING_TOOL, _LOOKUP_COMPETITORS_TOOL, _SEARCH_TOOL]


class MCPClientError(Exception):
    pass


class MCPClient:
    """
    Unified MCPClient supporting ddgs, custom, and hybrid backends.

    The interface (connect / list_tools / call_tool / disconnect) is identical
    across backends so agents require no changes.
    """

    def __init__(self, backend: str = "ddgs") -> None:
        self._backend = backend
        self._connected = False
        self._store = None  # StartupDataStore, populated for custom/hybrid

    async def connect(self) -> None:
        if self._backend in ("custom", "hybrid"):
            from src.mcp.server import StartupDataStore
            self._store = StartupDataStore()
            logger.info("startup_data_loaded", companies=len(self._store._companies))
        self._connected = True
        logger.info("mcp_client_connected", backend=self._backend)

    async def disconnect(self) -> None:
        self._connected = False
        self._store = None
        logger.info("mcp_client_disconnected")

    def list_tools(self) -> list[dict[str, Any]]:
        if self._backend == "custom":
            return _CUSTOM_TOOLS
        if self._backend == "hybrid":
            return _HYBRID_TOOLS
        return _DDGS_TOOLS  # ddgs default

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if not self._connected:
            raise MCPClientError("Not connected. Call connect() first.")

        if name == "duckduckgo_search":
            return await self._call_ddgs(arguments)
        if name in ("lookup_company", "lookup_funding", "lookup_competitors"):
            return await self._call_custom(name, arguments)
        raise MCPClientError(f"Unknown tool: {name!r}")

    # ── ddgs backend ──────────────────────────────────────────────────────────

    async def _call_ddgs(self, arguments: dict[str, Any]) -> str:
        query: str = arguments.get("query", "")
        max_results: int = int(arguments.get("max_results", 5))

        if not query.strip():
            raise MCPClientError("'query' argument is required and must not be empty.")

        logger.info("ddgs_search", query=query, max_results=max_results)
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: list(DDGS().text(query, max_results=max_results)),
            )
        except Exception as exc:
            raise MCPClientError(f"DuckDuckGo search failed: {exc}") from exc

        if not results:
            return "No results found."

        lines: list[str] = []
        for r in results:
            lines.append(f"Title: {r.get('title', 'N/A')}")
            lines.append(f"URL:   {r.get('href', 'N/A')}")
            lines.append(f"Body:  {r.get('body', 'N/A')}")
            lines.append("")
        return "\n".join(lines)

    # ── custom (in-process) backend ───────────────────────────────────────────

    async def _call_custom(self, name: str, arguments: dict[str, Any]) -> str:
        if self._store is None:
            raise MCPClientError("Custom backend not initialized. Call connect() first.")

        if name == "lookup_company":
            company = self._store.find_company(arguments.get("company_name", ""))
            if not company:
                return json.dumps(
                    {"error": "Company not found in dataset", "query": arguments.get("company_name")}
                )
            return json.dumps(company, indent=2)

        if name == "lookup_funding":
            company = self._store.find_company(arguments.get("company_name", ""))
            if not company:
                return json.dumps({"error": "Company not found"})
            return json.dumps(
                {
                    "company": company["name"],
                    "total_funding_usd_million": company.get("total_funding_usd_million"),
                    "current_valuation_usd_billion": company.get("valuation_usd_billion"),
                    "valuation_year": company.get("valuation_year"),
                    "rounds": company.get("funding_rounds", []),
                    "key_investors": company.get("key_investors", []),
                },
                indent=2,
            )

        if name == "lookup_competitors":
            company = self._store.find_company(arguments.get("company_name", ""))
            if not company:
                return json.dumps({"error": "Company not found"})
            competitors = self._store.find_competitors(
                company["sector"], exclude=company["name"]
            )
            return json.dumps(
                {
                    "company": company["name"],
                    "sector": company["sector"],
                    "known_competitors": company.get("competitors", []),
                    "other_companies_in_sector": competitors,
                },
                indent=2,
            )

        raise MCPClientError(f"Unknown custom tool: {name!r}")

    # ── context manager support ───────────────────────────────────────────────

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()


# ── factories ─────────────────────────────────────────────────────────────────

def make_duckduckgo_client() -> MCPClient:
    """Backward-compatible factory. Returns ddgs-only client."""
    return MCPClient(backend="ddgs")


def make_mcp_client() -> MCPClient:
    """Settings-aware factory. Reads mcp_backend from config."""
    from config.settings import settings
    return MCPClient(backend=getattr(settings, "mcp_backend", "hybrid"))
