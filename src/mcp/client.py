"""
Search client with the MCPClient interface.

Week 1: implemented via duckduckgo-search (HTTP) because the mcp-duckduckgo
npm package fails to complete the MCP stdio handshake on this Windows host.
Week 2: swap the DDGSSearchBackend for a real MCP stdio backend; the
ResearcherAgent and all callers need zero changes.
"""
from __future__ import annotations

import asyncio
from typing import Any

from ddgs import DDGS

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── public tool descriptor (mirrors what a real MCP server would advertise) ──

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


class MCPClientError(Exception):
    pass


class MCPClient:
    """
    Drop-in MCPClient backed by duckduckgo-search (HTTP) instead of MCP stdio.
    Exposes the same connect / list_tools / call_tool / disconnect interface
    so the rest of the system is unaware of the implementation difference.
    """

    def __init__(self) -> None:
        self._connected = False

    async def connect(self) -> None:
        self._connected = True
        logger.info("search_client_connected", backend="duckduckgo-search-http")

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("search_client_disconnected")

    def list_tools(self) -> list[dict[str, Any]]:
        return [_SEARCH_TOOL]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if not self._connected:
            raise MCPClientError("Not connected. Call connect() first.")
        if name != "duckduckgo_search":
            raise MCPClientError(f"Unknown tool: {name!r}")

        query: str = arguments.get("query", "")
        max_results: int = int(arguments.get("max_results", 5))

        if not query.strip():
            raise MCPClientError("'query' argument is required and must not be empty.")

        logger.info("ddgs_search", query=query, max_results=max_results)
        try:
            # DDGS.text is synchronous; run in thread pool to stay async-safe
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: list(DDGS().text(query, max_results=max_results)),
            )
        except Exception as exc:
            raise MCPClientError(f"DuckDuckGo search failed: {exc}") from exc

        if not results:
            return "No results found."

        lines = []
        for r in results:
            lines.append(f"Title: {r.get('title', 'N/A')}")
            lines.append(f"URL:   {r.get('href', 'N/A')}")
            lines.append(f"Body:  {r.get('body', 'N/A')}")
            lines.append("")
        return "\n".join(lines)

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()


def make_duckduckgo_client() -> MCPClient:
    """Factory used by api/main.py lifespan."""
    return MCPClient()
