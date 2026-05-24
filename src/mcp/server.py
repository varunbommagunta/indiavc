"""IndiaVC Custom MCP Server

Exposes structured Indian startup data via the MCP protocol (stdio transport).

Run standalone for testing:
    python -m src.mcp.server

Or import StartupDataStore directly for in-process use (Windows stdio workaround).
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "data" / "indian_startups.json"


class StartupDataStore:
    """Loads and queries the curated Indian startup dataset."""

    def __init__(self, data_path: Path = DATA_PATH) -> None:
        with open(data_path, encoding="utf-8") as f:
            self._companies: list[dict[str, Any]] = json.load(f)
        logger.info("Loaded %d companies from %s", len(self._companies), data_path)

    def find_company(self, query: str) -> dict[str, Any] | None:
        """Find a company by name or alias. Case-insensitive, with partial match fallback."""
        q = query.lower().strip()
        if not q:
            return None
        # Exact name match
        for company in self._companies:
            if company["name"].lower() == q:
                return company
        # Alias match
        for company in self._companies:
            for alias in company.get("aliases", []):
                if alias.lower() == q:
                    return company
        # Partial match (company name contains query or query contains company name)
        for company in self._companies:
            name_lower = company["name"].lower()
            if q in name_lower or name_lower in q:
                return company
        return None

    def find_competitors(self, sector: str, exclude: str = "") -> list[dict[str, Any]]:
        """Return companies in the same sector, excluding the named company."""
        s = sector.lower().strip()
        e = exclude.lower().strip()
        return [
            {
                "name": c["name"],
                "description": c["description"],
                "sector": c["sector"],
            }
            for c in self._companies
            if c["sector"].lower() == s and c["name"].lower() != e
        ]

    def list_companies(self) -> list[dict[str, Any]]:
        """Return lightweight index of all companies."""
        return [
            {"name": c["name"], "sector": c["sector"], "founded": c.get("founded")}
            for c in self._companies
        ]


# ── MCP server setup ──────────────────────────────────────────────────────────

server: Server = Server("indiavc-startup-data")
store: StartupDataStore = StartupDataStore()


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="lookup_company",
            description=(
                "Look up structured data about an Indian startup by name. "
                "Returns founders, founding year, headquarters, valuation, funding history, "
                "competitors, and notable facts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The company name to look up, e.g. 'Razorpay' or 'PhonePe'",
                    }
                },
                "required": ["company_name"],
            },
        ),
        types.Tool(
            name="lookup_funding",
            description=(
                "Get detailed funding round history and investor information for an Indian startup."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Company name",
                    }
                },
                "required": ["company_name"],
            },
        ),
        types.Tool(
            name="lookup_competitors",
            description=(
                "Find competitors of an Indian startup based on sector. "
                "Returns both the company's known competitors and other companies in the same sector."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Company name to find competitors for",
                    }
                },
                "required": ["company_name"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(
    name: str, arguments: dict[str, Any]
) -> list[types.TextContent]:
    if name == "lookup_company":
        company = store.find_company(arguments.get("company_name", ""))
        if not company:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "Company not found in dataset", "query": arguments.get("company_name")}),
            )]
        return [types.TextContent(type="text", text=json.dumps(company, indent=2))]

    elif name == "lookup_funding":
        company = store.find_company(arguments.get("company_name", ""))
        if not company:
            return [types.TextContent(type="text", text=json.dumps({"error": "Company not found"}))]
        funding_data = {
            "company": company["name"],
            "total_funding_usd_million": company.get("total_funding_usd_million"),
            "current_valuation_usd_billion": company.get("valuation_usd_billion"),
            "valuation_year": company.get("valuation_year"),
            "rounds": company.get("funding_rounds", []),
            "key_investors": company.get("key_investors", []),
        }
        return [types.TextContent(type="text", text=json.dumps(funding_data, indent=2))]

    elif name == "lookup_competitors":
        company = store.find_company(arguments.get("company_name", ""))
        if not company:
            return [types.TextContent(type="text", text=json.dumps({"error": "Company not found"}))]
        competitors = store.find_competitors(company["sector"], exclude=company["name"])
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "company": company["name"],
                "sector": company["sector"],
                "known_competitors": company.get("competitors", []),
                "other_companies_in_sector": competitors,
            }, indent=2),
        )]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="indiavc-startup-data",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
