from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)

COMPETITOR_ANALYZER_SYSTEM_PROMPT = """You are a Competitor Analysis Agent. Your job: identify and analyze competitors of the target company in the Indian startup ecosystem. For each competitor found:

- Company name and what they do
- Approximate size/funding if known
- Key differentiators from the target company
- Market position (leader / challenger / niche player)

Use search tools to find competitor information.
Identify 3-5 main competitors. Don't pad the list with irrelevant companies.
Cite sources for each competitor.
Output format: ranked competitor list with brief analysis of each."""


class CompetitorAnalyzerAgent(BaseAgent):
    name = "competitor_analyzer"
    role = "Identifies and analyzes competitors"

    async def execute(
        self, task: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        messages = self._build_messages(COMPETITOR_ANALYZER_SYSTEM_PROMPT, task, context)
        answer, sources, tool_calls = await self._run_tool_loop(messages)
        logger.info("competitor_analyzer_done", tool_calls=tool_calls, sources=len(sources))
        return {"output": answer, "sources": sources, "tool_calls": tool_calls}
