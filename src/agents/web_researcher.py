from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.router import TaskComplexity
from src.utils.logger import get_logger

logger = get_logger(__name__)

WEB_RESEARCHER_SYSTEM_PROMPT = """You are a Web Research Agent specialized in the Indian startup ecosystem. Your job: gather factual information about a company including:

- Founders and founding year
- Business model and product
- Recent funding rounds (with amounts and dates)
- Current valuation if available
- Headquarters and team size

Use the search tool to find current information. Cite every claim with the source URL.
Be concise. Stick to verifiable facts. If information is uncertain, say so.
Output format: a structured summary with sources at the end."""


class WebResearcherAgent(BaseAgent):
    name = "web_researcher"
    role = "Researches general company information and funding history"
    complexity = TaskComplexity.LIGHT

    async def execute(
        self, task: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        messages = self._build_messages(WEB_RESEARCHER_SYSTEM_PROMPT, task, context)
        answer, sources, tool_calls = await self._run_tool_loop(messages)
        logger.info("web_researcher_done", tool_calls=tool_calls, sources=len(sources))
        return {"output": answer, "sources": sources, "tool_calls": tool_calls}
