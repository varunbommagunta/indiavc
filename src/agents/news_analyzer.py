from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.router import TaskComplexity
from src.utils.logger import get_logger

logger = get_logger(__name__)

NEWS_ANALYZER_SYSTEM_PROMPT = """You are a News Analysis Agent. Your job: investigate the reputation and recent news around a company. Look for:

- Layoffs or restructuring announcements
- Founder departures or leadership changes
- Legal issues or regulatory actions
- Negative press coverage
- Customer complaints or product issues
- Recent positive milestones (counterbalance)

Be objective. Report what you find, both negative and positive. Use search tools to find recent news.
Cite every claim with source URL and date when available.
Output format: structured findings with a severity assessment for each item (low / medium / high concern)."""


class NewsAnalyzerAgent(BaseAgent):
    name = "news_analyzer"
    role = "Investigates company reputation and controversies"
    complexity = TaskComplexity.LIGHT

    async def execute(
        self, task: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        messages = self._build_messages(NEWS_ANALYZER_SYSTEM_PROMPT, task, context)
        answer, sources, tool_calls = await self._run_tool_loop(messages)
        logger.info("news_analyzer_done", tool_calls=tool_calls, sources=len(sources))
        return {"output": answer, "sources": sources, "tool_calls": tool_calls}
