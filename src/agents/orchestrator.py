from __future__ import annotations

import asyncio
from time import time
from typing import Any

from src.agents.base import BaseAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """Coordinates the multi-agent research workflow. Not an agent — it's the conductor."""

    def __init__(self, agents: dict[str, BaseAgent]) -> None:
        self._agents = agents

    async def run_research(self, question: str) -> dict[str, Any]:
        """
        Phase 1: run web_researcher, news_analyzer, competitor_analyzer in parallel.
        Phase 2: pass all results to writer for the final investor brief.
        """
        start = time()
        logger.info("orchestrator_starting", question=question)

        # Phase 1 — parallel research
        phase1_results = await asyncio.gather(
            self._agents["web_researcher"].execute(
                f"Research general information about: {question}"
            ),
            self._agents["news_analyzer"].execute(
                f"Investigate news, controversies, and reputation of: {question}"
            ),
            self._agents["competitor_analyzer"].execute(
                f"Identify and analyze competitors of: {question}"
            ),
            return_exceptions=True,
        )

        phase1_names = ["web_researcher", "news_analyzer", "competitor_analyzer"]
        agent_outputs: dict[str, Any] = {}
        for name, result in zip(phase1_names, phase1_results):
            if isinstance(result, Exception):
                logger.error("agent_failed", agent=name, error=str(result))
                agent_outputs[name] = {
                    "output": f"[Agent failed: {result}]",
                    "sources": [],
                    "tool_calls": 0,
                }
            else:
                agent_outputs[name] = result

        # Phase 2 — writer synthesizes
        logger.info("writer_starting")
        writer_result = await self._agents["writer"].execute(
            task=f"Produce an investor brief for: {question}",
            context=agent_outputs,
        )

        elapsed = time() - start
        total_calls = sum(
            o.get("tool_calls", 0) for o in agent_outputs.values()
        )
        logger.info(
            "orchestrator_done",
            elapsed=round(elapsed, 1),
            total_tool_calls=total_calls,
        )

        return {
            "final_brief": writer_result["output"],
            "agent_outputs": agent_outputs,
            "total_tool_calls": total_calls,
            "execution_time_seconds": elapsed,
        }
