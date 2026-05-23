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

    def _task_for(self, agent_name: str, question: str) -> str:
        """Return the task string for a given agent."""
        tasks = {
            "web_researcher": f"Research general information about: {question}",
            "news_analyzer": f"Investigate news, controversies, and reputation of: {question}",
            "competitor_analyzer": f"Identify and analyze competitors of: {question}",
        }
        return tasks.get(agent_name, question)

    async def run_research(self, question: str) -> dict[str, Any]:
        """
        Phase 1: run web_researcher, news_analyzer, competitor_analyzer in parallel.
        Phase 2: critic reviews all Phase 1 findings.
        Phase 3: writer synthesizes everything (including critic's assessment).
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

        # Phase 2 — critic reviews research findings
        logger.info("critic_starting")
        critic_result = await self._agents["critic"].execute(
            task=f"Critically review research findings for: {question}",
            context=agent_outputs,
        )
        agent_outputs["critic"] = critic_result

        # Phase 3 — writer synthesizes (now includes critic's assessment)
        logger.info("writer_starting")
        writer_result = await self._agents["writer"].execute(
            task=f"Produce an investor brief for: {question}",
            context=agent_outputs,
        )

        elapsed = time() - start
        # critic and writer have tool_calls=0 so only count Phase 1 search calls
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
