from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent, _extract_urls
from src.utils.logger import get_logger

logger = get_logger(__name__)

WRITER_SYSTEM_PROMPT = """You are a Writer Agent. Your job: synthesize findings from research agents into a structured investor brief.

Use ONLY information provided in the context from other agents. Do not add facts not in the context.

Output format (use this exact structure):

INVESTOR BRIEF — [Company Name]

## Company Overview
[2-3 sentences from web researcher findings]

## Funding History
[Bullet points from web researcher's funding data]

## Market Position & Competitors
[From competitor analyzer findings]

## Reputation & Risks
[From news analyzer findings, both positive and negative]

## Bull Case (3 reasons to invest)
[Synthesize the positives]

## Bear Case (3 reasons NOT to invest)
[Synthesize the negatives and risks]

## Sources
[Aggregate all source URLs from the agents]

## Disclaimer
This is automated research, not financial advice. Verify all claims independently before any investment decision.

Be balanced. If the agents found mostly positive info, still try to identify risks. If mostly negative, identify any positives. The user needs a complete picture."""


class WriterAgent(BaseAgent):
    name = "writer"
    role = "Synthesizes research findings into a structured investor brief"

    async def execute(
        self, task: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        messages = self._build_messages(WRITER_SYSTEM_PROMPT, task, context)
        # Writer uses no tools — pure LLM synthesis
        response = await self._openai.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        answer = response.choices[0].message.content or ""
        # Collect all sources from context agents plus any URLs in the brief itself
        all_sources: list[str] = []
        if context:
            for agent_out in context.values():
                if isinstance(agent_out, dict):
                    all_sources.extend(agent_out.get("sources", []))
        all_sources.extend(_extract_urls(answer))
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped = [s for s in all_sources if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]

        logger.info("writer_done", sources=len(deduped))
        return {"output": answer, "sources": deduped, "tool_calls": 0}
