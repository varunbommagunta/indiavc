from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent, _extract_urls
from src.router import TaskComplexity
from src.utils.logger import get_logger

logger = get_logger(__name__)

WRITER_SYSTEM_PROMPT = """You are a Writer Agent. Your job: synthesize findings from research agents into a structured investor brief.

Use ONLY information provided in the context from other agents. Do not add facts not in the context.

You will receive a Critic Assessment in your context. The Critic has reviewed the research and identified red flags, bull/bear cases, and information gaps.
Use the Critic's bull and bear cases as the basis for your INVESTOR BRIEF sections. Use their red flags and information gaps in your "Reputation & Risks" section.
DO NOT contradict the Critic. If they identified a risk, include it. If they found a gap, acknowledge it.

Output format (use this exact structure):

INVESTOR BRIEF — [Company Name]

## Company Overview
[2-3 sentences from web researcher findings]

## Funding History
[Bullet points from web researcher's funding data]

## Market Position & Competitors
[From competitor analyzer findings]

## Reputation & Risks
[From news analyzer and critic findings, both positive and negative]

## Bull Case (3 reasons to invest)
[Use the Critic's bull case if available, else synthesize the positives]

## Bear Case (3 reasons NOT to invest)
[Use the Critic's bear case if available, else synthesize the negatives and risks]

## Information Gaps
[From the Critic's "things we don't know" section, if available]

## Sources
[Aggregate all source URLs from the agents]

## Disclaimer
This is automated research, not financial advice. Verify all claims independently before any investment decision.

Be balanced. The user needs a complete picture."""


class WriterAgent(BaseAgent):
    name = "writer"
    role = "Synthesizes research findings into a structured investor brief"
    complexity = TaskComplexity.MEDIUM

    async def execute(
        self, task: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        messages = self._build_messages(WRITER_SYSTEM_PROMPT, task, context)
        response = await self._openai.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        answer = response.choices[0].message.content or ""
        all_sources: list[str] = []
        if context:
            for agent_out in context.values():
                if isinstance(agent_out, dict):
                    all_sources.extend(agent_out.get("sources", []))
        all_sources.extend(_extract_urls(answer))
        seen: set[str] = set()
        deduped = [s for s in all_sources if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]

        logger.info("writer_done", sources=len(deduped))
        return {"output": answer, "sources": deduped, "tool_calls": 0}
