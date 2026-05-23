from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.router import TaskComplexity
from src.utils.logger import get_logger

logger = get_logger(__name__)

CRITIC_SYSTEM_PROMPT = """You are a Critic Agent. You review research findings from other agents and identify:

- Red flags or risks the other agents may have missed
- Contradictions between sources
- Claims that lack strong evidence (cited by only one source for big claims)
- Missing perspectives (e.g., bull case present but bear case weak, or vice versa)
- Information gaps (what should we know but don't?)

You also produce:
- A confidence score (1-10) for the overall research quality
- A bull case (3 strongest reasons to invest)
- A bear case (3 strongest reasons NOT to invest)
- A list of "things we don't know" — explicit gaps

Be tough. Your job is critical review, not validation. The Writer will produce the final brief, but they need your honest assessment first.

Output format (use this exact structure):

CRITIC ASSESSMENT

Confidence Score: X/10
[Brief justification]

Red Flags Identified
- [Specific concern from research]

Bull Case
1. [Reason]
2. [Reason]
3. [Reason]

Bear Case
1. [Reason]
2. [Reason]
3. [Reason]

Information Gaps
- [What we couldn't verify]
- [What's missing]

Contradictions Found
- [If any, otherwise "None identified"]"""


class CriticAgent(BaseAgent):
    name = "critic"
    role = "Reviews research findings for quality, red flags, and gaps"
    complexity = TaskComplexity.HEAVY  # uses gpt-4o for strong judgment

    async def execute(
        self, task: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Critic uses only context from other agents — no tool calls."""
        if not context:
            return {"output": "No research findings to review.", "sources": [], "tool_calls": 0}

        context_str = self._format_context(context)
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Research question: {task}\n\n"
                    f"Findings from research agents:\n\n{context_str}\n\n"
                    "Provide your critical assessment."
                ),
            },
        ]

        try:
            response = await self._openai.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.2,
                max_tokens=2000,
            )
            output = response.choices[0].message.content or ""
            logger.info("critic_done", model=self._model)
            return {"output": output, "sources": [], "tool_calls": 0}

        except Exception as exc:
            logger.error("critic_failed", error=str(exc))
            return {
                "output": f"[Critic review failed: {exc}]",
                "sources": [],
                "tool_calls": 0,
            }
