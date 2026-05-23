from __future__ import annotations

import json
from typing import Any, Protocol

from openai import AsyncOpenAI

from src.mcp.client import MCPClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Agent(Protocol):
    name: str
    role: str

    async def execute(
        self, task: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...


class BaseAgent:
    name: str = ""
    role: str = ""

    # Subclasses declare their complexity tier; router picks the model.
    # Import deferred to avoid circular import at module load time.
    complexity: Any = None  # set to TaskComplexity.MEDIUM by default in __init__

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model: str | None = None,
        mcp_client: MCPClient | None = None,
    ) -> None:
        from src.router import TaskComplexity, router

        self._openai = openai_client
        self._mcp = mcp_client
        # Use explicit model override (e.g. from tests) or route via complexity tier
        effective_complexity = self.complexity if self.complexity is not None else TaskComplexity.MEDIUM
        self._model = model if model is not None else router.model_for(effective_complexity)
        logger.info(
            "agent_initialized",
            agent=self.__class__.__name__,
            model=self._model,
            complexity=str(effective_complexity),
        )

    def _build_messages(
        self, system_prompt: str, task: str, context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if context:
            ctx_str = self._format_context(context)
            messages.append(
                {
                    "role": "user",
                    "content": f"Context from other agents:\n{ctx_str}\n\nYour task: {task}",
                }
            )
        else:
            messages.append({"role": "user", "content": task})
        return messages

    def _format_context(self, context: dict[str, Any]) -> str:
        parts = []
        for agent_name, output in context.items():
            if isinstance(output, dict) and "output" in output:
                parts.append(f"--- {agent_name} ---\n{output['output']}")
        return "\n\n".join(parts)

    def _build_tools(self) -> list[dict[str, Any]]:
        if self._mcp is None:
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"] or {"type": "object", "properties": {}},
                },
            }
            for t in self._mcp.list_tools()
        ]

    async def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        max_iterations: int = 6,
    ) -> tuple[str, list[dict[str, Any]], int]:
        """Run OpenAI tool-calling loop. Returns (answer, sources, tool_call_count)."""
        from src.mcp.client import MCPClientError

        tools = self._build_tools()
        tool_calls_made = 0
        sources: list[str] = []

        for _ in range(max_iterations):
            kwargs: dict[str, Any] = {"model": self._model, "messages": messages}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = await self._openai.chat.completions.create(**kwargs)
            msg = response.choices[0].message

            if not msg.tool_calls:
                answer = msg.content or ""
                sources = _extract_urls(answer)
                return answer, sources, tool_calls_made

            messages.append(msg.model_dump(exclude_unset=True))

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    fn_args = {}

                logger.info("tool_call", agent=self.name, tool=fn_name)
                try:
                    result_text = await self._mcp.call_tool(fn_name, fn_args)  # type: ignore[union-attr]
                except MCPClientError as exc:
                    result_text = f"Error: {exc}"

                tool_calls_made += 1
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result_text}
                )

        # Iterations exhausted — ask for final answer without tools
        messages.append(
            {"role": "user", "content": "Synthesize what you found into a final answer."}
        )
        response = await self._openai.chat.completions.create(
            model=self._model, messages=messages
        )
        answer = response.choices[0].message.content or ""
        return answer, _extract_urls(answer), tool_calls_made


def _extract_urls(text: str) -> list[str]:
    import re
    return list(dict.fromkeys(re.findall(r"https?://[^\s\)\]\>\"\']+", text)))
