"""Researcher agent: uses DuckDuckGo via MCP to answer startup research questions."""
from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from src.mcp.client import MCPClient, MCPClientError
from src.utils.logger import get_logger

logger = get_logger(__name__)

RESEARCHER_SYSTEM_PROMPT = """You are a research assistant specialized in Indian startup ecosystem analysis. Given a question:

1. Search the web for relevant information using available tools
2. Read the search results carefully
3. Synthesize a brief, factual answer
4. Cite each claim with the source URL

Be concise. Be honest about gaps in available information. Never make up facts."""


class ResearcherAgent:
    def __init__(
        self,
        mcp_client: MCPClient,
        openai_client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        max_iterations: int = 5,
    ) -> None:
        self._mcp = mcp_client
        self._openai = openai_client
        self._model = model
        self._max_iterations = max_iterations

    def _build_tools(self) -> list[dict[str, Any]]:
        """Convert MCP tool descriptors to OpenAI function-call format."""
        tools = []
        for t in self._mcp.list_tools():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"] or {"type": "object", "properties": {}},
                    },
                }
            )
        return tools

    async def research(self, question: str) -> dict[str, Any]:
        """
        Run the agent loop:
        1. LLM decides what to search
        2. Call DuckDuckGo via MCP
        3. LLM reads results and produces final answer
        """
        tools = self._build_tools()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tool_calls_made: list[dict[str, Any]] = []
        sources: list[str] = []

        for iteration in range(self._max_iterations):
            logger.info("agent_iteration", iteration=iteration, model=self._model)

            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = await self._openai.chat.completions.create(**kwargs)
            message = response.choices[0].message

            # No tool calls → final answer
            if not message.tool_calls:
                answer = message.content or ""
                sources = _extract_urls(answer)
                logger.info("agent_done", iterations=iteration + 1, sources=len(sources))
                return {
                    "answer": answer,
                    "sources": sources,
                    "tool_calls_made": tool_calls_made,
                }

            # Append assistant turn with tool calls
            messages.append(message.model_dump(exclude_unset=True))

            # Execute each tool call
            for tc in message.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    fn_args = {}

                logger.info("tool_call", tool=fn_name, args=fn_args)
                try:
                    result_text = await self._mcp.call_tool(fn_name, fn_args)
                except MCPClientError as exc:
                    result_text = f"Error: {exc}"

                tool_calls_made.append({"tool": fn_name, "args": fn_args})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    }
                )

        # Exhausted iterations — ask for a final answer without tools
        messages.append(
            {
                "role": "user",
                "content": "Please synthesize everything you have found into a final answer.",
            }
        )
        response = await self._openai.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        answer = response.choices[0].message.content or ""
        sources = _extract_urls(answer)
        return {
            "answer": answer,
            "sources": sources,
            "tool_calls_made": tool_calls_made,
        }


def _extract_urls(text: str) -> list[str]:
    """Naively pull http/https URLs out of text for the sources list."""
    import re
    return list(dict.fromkeys(re.findall(r"https?://[^\s\)\]\>\"\']+", text)))
