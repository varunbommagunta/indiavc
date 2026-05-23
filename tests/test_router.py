"""Tests for the LLM Router."""
from __future__ import annotations


def test_router_returns_gpt4o_for_heavy_complexity() -> None:
    from src.router import TaskComplexity, router

    assert router.model_for(TaskComplexity.HEAVY) == "gpt-4o"


def test_router_returns_gpt4o_mini_for_light_complexity() -> None:
    from src.router import TaskComplexity, router

    assert router.model_for(TaskComplexity.LIGHT) == "gpt-4o-mini"


def test_router_returns_gpt4o_mini_for_medium_complexity() -> None:
    from src.router import TaskComplexity, router

    assert router.model_for(TaskComplexity.MEDIUM) == "gpt-4o-mini"
