from enum import Enum

from config.settings import settings


class TaskComplexity(str, Enum):
    HEAVY = "heavy"    # planning, reasoning, judgment → gpt-4o
    MEDIUM = "medium"  # synthesis with structure → gpt-4o-mini
    LIGHT = "light"    # tool-heavy mechanical work → gpt-4o-mini


class LLMRouter:
    """Routes agents to appropriate models based on task complexity."""

    def __init__(self) -> None:
        self._routing = {
            TaskComplexity.HEAVY: settings.orchestrator_model,
            TaskComplexity.MEDIUM: settings.worker_model,
            TaskComplexity.LIGHT: settings.worker_model,
        }

    def model_for(self, complexity: TaskComplexity) -> str:
        return self._routing[complexity]


router = LLMRouter()
