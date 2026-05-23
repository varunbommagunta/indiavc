from pydantic import BaseModel


class ResearchRequest(BaseModel):
    question: str


class AgentOutput(BaseModel):
    output: str
    sources: list[str] = []
    tool_calls: int = 0


class ResearchResponse(BaseModel):
    brief: str
    agent_outputs: dict[str, AgentOutput]
    total_tool_calls: int
    execution_time_seconds: float
