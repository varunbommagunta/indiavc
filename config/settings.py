from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    openai_api_key: str = Field(..., description="OpenAI API key")
    log_level: str = Field(default="INFO")
    max_agent_iterations: int = Field(default=10, ge=1, le=50)
    daily_cost_cap_inr: int = Field(default=50, ge=0)

    orchestrator_model: str = Field(default="gpt-4o")
    critic_model: str = Field(default="gpt-4o")
    worker_model: str = Field(default="gpt-4o-mini")

    mcp_backend: str = Field(
        default="hybrid",
        description="MCP backend: 'ddgs' (web only), 'custom' (structured data only), or 'hybrid' (both)",
    )

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def strip_api_key(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v


settings = Settings()
