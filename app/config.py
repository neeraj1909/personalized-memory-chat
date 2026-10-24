"""Runtime settings; credentials stay in the private .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration with safe defaults for local development and tests."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    app_name: str = "MyNaksh Personalized AI Chat"
    database_path: Path = Path("memory.db")
    max_message_chars: int = Field(default=128_000, ge=1)
    prompt_max_chars: int = Field(default=256_000, ge=1000)
    memory_limit: int = Field(default=5, ge=1, le=20)
    memory_candidates: int = Field(default=40, ge=1, le=200)
    language_history_limit: int = Field(default=3, ge=0, le=20)
    language_model: str | None = None
    eval_judge_model: str | None = None
    otel_exporter_otlp_endpoint: str | None = None

    llm_provider: str = Field(
        default="openai-compatible",
        validation_alias=AliasChoices("LLM_PROVIDER", "LITELLM_PROVIDER"),
    )
    llm_model: str = Field(
        default="chatgpt/gpt-6.1-sol",
        validation_alias=AliasChoices("LLM_MODEL", "LITELLM_MODEL"),
    )
    llm_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_BASE_URL", "LITELLM_PROVIDER_BASE_URL"),
    )
    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LLM_API_KEY",
            "LITELLM_PROVIDER_API_KEY",
        ),
    )
    llm_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=120,
        validation_alias=AliasChoices("LLM_TIMEOUT_SECONDS", "LITELLM_TIMEOUT_SECONDS"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
