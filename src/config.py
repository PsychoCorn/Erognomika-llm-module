from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_model: str = Field(default="deepseek-chat")
    llm_base_url: str = Field(default="https://api.deepseek.com")
    llm_api_key: str = Field(...)

    llm_temperature: float = Field(default=0.2)
    llm_max_tokens: int = Field(default=2000)
    llm_timeout: float = Field(default=60.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()