"""Application configuration."""
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App settings from environment."""

    openai_api_key: str = ""
    openai_api_base: str | None = None
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./data/chroma_db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def get_settings() -> Settings:
    return Settings()


def ensure_data_dir() -> Path:
    path = Path(__file__).parent / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path
