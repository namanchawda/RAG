"""Application configuration and environment variable loading."""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Centralized settings for local development and deployment."""

    APP_NAME: str = os.getenv("APP_NAME", "rag-system")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    SEC_EDGAR_USER_AGENT: str = os.getenv(
        "SEC_EDGAR_USER_AGENT",
        "SEC Filing RAG YourName@YourEmail.com",
    )

    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "sec_rag")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_SSLMODE: str = os.getenv("POSTGRES_SSLMODE", "prefer")

    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5",
    )
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    VECTOR_TABLE: str = os.getenv("VECTOR_TABLE", "document_chunks")
    QUERY_TOP_K: int = int(os.getenv("QUERY_TOP_K", "5"))


settings = Settings()


def get_settings() -> Settings:
    """Return the active application settings object."""
    return settings
