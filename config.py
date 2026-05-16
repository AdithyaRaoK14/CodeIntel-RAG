"""
config.py
─────────
Centralised settings — all configuration in one place.

Every module imports from here instead of scattering os.getenv() calls.
Values are read from environment variables or a .env file (via pydantic-settings).

Usage:
    from config import settings
    model = settings.ollama_model

CHANGES
───────
• FIX: rerank_candidates raised from 15 → 20 so the cross-encoder reranker
  sees a wider candidate pool and correct answers ranked 16-20 by the hybrid
  scorer are not silently dropped before final ordering.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Security
    api_key: str = ""

    # Paths
    repo_base: str = "./repos"
    default_repo: str = "sample_repo"

    # LLM
    ollama_model: str = "llama3.2:3b"

    # Retrieval
    top_k: int = 5
    rerank_candidates: int = 20  # FIX: was 15 — wider pool improves Hit@1
    cache_ttl: int = 3600

    # Upload limits
    max_repo_size_mb: int = 500

    # CORS
    allowed_origins: str = (
        "http://localhost:3000,http://localhost:5173,"
        "http://localhost:8000,http://127.0.0.1:8000"
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
