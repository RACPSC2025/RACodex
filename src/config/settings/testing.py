"""
Testing settings — hereda de base y sobreescribe para tests.
"""

from __future__ import annotations

from pydantic_settings import SettingsConfigDict

from src.config.settings.base import BaseSettings


class TestingSettings(BaseSettings):
    """
    Settings para entorno de testing.

    Características:
    - ERROR logging (mínimo ruido en output de tests)
    - SQLite en memoria para tests de BD (rápido, aislado)
    - Chroma en memoria (InMemorySaver)
    - Pool mínimo
    - Sin workers múltiples
    """

    model_config = SettingsConfigDict(
        env_file=".env.test",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "testing"
    log_level: str = "ERROR"

    # SQLite en memoria para tests — rápido y aislado
    database_url: str = "sqlite+aiosqlite:///./test_fenix_rag.db"
    database_pool_size: int = 1
    database_max_overflow: int = 0

    api_workers: int = 1

    # Chroma en memoria para tests
    chroma_persist_dir: str = "./storage/chroma_test"
    chroma_collection_name: str = "fenix_legal_test"

    # Caches aislados para tests
    bm25_cache_dir: str = "./storage/bm25_cache_test"
    flashrank_cache_dir: str = "./storage/models_test"
