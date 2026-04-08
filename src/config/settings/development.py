"""
Development settings — hereda de base y sobreescribe lo necesario para desarrollo local.
"""

from __future__ import annotations

from pydantic_settings import SettingsConfigDict

from src.config.settings.base import BaseSettings


class DevelopmentSettings(BaseSettings):
    """
    Settings para entorno de desarrollo.

    Características:
    - DEBUG logging activado
    - Base de datos local
    - Docs de API habilitadas
    - Pool pequeño de BD
    - Chroma en directorio local
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "DEBUG"

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/fenix_rag"
    database_pool_size: int = 5
    database_max_overflow: int = 10

    api_workers: int = 1

    # En desarrollo, los modelos se cachean localmente
    flashrank_cache_dir: str = "./storage/models"
    bm25_cache_dir: str = "./storage/bm25_cache"
