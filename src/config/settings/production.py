"""
Production settings — hereda de base y sobreescribe para producción.
"""

from __future__ import annotations

from pydantic_settings import SettingsConfigDict

from src.config.settings.base import BaseSettings


class ProductionSettings(BaseSettings):
    """
    Settings para entorno de producción.

    Características:
    - INFO logging (no DEBUG)
    - DB pool grande
    - Múltiples workers de API
    - Chroma en persist_dir optimizado
    - Docs de API deshabilitadas (se maneja en main.py)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "production"
    log_level: str = "INFO"

    database_url: str = ""  # Debe venir del .env en producción
    database_pool_size: int = 20
    database_max_overflow: int = 40

    api_workers: int = 4

    # Modelos precargados en producción
    flashrank_cache_dir: str = "/opt/fenix/models"
    bm25_cache_dir: str = "/opt/fenix/bm25_cache"
    chroma_persist_dir: str = "/opt/fenix/chroma"
