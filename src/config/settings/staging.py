"""
Staging settings — hereda de base y sobreescribe para pre-producción.
"""

from __future__ import annotations

from pydantic_settings import SettingsConfigDict

from src.config.settings.base import BaseSettings


class StagingSettings(BaseSettings):
    """
    Settings para entorno de staging (pre-producción).

    Características:
    - Similar a producción pero con DB separada
    - WARNING logging (menos verboso que dev)
    - Pool intermedio
    - Docs de API habilitadas para debugging
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "staging"
    log_level: str = "WARNING"

    database_url: str = ""  # Debe venir del .env en staging
    database_pool_size: int = 10
    database_max_overflow: int = 20

    api_workers: int = 2
