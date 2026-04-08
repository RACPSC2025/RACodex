"""
Settings package — factory que carga el entorno correcto.

Uso:
    from src.config.settings import get_settings

    settings = get_settings()
    print(settings.app_env)
    print(settings.database_url)

El factory selecciona la clase correcta según `APP_ENV`:
  - development → DevelopmentSettings
  - production  → ProductionSettings
  - staging     → StagingSettings
  - testing     → TestingSettings

Todos heredan de BaseSettings (base.py).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.settings.base import BaseSettings


def _get_settings_class():
    """
    Selecciona la clase de settings según APP_ENV.

    Orden de precedencia:
    1. Variable de entorno APP_ENV
    2. Default: development
    """
    env = os.getenv("APP_ENV", "development")

    match env:
        case "production":
            from src.config.settings.production import ProductionSettings
            return ProductionSettings
        case "staging":
            from src.config.settings.staging import StagingSettings
            return StagingSettings
        case "testing":
            from src.config.settings.testing import TestingSettings
            return TestingSettings
        case _:
            from src.config.settings.development import DevelopmentSettings
            return DevelopmentSettings


@lru_cache(maxsize=1)
def get_settings() -> "BaseSettings":
    """
    Retorna la instancia singleton de Settings para el entorno activo.

    Gracias a @lru_cache, se carga UNA sola vez y se reutiliza
    en todas las llamadas a get_settings().

    Para tests, usa clear_settings_cache() antes de cada test.
    """
    settings_cls = _get_settings_class()
    return settings_cls()


def clear_settings_cache() -> None:
    """
    Limpia la cache de settings.

    Útil en tests para cambiar el entorno entre pruebas.
    """
    get_settings.cache_clear()


# ─── Re-exports ───────────────────────────────────────────────────────────────
# Permite importar directamente desde src.config.settings

__all__ = ["get_settings", "clear_settings_cache"]
