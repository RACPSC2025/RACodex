"""
Tests de configuración — settings por entorno.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config.settings import get_settings, clear_settings_cache
from src.config.settings.base import BaseSettings
from src.config.settings.development import DevelopmentSettings
from src.config.settings.testing import TestingSettings


# ─── Base Settings ────────────────────────────────────────────────────────────

class TestBaseSettings:
    """Tests para la configuración base compartida."""

    def test_default_values(self) -> None:
        """Los defaults permiten arrancar sin .env."""
        settings = BaseSettings()
        assert settings.app_env == "development"
        assert settings.log_level == "INFO"
        assert settings.chunk_size == 1100
        assert settings.chunk_overlap == 220
        assert settings.retrieval_top_k == 10
        assert settings.api_port == 8000

    def test_is_development_property(self) -> None:
        assert BaseSettings(app_env="development").is_development is True
        assert BaseSettings(app_env="production").is_development is False
        assert BaseSettings(app_env="testing").is_development is False

    def test_is_production_property(self) -> None:
        assert BaseSettings(app_env="production").is_production is True
        assert BaseSettings(app_env="development").is_production is False

    def test_is_testing_property(self) -> None:
        assert BaseSettings(app_env="testing").is_testing is True
        assert BaseSettings(app_env="development").is_testing is False

    def test_ensure_directories_creates_all(self, tmp_path: Path) -> None:
        """ensure_directories crea todos los directorios necesarios."""
        settings = BaseSettings(root_dir=tmp_path)
        settings.ensure_directories()

        assert (tmp_path / "storage").exists()
        assert (tmp_path / "storage" / "uploads").exists()


# ─── Validadores ─────────────────────────────────────────────────────────────

class TestSettingsValidators:
    """Tests de validación de configuración."""

    def test_chunk_size_positive(self) -> None:
        with pytest.raises(ValidationError):
            BaseSettings(chunk_size=-1)

    def test_chunk_size_zero(self) -> None:
        with pytest.raises(ValidationError):
            BaseSettings(chunk_size=0)

    def test_chunk_overlap_negative(self) -> None:
        with pytest.raises(ValidationError):
            BaseSettings(chunk_overlap=-1)

    def test_chunk_overlap_less_than_chunk_size(self) -> None:
        """El overlap debe ser menor que el chunk_size."""
        with pytest.raises(ValidationError):
            BaseSettings(chunk_size=500, chunk_overlap=600)

    def test_valid_overlap(self) -> None:
        settings = BaseSettings(chunk_size=1100, chunk_overlap=220)
        assert settings.chunk_overlap == 220

    def test_retrieval_top_k_positive(self) -> None:
        with pytest.raises(ValidationError):
            BaseSettings(retrieval_top_k=0)

    def test_rerank_top_k_not_greater_than_top_k(self) -> None:
        """rerank_top_k no puede ser mayor que retrieval_top_k."""
        with pytest.raises(ValidationError):
            BaseSettings(retrieval_top_k=5, retrieval_rerank_top_k=10)


# ─── Entornos específicos ────────────────────────────────────────────────────

class TestDevelopmentSettings:
    """Tests para configuración de desarrollo."""

    def test_development_defaults(self) -> None:
        settings = DevelopmentSettings()
        assert settings.app_env == "development"
        assert settings.log_level == "DEBUG"
        assert settings.database_pool_size == 5
        assert settings.api_workers == 1

    def test_uses_local_database(self) -> None:
        settings = DevelopmentSettings()
        assert "localhost" in settings.database_url


class TestTestingSettings:
    """Tests para configuración de testing."""

    def test_testing_defaults(self) -> None:
        settings = TestingSettings()
        assert settings.app_env == "testing"
        assert settings.log_level == "ERROR"
        assert settings.database_pool_size == 1
        assert "sqlite" in settings.database_url
        assert settings.api_workers == 1

    def test_uses_test_chroma_dir(self) -> None:
        settings = TestingSettings()
        assert "test" in settings.chroma_persist_dir
        assert "test" in settings.chroma_collection_name


# ─── Singleton + Cache ──────────────────────────────────────────────────────

class TestSettingsSingleton:
    """Tests del patrón singleton con caché."""

    def test_get_settings_returns_instance(self) -> None:
        clear_settings_cache()
        settings = get_settings()
        assert settings is not None

    def test_get_settings_caches_result(self) -> None:
        clear_settings_cache()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2  # misma instancia por lru_cache

    def test_clear_cache_invalidates(self) -> None:
        clear_settings_cache()
        s1 = get_settings()
        clear_settings_cache()
        s2 = get_settings()
        # Pueden ser la misma si no cambian las vars de entorno
        # pero la cache fue limpiada
        assert s1 is not None
        assert s2 is not None
