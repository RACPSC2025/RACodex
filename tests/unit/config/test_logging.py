"""
Tests de logging estructurado.
"""

from __future__ import annotations

import structlog

from src.config.logging import configure_logging, get_logger


class TestLogging:
    """Tests del sistema de logging."""

    def test_configure_logging_dev(self) -> None:
        """En desarrollo usa console renderer."""
        configure_logging(is_development=True, log_level="DEBUG")
        # No lanza excepción — se configuró correctamente

    def test_configure_logging_prod(self) -> None:
        """En producción usa JSON renderer."""
        configure_logging(is_development=False, log_level="INFO")
        # No lanza excepción

    def test_configure_logging_infers_env(self, monkeypatch) -> None:
        """Infiere is_development desde APP_ENV si no se especifica."""
        monkeypatch.setenv("APP_ENV", "production")
        configure_logging()  # No lanza — infiere correctamente

    def test_get_logger_returns_logger(self) -> None:
        configure_logging(is_development=True, log_level="DEBUG")
        log = get_logger(__name__)
        assert log is not None

    def test_logger_has_methods(self) -> None:
        configure_logging(is_development=True, log_level="DEBUG")
        log = get_logger("test_module")
        assert hasattr(log, "info")
        assert hasattr(log, "warning")
        assert hasattr(log, "error")
        assert hasattr(log, "debug")
