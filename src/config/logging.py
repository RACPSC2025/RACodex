"""
Logging centralizado — structlog con output estructurado.

Todos los módulos obtienen su logger desde aquí:

    from src.config.logging import get_logger

    log = get_logger(__name__)
    log.info("operacion_exitosa", dato1="valor", dato2=123)

En desarrollo: output legible para consola (console renderer).
En producción: output JSON para ingestión por sistemas de log (Logstash, etc.).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


# ─── Procesadores compartidos ─────────────────────────────────────────────────

def _add_pid(_: logging.Logger, __: str, event_dict: dict) -> dict:
    """Agrega PID del proceso al log (útil para debugging multi-worker)."""
    import os  # noqa: PLC0415
    event_dict["pid"] = os.getpid()
    return event_dict


# ─── Configuración ────────────────────────────────────────────────────────────

def _configure_structlog(
    is_development: bool = True,
    log_level: str = "DEBUG",
) -> None:
    """
    Configura structlog con el renderer apropiado según el entorno.

    Args:
        is_development: True para console renderer legible, False para JSON.
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Renderers: en dev se usa console (legible), en prod JSON (parseable)
    if is_development:
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            # Contexto: inyecta variables de contexto (ej: request_id)
            structlog.contextvars.merge_contextvars,
            # Procesadores estándar de structlog
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            # Timestamp ISO 8601
            structlog.processors.TimeStamper(fmt="iso"),
            # PID del proceso
            _add_pid,
            # Ordenar claves para output determinista
            structlog.processors.OrderedKeyValueRenderer.pair_formatter.order(
                "timestamp", "level", "event", "logger"
            ),
            # Renderer final (console o JSON)
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def configure_logging(
    is_development: bool | None = None,
    log_level: str = "DEBUG",
) -> None:
    """
    Inicializa el sistema de logging.

    Se llama UNA vez en el lifespan de FastAPI (api/main.py).
    Los módulos que llaman get_logger() antes de configure_logging()
    funcionan con defaults; se recomienda llamar esto al inicio.

    Args:
        is_development: Si True, usa console renderer. Si None, se infiere de APP_ENV.
        log_level: Nivel de logging.
    """
    if is_development is None:
        # Inferir desde entorno si no se especifica
        import os  # noqa: PLC0415
        app_env = os.getenv("APP_ENV", "development")
        is_development = app_env == "development"

    # Configurar logging estándar de Python (para librerías de terceros)
    std_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=std_level,
    )

    # Configurar structlog
    _configure_structlog(is_development=is_development, log_level=log_level)


# ─── Factory de loggers ───────────────────────────────────────────────────────

def get_logger(name: str) -> Any:
    """
    Obtiene un logger estructurado para el módulo dado.

    Uso típico en cualquier módulo:
        from src.config.logging import get_logger

        log = get_logger(__name__)
        log.info("evento", clave="valor", count=42)

    Args:
        name: Nombre del logger, típicamente __name__ del módulo.

    Returns:
        Logger con métodos info(), warning(), error(), debug(), exception().
    """
    return structlog.get_logger(name)
