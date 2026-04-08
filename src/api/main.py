"""
Aplicación FastAPI principal — Application Factory pattern.

uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from src.api.middleware import register_exception_handlers, register_middleware
from src.api.routes import admin, chat, documents, health, sessions
from src.config.logging import configure_logging, get_logger
from src.config.settings import get_settings

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: configurar logging, BD, vector store. Shutdown: liberar recursos."""
    settings = get_settings()
    configure_logging()
    settings.ensure_directories()

    log.info("fenix_rag_starting", env=settings.app_env, port=settings.api_port)

    # Tablas BD (solo en dev — en prod usar: alembic upgrade head)
    if settings.is_development:
        try:
            from src.persistence.database import create_tables  # noqa: PLC0415
            await create_tables()
        except Exception as exc:
            log.warning("create_tables_failed", error=str(exc))

    # Vector store
    try:
        from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415
        get_vector_store().open_or_create()
    except Exception as exc:
        log.warning("vector_store_init_failed", error=str(exc))

    log.info("fenix_rag_started")
    yield

    log.info("fenix_rag_shutting_down")
    try:
        from src.persistence.database import reset_engine_cache  # noqa: PLC0415
        await reset_engine_cache()
    except Exception:
        pass
    log.info("fenix_rag_stopped")


def create_app() -> FastAPI:
    """Crea y configura la instancia de FastAPI."""
    settings = get_settings()

    app = FastAPI(
        title="Fénix RAG API",
        description=(
            "Sistema de consulta legal colombiana con retrieval híbrido y auto-reflexión.\n\n"
            "**Stack:** LangGraph · LangChain · AWS Bedrock · Chroma · PostgreSQL"
        ),
        version="1.0.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    register_middleware(app)
    register_exception_handlers(app)

    prefix = "/api/v1"
    app.include_router(health.router)
    app.include_router(sessions.router, prefix=prefix)
    app.include_router(chat.router, prefix=prefix)
    app.include_router(documents.router, prefix=prefix)
    app.include_router(admin.router, prefix=prefix)

    return app


app = create_app()
