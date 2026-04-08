"""
Capa de infraestructura de base de datos para Fénix RAG.

Proporciona:
  - AsyncEngine configurado desde settings
  - AsyncSessionFactory para crear sesiones
  - `get_db()` como dependency injection para FastAPI
  - `transaction()` como context manager para repositorios
  - Utilidades de health check y migración

Por qué async:
  El agente LangGraph ejecuta nodos en event loop. Si la capa de BD
  fuera síncrona, cada query bloquearía el event loop durante la I/O
  de red a PostgreSQL. Con AsyncSession toda la I/O es no bloqueante.

Por qué NO usar `create_engine` síncrono:
  SQLAlchemy 2 soporta ambos. En este proyecto usamos async porque
  FastAPI y LangGraph async nodes requieren que toda la I/O sea async.
  Para scripts y migraciones de Alembic usamos el engine síncrono
  separado (`create_sync_engine`).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import create_engine, text

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.persistence.models import Base

log = get_logger(__name__)

# ─── Engines (lazy-initialized) ───────────────────────────────────────────────

_async_engine: AsyncEngine | None = None
_sync_engine = None  # Solo para Alembic y scripts


def get_async_engine() -> AsyncEngine:
    """
    Retorna el AsyncEngine singleton.

    Configurado con pool de conexiones para alta concurrencia:
    - pool_size: conexiones mantenidas activas
    - max_overflow: conexiones adicionales bajo carga pico
    - pool_pre_ping: verifica conexiones antes de usarlas (resiliente a timeouts)
    - pool_recycle: recicla conexiones cada hora (previene conexiones muertas)
    """
    global _async_engine  # noqa: PLW0603

    if _async_engine is None:
        settings = get_settings()

        # Convertir URL psycopg síncrona a asyncpg para async
        # "postgresql+psycopg://..." → "postgresql+asyncpg://..."
        async_url = settings.database_url.replace(
            "postgresql+psycopg://", "postgresql+asyncpg://"
        ).replace(
            "postgresql://", "postgresql+asyncpg://"
        )

        _async_engine = create_async_engine(
            async_url,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=settings.is_development,  # SQL logging solo en dev
        )

        log.info(
            "async_engine_created",
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            echo=settings.is_development,
        )

    return _async_engine


def get_sync_engine():
    """
    Engine síncrono solo para Alembic y scripts de administración.

    NO usar en código de aplicación (rompe el event loop).
    """
    global _sync_engine  # noqa: PLW0603

    if _sync_engine is None:
        settings = get_settings()
        _sync_engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            echo=False,
        )

    return _sync_engine


# ─── Session Factory ──────────────────────────────────────────────────────────

def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Crea la factory de AsyncSession configurada.

    expire_on_commit=False:
      Mantiene los atributos accesibles después del commit.
      Sin esto, acceder a obj.id después de commit lanza LazyLoadingError
      porque la sesión expiró el objeto.
    """
    return async_sessionmaker(
        bind=get_async_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


# ─── Dependency injection ─────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency de FastAPI para inyectar sesión de BD.

    Uso en endpoints:
        @router.get("/sessions")
        async def list_sessions(db: AsyncSession = Depends(get_db)):
            ...

    Garantiza:
    - La sesión se cierra siempre (finally)
    - En caso de excepción hace rollback automático
    - Una sesión por request (no compartida entre requests)
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ─── Context manager para repositorios ───────────────────────────────────────

@asynccontextmanager
async def transaction() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager para operaciones transaccionales en repositorios.

    Uso en repositorios y servicios (fuera de FastAPI):
        async with transaction() as db:
            await session_repo.create(db, session_data)
            await document_repo.create(db, doc_data)
            # commit automático al salir del bloque
            # rollback automático si hay excepción

    Para operaciones de solo lectura, usar `read_session()` que
    es más eficiente (no adquiere locks de escritura).
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
            log.debug("transaction_committed")
        except Exception as exc:
            await session.rollback()
            log.warning("transaction_rolled_back", error=str(exc))
            raise


@asynccontextmanager
async def read_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager para operaciones de solo lectura.

    Más eficiente que `transaction()` porque no gestiona commits
    y usa autocommit implícito para reads.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        finally:
            await session.close()


# ─── Inicialización de tablas ─────────────────────────────────────────────────

async def create_tables() -> None:
    """
    Crea todas las tablas definidas en models.py.

    SOLO para desarrollo y tests. En producción usar Alembic migrations.
    """
    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("database_tables_created")


async def drop_tables() -> None:
    """
    Elimina todas las tablas. SOLO para tests.
    """
    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    log.warning("database_tables_dropped")


# ─── Health check ─────────────────────────────────────────────────────────────

async def check_database_connectivity() -> dict:
    """
    Verifica la conectividad con PostgreSQL.

    Retorna dict compatible con el endpoint /health.
    """
    try:
        async with read_session() as db:
            await db.execute(text("SELECT 1"))
        return {"database": "ok", "engine": "postgresql+asyncpg"}
    except Exception as exc:
        log.error("database_health_check_failed", error=str(exc))
        return {"database": "error", "error": str(exc)}


# ─── Reset de cache (para tests) ─────────────────────────────────────────────

async def reset_engine_cache() -> None:
    """Resetea los engines cacheados. Usar solo en tests."""
    global _async_engine, _sync_engine  # noqa: PLW0603

    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None

    if _sync_engine:
        _sync_engine.dispose()
        _sync_engine = None

    log.debug("engine_cache_reset")
