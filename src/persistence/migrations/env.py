"""
Configuración de Alembic para Fénix RAG.

Soporta:
  - Migraciones async con asyncpg (para correr online migrations)
  - Autogeneración de migraciones desde los modelos SQLAlchemy
  - Variables de entorno vía settings (no hardcoded en alembic.ini)

Para crear una nueva migración:
    alembic revision --autogenerate -m "descripcion breve"

Para aplicar todas las migraciones pendientes:
    alembic upgrade head

Para ver el historial:
    alembic history --verbose

Para hacer rollback de la última migración:
    alembic downgrade -1
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Importar Base para que Alembic detecte los modelos automáticamente
from src.persistence.models import Base
from src.config.settings import get_settings

# Configuración de Alembic desde alembic.ini
config = context.config

# Configurar logging si hay archivo de configuración
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata de los modelos para autogeneración
target_metadata = Base.metadata


def get_url() -> str:
    """Obtiene la URL de BD desde settings (no desde alembic.ini)."""
    settings = get_settings()
    # Alembic necesita la URL síncrona para las operaciones de migración
    return settings.database_url


def run_migrations_offline() -> None:
    """
    Modo offline: genera SQL sin conectarse a la BD.

    Útil para revisar las migraciones antes de aplicarlas,
    o para aplicarlas manualmente en entornos con acceso restringido.

    Comando: alembic upgrade head --sql > migration.sql
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,          # detectar cambios de tipo de columna
        compare_server_default=True, # detectar cambios en server_default
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Ejecuta las migraciones en la conexión dada."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        # Detectar tablas que existen en BD pero no en modelos
        include_schemas=False,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Modo online async: se conecta a la BD y aplica las migraciones.

    Usa asyncpg pero Alembic necesita una conexión síncrona,
    por eso usamos `run_sync` para hacer el puente.
    """
    # Convertir URL a asyncpg para el engine async
    url = get_url().replace(
        "postgresql+psycopg://", "postgresql+asyncpg://"
    ).replace(
        "postgresql://", "postgresql+asyncpg://"
    )

    connectable = async_engine_from_config(
        {"sqlalchemy.url": url},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # sin pool para migraciones (conexión única)
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point para modo online — llama al runtime async."""
    asyncio.run(run_async_migrations())


# ── Entry point de Alembic ───────────────────────────────────────────────────
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
