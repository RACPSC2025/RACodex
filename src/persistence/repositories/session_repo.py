"""
Repositorio de Sessions y Messages.

Patrón Repository:
  La lógica de acceso a datos está aquí — los nodos del agente
  no conocen SQLAlchemy ni SQL. Llaman a estas funciones y
  reciben modelos de dominio o None.

Patrón Unit of Work:
  La sesión de BD se inyecta desde fuera (viene del context manager
  `transaction()` o de la dependency FastAPI `get_db()`).
  El repositorio NUNCA hace commit ni rollback — eso es
  responsabilidad del caller. Esto permite coordinar múltiples
  repos en una sola transacción atómica.

Convención de nombres:
  get_*     → SELECT, retorna Optional[Model] o list[Model]
  create_*  → INSERT, retorna Model creado
  update_*  → UPDATE, retorna Model actualizado
  delete_*  → soft-delete (deleted_at), retorna bool
  count_*   → COUNT query
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.config.logging import get_logger
from src.persistence.models import Message, Session

log = get_logger(__name__)


# ─── Session repository ───────────────────────────────────────────────────────

async def create_session(
    db: AsyncSession,
    *,
    user_identifier: str,
    title: Optional[str] = None,
) -> Session:
    """
    Crea una nueva sesión de conversación.

    Args:
        db: AsyncSession inyectada.
        user_identifier: ID del usuario (email, sub JWT, o anónimo).
        title: Título opcional (se auto-genera desde la primera query si es None).

    Returns:
        Session creada (flush pero no commit — el caller hace commit).
    """
    session = Session(
        user_identifier=user_identifier,
        title=title,
    )
    db.add(session)
    await db.flush()  # obtener el ID sin hacer commit

    log.info("session_created", session_id=str(session.id), user=user_identifier)
    return session


async def get_session(
    db: AsyncSession,
    session_id: uuid.UUID,
    *,
    load_messages: bool = False,
) -> Optional[Session]:
    """
    Busca una sesión por ID.

    Args:
        db: AsyncSession.
        session_id: UUID de la sesión.
        load_messages: Si True, carga los mensajes en la misma query (eager loading).

    Returns:
        Session o None si no existe o está soft-deleted.
    """
    stmt = (
        select(Session)
        .where(Session.id == session_id)
        .where(Session.deleted_at.is_(None))
    )

    if load_messages:
        stmt = stmt.options(selectinload(Session.messages))

    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_sessions_by_user(
    db: AsyncSession,
    user_identifier: str,
    *,
    limit: int = 20,
    offset: int = 0,
    active_only: bool = True,
) -> list[Session]:
    """
    Lista las sesiones de un usuario paginadas.

    Ordenadas por actividad más reciente primero.
    """
    stmt = (
        select(Session)
        .where(Session.user_identifier == user_identifier)
        .where(Session.deleted_at.is_(None))
    )

    if active_only:
        stmt = stmt.where(Session.is_active.is_(True))

    stmt = stmt.order_by(Session.updated_at.desc()).limit(limit).offset(offset)

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def update_session_title(
    db: AsyncSession,
    session_id: uuid.UUID,
    title: str,
) -> Optional[Session]:
    """Actualiza el título de una sesión."""
    stmt = (
        update(Session)
        .where(Session.id == session_id)
        .where(Session.deleted_at.is_(None))
        .values(title=title[:500])
        .returning(Session)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def increment_session_counters(
    db: AsyncSession,
    session_id: uuid.UUID,
    *,
    messages_delta: int = 0,
    documents_delta: int = 0,
) -> None:
    """
    Incrementa los contadores de mensajes/documentos de una sesión.

    Usa UPDATE atómico en lugar de read-modify-write para evitar
    race conditions bajo carga concurrente.
    """
    values: dict = {}
    if messages_delta:
        values["total_messages"] = Session.total_messages + messages_delta
    if documents_delta:
        values["total_documents"] = Session.total_documents + documents_delta

    if not values:
        return

    await db.execute(
        update(Session)
        .where(Session.id == session_id)
        .values(**values)
    )


async def delete_session(
    db: AsyncSession,
    session_id: uuid.UUID,
) -> bool:
    """
    Soft-delete de una sesión (preserva historial para auditoría).

    Returns:
        True si encontró y marcó la sesión, False si no existía.
    """
    result = await db.execute(
        update(Session)
        .where(Session.id == session_id)
        .where(Session.deleted_at.is_(None))
        .values(
            deleted_at=datetime.now(timezone.utc),
            is_active=False,
        )
    )
    deleted = result.rowcount > 0
    if deleted:
        log.info("session_deleted", session_id=str(session_id))
    return deleted


async def count_user_sessions(
    db: AsyncSession,
    user_identifier: str,
) -> int:
    """Cuenta las sesiones activas de un usuario."""
    result = await db.execute(
        select(func.count(Session.id))
        .where(Session.user_identifier == user_identifier)
        .where(Session.deleted_at.is_(None))
        .where(Session.is_active.is_(True))
    )
    return result.scalar_one() or 0


# ─── Message repository ───────────────────────────────────────────────────────

async def create_message(
    db: AsyncSession,
    *,
    session_id: uuid.UUID,
    role: str,
    content: str,
    sources: Optional[list[dict]] = None,
    retrieval_strategy: Optional[str] = None,
    iteration_count: int = 0,
    reflection_score: Optional[float] = None,
    response_time_ms: Optional[int] = None,
) -> Message:
    """
    Crea un mensaje en una sesión.

    Args:
        db: AsyncSession.
        session_id: UUID de la sesión a la que pertenece.
        role: "human" | "assistant" | "system"
        content: Texto del mensaje.
        sources: Lista de fuentes usadas por el agente (solo para role="assistant").
        retrieval_strategy: Estrategia de retrieval usada.
        iteration_count: Número de iteraciones de reflexión.
        reflection_score: Score final de calidad [0.0-1.0].
        response_time_ms: Tiempo total de respuesta en ms.

    Returns:
        Message creado.
    """
    message = Message(
        session_id=session_id,
        role=role,
        content=content,
        sources=sources,
        retrieval_strategy=retrieval_strategy,
        iteration_count=iteration_count,
        reflection_score=reflection_score,
        response_time_ms=response_time_ms,
    )
    db.add(message)
    await db.flush()

    log.debug(
        "message_created",
        message_id=str(message.id),
        session_id=str(session_id),
        role=role,
    )
    return message


async def get_messages_by_session(
    db: AsyncSession,
    session_id: uuid.UUID,
    *,
    limit: int = 50,
    offset: int = 0,
) -> list[Message]:
    """
    Retorna los mensajes de una sesión, ordenados por tiempo.

    Paginados para no cargar historial completo de sesiones largas.
    """
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())


async def get_last_n_messages(
    db: AsyncSession,
    session_id: uuid.UUID,
    n: int = 10,
) -> list[Message]:
    """
    Retorna los últimos N mensajes de una sesión.

    Útil para construir el historial de conversación para el LLM
    sin cargar toda la sesión.
    """
    # Subquery: obtener los últimos N por created_at
    subq = (
        select(Message.id)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(n)
        .subquery()
    )

    result = await db.execute(
        select(Message)
        .where(Message.id.in_(select(subq)))
        .order_by(Message.created_at.asc())
    )
    return list(result.scalars().all())


async def count_messages_by_session(
    db: AsyncSession,
    session_id: uuid.UUID,
) -> int:
    """Cuenta los mensajes de una sesión."""
    result = await db.execute(
        select(func.count(Message.id)).where(Message.session_id == session_id)
    )
    return result.scalar_one() or 0
