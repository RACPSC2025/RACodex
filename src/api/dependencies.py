"""
Dependencias de FastAPI — inyección de BD, sesión del agente, paginación.
"""
from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.logging import get_logger
from src.persistence.database import get_db
from src.persistence.models import Session as DBSession
from src.persistence.repositories import session_repo

log = get_logger(__name__)

# Alias tipado: `db: DB` en los endpoints
DB = Annotated[AsyncSession, Depends(get_db)]


class PaginationParams:
    def __init__(
        self,
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> None:
        self.limit = limit
        self.offset = offset


Pagination = Annotated[PaginationParams, Depends(PaginationParams)]


async def get_agent_session(session_id: str, db: DB) -> DBSession:
    """
    Valida que el session_id existe y está activo.
    Raises 404 si no existe, 410 si está desactivada.
    """
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "session_id debe ser un UUID válido")

    session = await session_repo.get_session(db, sid)

    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Sesión '{session_id}' no encontrada")

    if not session.is_active:
        raise HTTPException(status.HTTP_410_GONE, f"Sesión '{session_id}' desactivada")

    return session


AgentSession = Annotated[DBSession, Depends(get_agent_session)]


def get_request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID") or str(uuid.uuid4())


RequestID = Annotated[str, Depends(get_request_id)]
