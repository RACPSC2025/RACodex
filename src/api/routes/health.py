"""Endpoint /health — estado de todos los componentes del sistema."""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, status

from src.api.schemas import ComponentStatus, HealthResponse
from src.config.logging import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["Health"])

_START_TIME = time.time()
_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse, summary="Estado del sistema")
async def health_check() -> HealthResponse:
    """
    Verifica conectividad de todos los componentes.
    status='healthy' → todo OK | 'degraded' → algún componente falla no-crítico
    | 'unhealthy' → BD caída.
    """
    components: dict[str, ComponentStatus] = {}

    # PostgreSQL
    try:
        from src.persistence.database import check_database_connectivity  # noqa: PLC0415
        r = await check_database_connectivity()
        components["database"] = ComponentStatus(
            status="ok" if r.get("database") == "ok" else "error",
            detail=r.get("error"),
        )
    except Exception as exc:
        components["database"] = ComponentStatus(status="error", detail=str(exc))

    # Chroma
    try:
        from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415
        store = get_vector_store()
        count = store.count() if store.is_initialized else 0
        components["vector_store"] = ComponentStatus(
            status="ok" if store.is_initialized else "degraded",
            detail=f"{count} chunks indexados",
        )
    except Exception as exc:
        components["vector_store"] = ComponentStatus(status="error", detail=str(exc))

    # Bedrock
    try:
        from src.config.providers import check_bedrock_connectivity  # noqa: PLC0415
        r = check_bedrock_connectivity()
        components["bedrock"] = ComponentStatus(
            status="ok" if r.get("bedrock") else "error",
            detail=r.get("error") or r.get("region"),
        )
    except Exception as exc:
        components["bedrock"] = ComponentStatus(status="error", detail=str(exc))

    statuses = [c.status for c in components.values()]
    if all(s == "ok" for s in statuses):
        overall = "healthy"
    elif components.get("database", ComponentStatus(status="ok")).status == "error":
        overall = "unhealthy"
    else:
        overall = "degraded"

    return HealthResponse(
        status=overall,
        version=_VERSION,
        components=components,
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@router.get("/health/ready", summary="Readiness probe (Kubernetes)")
async def readiness() -> dict:
    try:
        from src.persistence.database import check_database_connectivity  # noqa: PLC0415
        r = await check_database_connectivity()
        if r.get("database") == "ok":
            return {"ready": True}
    except Exception:
        pass
    raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Not ready")


@router.get("/health/live", summary="Liveness probe (Kubernetes)")
async def liveness() -> dict:
    return {"alive": True}
