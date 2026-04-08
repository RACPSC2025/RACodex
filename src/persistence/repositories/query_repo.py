"""
Repositorio de QueryLog.

Registra cada consulta del agente con sus métricas para:
  - Análisis de calidad del retrieval (reflection_score promedio)
  - Detección de queries frecuentes (cache de respuestas futuras)
  - Identificar qué documentos se usan más
  - Alertas cuando la calidad baja en un período

El QueryLog se crea SIEMPRE que el agente completa una respuesta,
independientemente de si hubo iteraciones de reflexión.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Optional

from sqlalchemy import Float, cast, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.logging import get_logger
from src.persistence.models import QueryLog

log = get_logger(__name__)


def hash_query(query_text: str) -> str:
    """MD5 de la query para detección de duplicados (no criptográfico)."""
    return hashlib.md5(query_text.lower().strip().encode()).hexdigest()


# ─── Create ───────────────────────────────────────────────────────────────────

async def create_query_log(
    db: AsyncSession,
    *,
    session_id: Optional[uuid.UUID],
    message_id: Optional[uuid.UUID],
    query_text: str,
    active_query: Optional[str] = None,
    retrieval_strategy: Optional[str] = None,
    docs_retrieved: int = 0,
    docs_after_rerank: int = 0,
    top_sources: Optional[list[dict]] = None,
    retrieval_time_ms: Optional[int] = None,
    generation_time_ms: Optional[int] = None,
    total_time_ms: Optional[int] = None,
    reflection_score: Optional[float] = None,
    iteration_count: int = 1,
) -> QueryLog:
    """
    Registra una consulta completada con sus métricas.

    `was_reformulated` se calcula automáticamente comparando
    query_text con active_query.
    """
    was_reformulated = (
        active_query is not None
        and active_query.strip() != query_text.strip()
    )

    query_log = QueryLog(
        session_id=session_id,
        message_id=message_id,
        query_text=query_text,
        query_hash=hash_query(query_text),
        active_query=active_query if was_reformulated else None,
        retrieval_strategy=retrieval_strategy,
        docs_retrieved=docs_retrieved,
        docs_after_rerank=docs_after_rerank,
        top_sources=top_sources,
        retrieval_time_ms=retrieval_time_ms,
        generation_time_ms=generation_time_ms,
        total_time_ms=total_time_ms,
        reflection_score=reflection_score,
        iteration_count=iteration_count,
        was_reformulated=was_reformulated,
    )
    db.add(query_log)
    await db.flush()

    log.debug(
        "query_log_created",
        query_log_id=str(query_log.id),
        score=reflection_score,
        iterations=iteration_count,
        reformulated=was_reformulated,
    )
    return query_log


# ─── Read ─────────────────────────────────────────────────────────────────────

async def get_query_logs_by_session(
    db: AsyncSession,
    session_id: uuid.UUID,
    *,
    limit: int = 50,
) -> list[QueryLog]:
    """Retorna los logs de consulta de una sesión."""
    result = await db.execute(
        select(QueryLog)
        .where(QueryLog.session_id == session_id)
        .order_by(QueryLog.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def find_similar_query(
    db: AsyncSession,
    query_text: str,
) -> Optional[QueryLog]:
    """
    Busca si la misma query (por hash) ya se ejecutó antes.

    Útil para detectar queries repetidas que podrían cachearse.
    Solo retorna la más reciente con reflection_score alto.
    """
    result = await db.execute(
        select(QueryLog)
        .where(QueryLog.query_hash == hash_query(query_text))
        .where(QueryLog.reflection_score >= 0.8)
        .order_by(QueryLog.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# ─── Analytics ────────────────────────────────────────────────────────────────

async def get_quality_metrics(
    db: AsyncSession,
    *,
    last_n_days: int = 7,
) -> dict:
    """
    Métricas de calidad del sistema para los últimos N días.

    Retorna:
      avg_score        — score promedio de reflection
      low_score_pct    — % de queries con score < 0.6
      avg_iterations   — promedio de iteraciones por query
      reformulated_pct — % de queries que necesitaron reformulación
      total_queries    — total de queries en el período
    """
    time_filter = text(
        f"created_at >= NOW() - INTERVAL '{last_n_days} days'"
    )

    result = await db.execute(
        select(
            func.count(QueryLog.id).label("total"),
            func.avg(QueryLog.reflection_score).label("avg_score"),
            func.avg(QueryLog.iteration_count).label("avg_iterations"),
            func.sum(
                cast(QueryLog.was_reformulated, Float)
            ).label("reformulated_count"),
            func.sum(
                cast(QueryLog.reflection_score < 0.6, Float)
            ).label("low_score_count"),
        )
        .where(time_filter)
        .where(QueryLog.reflection_score.isnot(None))
    )

    row = result.one()
    total = row.total or 0

    return {
        "total_queries": total,
        "avg_reflection_score": round(float(row.avg_score or 0), 3),
        "avg_iterations": round(float(row.avg_iterations or 0), 2),
        "reformulated_pct": round(
            (float(row.reformulated_count or 0) / total * 100) if total else 0, 1
        ),
        "low_score_pct": round(
            (float(row.low_score_count or 0) / total * 100) if total else 0, 1
        ),
        "period_days": last_n_days,
    }


async def get_top_queries(
    db: AsyncSession,
    *,
    limit: int = 10,
    last_n_days: int = 30,
) -> list[dict]:
    """
    Queries más frecuentes en el período dado.

    Útil para identificar temas de alta demanda y considerar
    documentos adicionales a indexar en el corpus.
    """
    time_filter = text(
        f"created_at >= NOW() - INTERVAL '{last_n_days} days'"
    )

    result = await db.execute(
        select(
            QueryLog.query_text,
            QueryLog.query_hash,
            func.count(QueryLog.id).label("frequency"),
            func.avg(QueryLog.reflection_score).label("avg_score"),
        )
        .where(time_filter)
        .group_by(QueryLog.query_text, QueryLog.query_hash)
        .order_by(func.count(QueryLog.id).desc())
        .limit(limit)
    )

    return [
        {
            "query": row.query_text[:100],
            "frequency": row.frequency,
            "avg_score": round(float(row.avg_score or 0), 3),
        }
        for row in result.all()
    ]


async def get_strategy_performance(
    db: AsyncSession,
) -> list[dict]:
    """
    Rendimiento por estrategia de retrieval (avg score, avg tiempo).

    Permite comparar hybrid vs full vs vector para decidir
    la estrategia default del EnsembleRetriever.
    """
    result = await db.execute(
        select(
            QueryLog.retrieval_strategy,
            func.count(QueryLog.id).label("total"),
            func.avg(QueryLog.reflection_score).label("avg_score"),
            func.avg(QueryLog.total_time_ms).label("avg_time_ms"),
            func.avg(QueryLog.iteration_count).label("avg_iterations"),
        )
        .where(QueryLog.retrieval_strategy.isnot(None))
        .where(QueryLog.reflection_score.isnot(None))
        .group_by(QueryLog.retrieval_strategy)
        .order_by(func.avg(QueryLog.reflection_score).desc())
    )

    return [
        {
            "strategy": row.retrieval_strategy,
            "total_queries": row.total,
            "avg_score": round(float(row.avg_score or 0), 3),
            "avg_time_ms": round(float(row.avg_time_ms or 0), 0),
            "avg_iterations": round(float(row.avg_iterations or 0), 2),
        }
        for row in result.all()
    ]
