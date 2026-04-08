"""
Schemas Pydantic v2 para la API de Fénix RAG.

Separación de schemas por dirección:
  *Request  — body de entrada (validado por FastAPI automáticamente)
  *Response — estructura de salida (serializada por FastAPI)

Principios:
  - Schemas de entrada y salida siempre separados
  - Todos los campos tienen description para OpenAPI
  - UUIDs como strings en JSON (portable)
  - Timestamps ISO 8601 con timezone
  - PaginatedResponse[T] genérico para todas las listas
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

T = TypeVar("T")


class FenixBaseModel(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class PaginatedResponse(FenixBaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int
    has_more: bool

    @classmethod
    def from_list(cls, items: list[T], total: int, limit: int, offset: int) -> "PaginatedResponse[T]":
        return cls(items=items, total=total, limit=limit, offset=offset, has_more=(offset + limit) < total)


# ── Session ───────────────────────────────────────────────────────────────────

class SessionCreateRequest(FenixBaseModel):
    user_identifier: str = Field(min_length=1, max_length=255, examples=["user@empresa.co"])
    title: Optional[str] = Field(default=None, max_length=500)


class SessionResponse(FenixBaseModel):
    id: str
    user_identifier: str
    title: Optional[str]
    is_active: bool
    total_messages: int
    total_documents: int
    created_at: datetime
    updated_at: datetime

    @field_validator("id", mode="before")
    @classmethod
    def coerce_uuid(cls, v: object) -> str:
        return str(v)


# ── Message / Chat ────────────────────────────────────────────────────────────

class SourceReference(FenixBaseModel):
    source: str
    article: str = ""
    page: str = ""


class ReflectionInfo(FenixBaseModel):
    score: float
    is_grounded: bool
    has_hallucination: bool
    cites_source: bool
    feedback: str = ""


class MessageResponse(FenixBaseModel):
    id: str
    session_id: str
    role: str
    content: str
    sources: Optional[list[SourceReference]] = None
    retrieval_strategy: Optional[str] = None
    reflection_score: Optional[float] = None
    iteration_count: int = 0
    response_time_ms: Optional[int] = None
    created_at: datetime

    @field_validator("id", "session_id", mode="before")
    @classmethod
    def coerce_uuid(cls, v: object) -> str:
        return str(v)


class ChatRequest(FenixBaseModel):
    session_id: str = Field(description="UUID de la sesión activa")
    query: str = Field(min_length=3, max_length=4000)
    max_iterations: int = Field(default=2, ge=1, le=4)

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("La query no puede estar vacía")
        return v.strip()


class ChatResponse(FenixBaseModel):
    message_id: str
    session_id: str
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    retrieval_strategy: str = ""
    iteration_count: int
    reflection: Optional[ReflectionInfo] = None
    response_time_ms: int


class ChatStreamChunk(FenixBaseModel):
    type: str  # 'token' | 'source' | 'done' | 'error'
    content: str = ""
    metadata: Optional[dict] = None


# ── Document ──────────────────────────────────────────────────────────────────

class DocumentIngestResponse(FenixBaseModel):
    document_id: str
    filename: str
    document_type: str
    loader_used: str
    chunk_count: int
    page_count: int
    classifier_confidence: float
    already_indexed: bool

    @field_validator("document_id", mode="before")
    @classmethod
    def coerce_uuid(cls, v: object) -> str:
        return str(v)


class DocumentResponse(FenixBaseModel):
    id: str
    filename: str
    document_type: str
    loader_used: str
    chunk_count: int
    page_count: int
    is_indexed: bool
    classifier_confidence: Optional[float]
    created_at: datetime

    @field_validator("id", mode="before")
    @classmethod
    def coerce_uuid(cls, v: object) -> str:
        return str(v)


class CorpusStatsResponse(FenixBaseModel):
    total_documents: int
    total_chunks: int
    by_document_type: dict[str, int]
    by_loader: dict[str, int]


# ── Health ────────────────────────────────────────────────────────────────────

class ComponentStatus(FenixBaseModel):
    status: str  # 'ok' | 'degraded' | 'error'
    detail: Optional[str] = None


class HealthResponse(FenixBaseModel):
    status: str  # 'healthy' | 'degraded' | 'unhealthy'
    version: str
    components: dict[str, ComponentStatus]
    uptime_seconds: float


# ── Admin ─────────────────────────────────────────────────────────────────────

class QualityMetricsResponse(FenixBaseModel):
    total_queries: int
    avg_reflection_score: float
    avg_iterations: float
    reformulated_pct: float
    low_score_pct: float
    period_days: int


class StrategyPerformanceItem(FenixBaseModel):
    strategy: str
    total_queries: int
    avg_score: float
    avg_time_ms: float
    avg_iterations: float


class TopQueryItem(FenixBaseModel):
    query: str
    frequency: int
    avg_score: float


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(FenixBaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None
