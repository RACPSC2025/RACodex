"""
Modelos SQLAlchemy 2 para Fénix RAG.

Entidades:
  Session     — conversación de usuario (agrupa mensajes y documentos)
  Message     — mensaje individual dentro de una sesión
  Document    — registro de documento indexado (PDF, Word, Excel, etc.)
  Chunk       — chunk individual indexado en Chroma con su metadata
  QueryLog    — log de consultas para análisis de retrieval
  ReflectionLog — historial de ciclos de auto-reflexión por sesión

Principios:
  - UUIDs como PKs en entidades públicas (sin enumeración)
  - Timestamps automáticos (created_at / updated_at) en todas las tablas
  - Soft-delete en Session y Document (deleted_at)
  - Índices en todas las columnas de búsqueda frecuente
  - Constraints explícitos (no solo en el código de aplicación)

Por qué SQLAlchemy 2 con DeclarativeBase y mapped_column:
  - Type-safe: `mapped_column(String)` en vez de `Column(String)`
  - Integración con mypy mediante `MappedColumn` y `Mapped[T]`
  - `relationship()` con `lazy="select"` explícito (no lazy default)
  - Async-compatible via `AsyncSession` + `AsyncEngine`
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ─── Base ─────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Base declarativa con timestamps automáticos."""
    pass


# ─── Mixin de timestamps ──────────────────────────────────────────────────────

class TimestampMixin:
    """created_at / updated_at automáticos para todas las tablas."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# ─── Session ──────────────────────────────────────────────────────────────────

class Session(TimestampMixin, Base):
    """
    Sesión de conversación de un usuario.

    Agrupa todos los mensajes e ingestions de una sesión de trabajo.
    Soft-delete: deleted_at en lugar de DELETE físico (preserva auditoría).
    """

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_identifier: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="ID del usuario (email, sub JWT, o anónimo)",
    )
    title: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Título auto-generado desde la primera query",
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    total_messages: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_documents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Relationships
    messages: Mapped[list[Message]] = relationship(
        "Message",
        back_populates="session",
        lazy="select",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    documents: Mapped[list[Document]] = relationship(
        "Document",
        back_populates="session",
        lazy="select",
        cascade="all, delete-orphan",
    )
    query_logs: Mapped[list[QueryLog]] = relationship(
        "QueryLog",
        back_populates="session",
        lazy="select",
    )

    __table_args__ = (
        Index("ix_sessions_user_active", "user_identifier", "is_active"),
        Index("ix_sessions_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"Session(id={self.id}, user={self.user_identifier!r})"


# ─── Message ──────────────────────────────────────────────────────────────────

class Message(TimestampMixin, Base):
    """
    Mensaje individual dentro de una sesión.

    Almacena tanto la query del usuario como la respuesta del agente.
    Los campos de retrieval y reflexión permiten análisis post-facto
    de la calidad del sistema.
    """

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Contenido
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="'human' | 'assistant' | 'system'",
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Metadata de la respuesta del agente
    sources: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="[{source, article, page}] — documentos usados",
    )
    retrieval_strategy: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )
    iteration_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    reflection_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Timing
    response_time_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Tiempo de respuesta en ms",
    )

    # Relationship
    session: Mapped[Session] = relationship("Session", back_populates="messages")

    __table_args__ = (
        Index("ix_messages_session_created", "session_id", "created_at"),
        Index("ix_messages_role", "role"),
    )

    def __repr__(self) -> str:
        return f"Message(id={self.id}, role={self.role!r}, session={self.session_id})"


# ─── Document ─────────────────────────────────────────────────────────────────

class Document(TimestampMixin, Base):
    """
    Registro de un documento indexado en el sistema.

    Cada vez que el usuario sube un archivo y se indexa en Chroma,
    se crea un registro aquí para:
    - Saber qué documentos tiene indexados cada sesión
    - Evitar re-indexar el mismo archivo
    - Trazabilidad de qué loader y cleaner se usaron
    - Poder eliminar documentos del vector store por ID
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="NULL = documento del corpus global (no de una sesión específica)",
    )

    # Identidad del archivo
    filename: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    file_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA-256 del archivo para detección de duplicados",
    )
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # Clasificación y procesamiento
    document_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="decreto | resolución | circular | ley | contrato | excel | otro",
    )
    loader_used: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="pymupdf | ocr | docling | word | excel",
    )
    cleaner_profile: Mapped[str] = mapped_column(String(50), nullable=False)
    required_ocr: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Resultado del indexado
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chroma_collection: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="Nombre de la colección Chroma donde se indexó",
    )

    # Estado
    is_indexed: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Classifier metadata
    classifier_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    classifier_reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    session: Mapped[Optional[Session]] = relationship(
        "Session", back_populates="documents"
    )
    chunks: Mapped[list[Chunk]] = relationship(
        "Chunk",
        back_populates="document",
        lazy="select",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        # Prevenir re-indexar el mismo archivo en la misma sesión
        UniqueConstraint("session_id", "file_hash", name="uq_document_session_hash"),
        Index("ix_documents_filename", "filename"),
        Index("ix_documents_type", "document_type"),
        Index("ix_documents_indexed", "is_indexed"),
    )

    def __repr__(self) -> str:
        return f"Document(id={self.id}, filename={self.filename!r}, chunks={self.chunk_count})"


# ─── Chunk ────────────────────────────────────────────────────────────────────

class Chunk(TimestampMixin, Base):
    """
    Chunk individual indexado en Chroma.

    Mantiene el registro SQL de cada chunk para:
    - Correlacionar chunks de Chroma con documentos SQL
    - Buscar chunks por metadata (artículo, página) sin ir a Chroma
    - Auditoría de qué texto exacto fue indexado
    - Poder eliminar chunks específicos del vector store

    IMPORTANTE: page_content se almacena truncado (máx 2000 chars)
    para auditoría. El texto completo vive en Chroma.
    """

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Identidad en Chroma
    chroma_id: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        unique=True,
        index=True,
        comment="ID del chunk en Chroma (source::chunk_index)",
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Contenido (truncado para auditoría)
    content_preview: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Primeros 2000 chars del chunk para auditoría",
    )
    content_length: Mapped[int] = mapped_column(Integer, nullable=False)

    # Metadata estructurada (espeja el metadata de Chroma)
    page: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, index=True)
    article_number: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, index=True
    )
    document_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    loader_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Relationship
    document: Mapped[Document] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_index", "document_id", "chunk_index"),
        Index("ix_chunks_article", "article_number"),
        Index("ix_chunks_page", "page"),
    )

    def __repr__(self) -> str:
        return (
            f"Chunk(id={self.id}, doc={self.document_id}, "
            f"index={self.chunk_index}, article={self.article_number!r})"
        )


# ─── QueryLog ─────────────────────────────────────────────────────────────────

class QueryLog(TimestampMixin, Base):
    """
    Log de consultas para análisis de calidad del retrieval.

    Permite:
    - Identificar queries frecuentes (para pre-cache)
    - Analizar qué documentos se recuperan más
    - Medir la eficacia del reranking
    - Detectar queries que producen respuestas de baja calidad
    """

    __tablename__ = "query_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    message_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Query
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="MD5 de la query para detección de duplicados",
    )
    active_query: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Query reformulada por reflection (si difiere de query_text)",
    )

    # Retrieval
    retrieval_strategy: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    docs_retrieved: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    docs_after_rerank: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    top_sources: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="[{source, article, rrf_score}] — top 5 resultados",
    )

    # Performance
    retrieval_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    generation_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Calidad
    reflection_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    iteration_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    was_reformulated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Relationship
    session: Mapped[Optional[Session]] = relationship(
        "Session", back_populates="query_logs"
    )

    __table_args__ = (
        Index("ix_query_logs_hash", "query_hash"),
        Index("ix_query_logs_score", "reflection_score"),
        Index("ix_query_logs_created", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"QueryLog(id={self.id}, "
            f"query={self.query_text[:40]!r}, "
            f"score={self.reflection_score})"
        )
