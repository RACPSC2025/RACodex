"""initial schema

Revision ID: 001_initial_schema
Revises:
Create Date: 2025-01-01 00:00:00.000000

Crea las 5 tablas principales del sistema:
  - sessions
  - messages
  - documents
  - chunks
  - query_logs

Con todos sus índices, constraints y relaciones.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# Identificadores de migración
revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # ── sessions ──────────────────────────────────────────────────────────────
    op.create_table(
        "sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_identifier", sa.String(255), nullable=False),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_messages", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_documents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sessions_user_active", "sessions", ["user_identifier", "is_active"])
    op.create_index("ix_sessions_created_at", "sessions", ["created_at"])
    op.create_index("ix_sessions_user_identifier", "sessions", ["user_identifier"])

    # ── messages ──────────────────────────────────────────────────────────────
    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sources", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("retrieval_strategy", sa.String(50), nullable=True),
        sa.Column("iteration_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reflection_score", sa.Float(), nullable=True),
        sa.Column("response_time_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"], ["sessions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_messages_session_created", "messages", ["session_id", "created_at"])
    op.create_index("ix_messages_role", "messages", ["role"])
    op.create_index("ix_messages_session_id", "messages", ["session_id"])

    # ── documents ─────────────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column("file_hash", sa.String(64), nullable=False),
        sa.Column("mime_type", sa.String(100), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("document_type", sa.String(50), nullable=False),
        sa.Column("loader_used", sa.String(50), nullable=False),
        sa.Column("cleaner_profile", sa.String(50), nullable=False),
        sa.Column("required_ocr", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("page_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chroma_collection", sa.String(200), nullable=False),
        sa.Column("is_indexed", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("classifier_confidence", sa.Float(), nullable=True),
        sa.Column("classifier_reasoning", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"], ["sessions.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "file_hash", name="uq_document_session_hash"),
    )
    op.create_index("ix_documents_filename", "documents", ["filename"])
    op.create_index("ix_documents_type", "documents", ["document_type"])
    op.create_index("ix_documents_indexed", "documents", ["is_indexed"])
    op.create_index("ix_documents_file_hash", "documents", ["file_hash"])
    op.create_index("ix_documents_session_id", "documents", ["session_id"])

    # ── chunks ────────────────────────────────────────────────────────────────
    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chroma_id", sa.String(512), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content_preview", sa.Text(), nullable=False),
        sa.Column("content_length", sa.Integer(), nullable=False),
        sa.Column("page", sa.String(20), nullable=True),
        sa.Column("article_number", sa.String(50), nullable=True),
        sa.Column("document_type", sa.String(50), nullable=True),
        sa.Column("loader_type", sa.String(50), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chroma_id", name="uq_chunks_chroma_id"),
    )
    op.create_index("ix_chunks_document_index", "chunks", ["document_id", "chunk_index"])
    op.create_index("ix_chunks_article", "chunks", ["article_number"])
    op.create_index("ix_chunks_page", "chunks", ["page"])
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])

    # ── query_logs ────────────────────────────────────────────────────────────
    op.create_table(
        "query_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("query_hash", sa.String(64), nullable=False),
        sa.Column("active_query", sa.Text(), nullable=True),
        sa.Column("retrieval_strategy", sa.String(50), nullable=True),
        sa.Column("docs_retrieved", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("docs_after_rerank", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("top_sources", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("retrieval_time_ms", sa.Integer(), nullable=True),
        sa.Column("generation_time_ms", sa.Integer(), nullable=True),
        sa.Column("total_time_ms", sa.Integer(), nullable=True),
        sa.Column("reflection_score", sa.Float(), nullable=True),
        sa.Column("iteration_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("was_reformulated", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"], ["sessions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["message_id"], ["messages.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_query_logs_hash", "query_logs", ["query_hash"])
    op.create_index("ix_query_logs_score", "query_logs", ["reflection_score"])
    op.create_index("ix_query_logs_created", "query_logs", ["created_at"])
    op.create_index("ix_query_logs_session_id", "query_logs", ["session_id"])

    # ── Trigger: updated_at automático ────────────────────────────────────────
    # PostgreSQL no tiene auto-update de columnas, necesitamos un trigger
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)

    for table in ["sessions", "messages", "documents", "chunks", "query_logs"]:
        op.execute(f"""
            CREATE TRIGGER trigger_{table}_updated_at
            BEFORE UPDATE ON {table}
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """)


def downgrade() -> None:
    # Eliminar triggers primero
    for table in ["sessions", "messages", "documents", "chunks", "query_logs"]:
        op.execute(f"DROP TRIGGER IF EXISTS trigger_{table}_updated_at ON {table};")

    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column();")

    # Eliminar tablas en orden inverso (por FKs)
    op.drop_table("query_logs")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("messages")
    op.drop_table("sessions")
