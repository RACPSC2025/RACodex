"""add user_profiles and reflection_logs tables

Revision ID: 002_user_profiles
Revises: 001_initial_schema
Create Date: 2026-04-12 00:00:00.000000

Fase 10A — Paso 2/10:
  - reflection_logs: auditoría de ciclos de auto-reflexión (modelo faltante)
  - user_profiles: preferencias de perfil de agente por usuario

Diseño:
  - reflection_logs vincula a sessions/messages via FK con SET NULL.
  - user_profiles NO tiene FK a tabla users (no existe hasta Fase 14).
    Vincula por user_identifier (string) — mismo campo que Session.
  - Ambos reutilizan la función update_updated_at_column() creada en 001.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "002_user_profiles"
down_revision: str | None = "001_initial_schema"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # ── reflection_logs ───────────────────────────────────────────────────────
    op.create_table(
        "reflection_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
            comment="FK → sessions (auditoría por sesión)",
        ),
        sa.Column(
            "message_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
            comment="FK → messages (auditoría por mensaje)",
        ),
        sa.Column(
            "score",
            sa.Float(),
            nullable=True,
            comment="Score de reflexión (0.0-1.0)",
        ),
        sa.Column(
            "is_grounded",
            sa.Boolean(),
            nullable=True,
            comment="¿Respuesta fundamentada en docs?",
        ),
        sa.Column(
            "has_hallucination",
            sa.Boolean(),
            nullable=True,
            comment="¿Hay invención detectada?",
        ),
        sa.Column(
            "feedback",
            sa.Text(),
            nullable=True,
            comment="Qué mejorar en la próxima iteración",
        ),
        sa.Column(
            "reformulated_query",
            sa.Text(),
            nullable=True,
            comment="Query reformulada para re-retrieval",
        ),
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

    op.create_index(
        "ix_reflection_logs_session", "reflection_logs", ["session_id", "created_at"]
    )
    op.create_index(
        "ix_reflection_logs_score", "reflection_logs", ["score"]
    )

    # Reutiliza la función update_updated_at_column() creada en 001
    op.execute("""
        CREATE TRIGGER trigger_reflection_logs_updated_at
        BEFORE UPDATE ON reflection_logs
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """)

    # ── user_profiles ─────────────────────────────────────────────────────────
    op.create_table(
        "user_profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "user_identifier",
            sa.String(255),
            nullable=False,
            comment="Email o JWT sub — mismo campo que Session.user_identifier",
        ),
        sa.Column(
            "preferred_profile",
            sa.String(100),
            nullable=False,
            server_default="general-dev",
            comment="Nombre del skill pack activo. Debe existir en registry.json",
        ),
        sa.Column(
            "custom_system_prompt",
            sa.Text(),
            nullable=True,
            comment="Override del system prompt. None = usar INDEX.md del pack",
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default="true",
            comment="False = ignorar y usar default del registry",
        ),
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
        sa.UniqueConstraint("user_identifier", name="uq_user_profiles_identifier"),
    )

    op.create_index(
        "ix_user_profiles_identifier", "user_profiles", ["user_identifier"]
    )
    op.create_index(
        "ix_user_profiles_profile", "user_profiles", ["preferred_profile"]
    )

    # Reutiliza la función update_updated_at_column() creada en 001
    op.execute("""
        CREATE TRIGGER trigger_user_profiles_updated_at
        BEFORE UPDATE ON user_profiles
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """)


def downgrade() -> None:
    # ── Triggers ──────────────────────────────────────────────────────────────
    op.execute(
        "DROP TRIGGER IF EXISTS trigger_user_profiles_updated_at ON user_profiles;"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS trigger_reflection_logs_updated_at ON reflection_logs;"
    )

    # ── Tablas ────────────────────────────────────────────────────────────────
    op.drop_table("user_profiles")
    op.drop_table("reflection_logs")
