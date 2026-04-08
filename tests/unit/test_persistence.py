"""
Tests unitarios para el módulo de persistencia.

Estrategia:
  - Tests de modelos: constructores, repr, relaciones
  - Tests de repos: lógica sin BD real (SQLite en memoria para integración)
  - Tests de database.py: engines, context managers
  - Tests de checkpointer: selección por entorno

Para tests de integración completos con PostgreSQL real:
  ver tests/integration/test_persistence_integration.py (Fase 6)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.persistence.models import Base, Chunk, Document, Message, QueryLog, Session


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_session(**kwargs) -> Session:
    """Construye una Session con valores por defecto."""
    defaults = {
        "id": uuid.uuid4(),
        "user_identifier": "user@example.com",
        "title": "Test session",
        "is_active": True,
        "total_messages": 0,
        "total_documents": 0,
    }
    defaults.update(kwargs)
    return Session(**defaults)


def make_document(session_id: uuid.UUID | None = None, **kwargs) -> Document:
    """Construye un Document con valores por defecto."""
    defaults = {
        "id": uuid.uuid4(),
        "session_id": session_id,
        "filename": "decreto_1072.pdf",
        "file_hash": "a" * 64,
        "mime_type": "application/pdf",
        "file_size_bytes": 1024 * 512,
        "document_type": "decreto",
        "loader_used": "pymupdf",
        "cleaner_profile": "legal_colombia",
        "required_ocr": False,
        "chunk_count": 42,
        "page_count": 10,
        "chroma_collection": "fenix_legal",
        "is_indexed": True,
    }
    defaults.update(kwargs)
    return Document(**defaults)


def make_message(session_id: uuid.UUID, **kwargs) -> Message:
    """Construye un Message con valores por defecto."""
    defaults = {
        "id": uuid.uuid4(),
        "session_id": session_id,
        "role": "human",
        "content": "¿Qué dice el artículo 1?",
        "iteration_count": 0,
    }
    defaults.update(kwargs)
    return Message(**defaults)


def make_query_log(session_id: uuid.UUID | None = None, **kwargs) -> QueryLog:
    defaults = {
        "id": uuid.uuid4(),
        "session_id": session_id,
        "query_text": "¿Cuáles son las obligaciones del empleador en SST?",
        "query_hash": "abc123",
        "docs_retrieved": 8,
        "docs_after_rerank": 5,
        "reflection_score": 0.85,
        "iteration_count": 1,
        "was_reformulated": False,
    }
    defaults.update(kwargs)
    return QueryLog(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Models
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionModel:
    def test_session_repr(self) -> None:
        session = make_session(user_identifier="alice@test.com")
        assert "alice@test.com" in repr(session)

    def test_session_default_is_active(self) -> None:
        session = make_session()
        assert session.is_active is True

    def test_session_soft_delete_field_exists(self) -> None:
        session = make_session()
        assert hasattr(session, "deleted_at")
        assert session.deleted_at is None

    def test_session_counters_start_at_zero(self) -> None:
        session = make_session()
        assert session.total_messages == 0
        assert session.total_documents == 0

    def test_session_uuid_pk(self) -> None:
        session = make_session()
        assert isinstance(session.id, uuid.UUID)


class TestMessageModel:
    def test_message_roles(self) -> None:
        session_id = uuid.uuid4()
        human = make_message(session_id, role="human")
        assistant = make_message(session_id, role="assistant")
        assert human.role == "human"
        assert assistant.role == "assistant"

    def test_message_repr(self) -> None:
        msg = make_message(uuid.uuid4(), role="human")
        assert "human" in repr(msg)

    def test_message_optional_fields_none(self) -> None:
        msg = make_message(uuid.uuid4())
        assert msg.sources is None
        assert msg.retrieval_strategy is None
        assert msg.reflection_score is None
        assert msg.response_time_ms is None

    def test_message_sources_as_list(self) -> None:
        sources = [{"source": "decreto.pdf", "article": "1", "page": "5"}]
        msg = make_message(uuid.uuid4(), sources=sources)
        assert msg.sources == sources


class TestDocumentModel:
    def test_document_repr(self) -> None:
        doc = make_document(chunk_count=42)
        assert "42" in repr(doc)
        assert "decreto_1072.pdf" in repr(doc)

    def test_document_is_indexed_default(self) -> None:
        doc = make_document()
        assert doc.is_indexed is True

    def test_document_soft_delete_field(self) -> None:
        doc = make_document()
        assert doc.deleted_at is None

    def test_document_required_ocr_false(self) -> None:
        doc = make_document(required_ocr=False)
        assert doc.required_ocr is False

    def test_document_ocr_required(self) -> None:
        doc = make_document(required_ocr=True)
        assert doc.required_ocr is True


class TestChunkModel:
    def test_chunk_repr(self) -> None:
        doc_id = uuid.uuid4()
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            chroma_id="decreto.pdf::5",
            chunk_index=5,
            content_preview="ARTÍCULO 5. Texto del artículo.",
            content_length=31,
            article_number="5",
            page="10",
        )
        assert "5" in repr(chunk)

    def test_chunk_article_number_optional(self) -> None:
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            chroma_id="doc::0",
            chunk_index=0,
            content_preview="texto",
            content_length=5,
        )
        assert chunk.article_number is None
        assert chunk.page is None


class TestQueryLogModel:
    def test_query_log_repr(self) -> None:
        log_entry = make_query_log(
            query_text="¿Qué dice el artículo 1?",
            reflection_score=0.9,
        )
        assert "0.9" in repr(log_entry)

    def test_query_log_was_reformulated_false(self) -> None:
        log_entry = make_query_log(was_reformulated=False)
        assert log_entry.was_reformulated is False

    def test_query_log_optional_times(self) -> None:
        log_entry = make_query_log()
        assert log_entry.retrieval_time_ms is None
        assert log_entry.generation_time_ms is None
        assert log_entry.total_time_ms is None


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Repository helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueryRepo:
    def test_hash_query_deterministic(self) -> None:
        from src.persistence.repositories.query_repo import hash_query
        q = "¿Qué es el COPASST?"
        assert hash_query(q) == hash_query(q)

    def test_hash_query_case_insensitive(self) -> None:
        from src.persistence.repositories.query_repo import hash_query
        assert hash_query("COPASST") == hash_query("copasst")

    def test_hash_query_strips_whitespace(self) -> None:
        from src.persistence.repositories.query_repo import hash_query
        assert hash_query("  query  ") == hash_query("query")

    def test_hash_query_different_queries_different_hash(self) -> None:
        from src.persistence.repositories.query_repo import hash_query
        assert hash_query("query A") != hash_query("query B")

    def test_hash_query_returns_32_chars(self) -> None:
        from src.persistence.repositories.query_repo import hash_query
        # MD5 = 32 hex chars
        assert len(hash_query("test")) == 32


class TestDocumentRepo:
    def test_compute_file_hash_returns_64_chars(self, tmp_path) -> None:
        from src.persistence.repositories.document_repo import compute_file_hash
        f = tmp_path / "test.pdf"
        f.write_bytes(b"contenido del archivo de prueba")
        file_hash = compute_file_hash(str(f))
        # SHA-256 = 64 hex chars
        assert len(file_hash) == 64

    def test_compute_file_hash_deterministic(self, tmp_path) -> None:
        from src.persistence.repositories.document_repo import compute_file_hash
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 contenido fijo")
        h1 = compute_file_hash(str(f))
        h2 = compute_file_hash(str(f))
        assert h1 == h2

    def test_compute_file_hash_different_content(self, tmp_path) -> None:
        from src.persistence.repositories.document_repo import compute_file_hash
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert compute_file_hash(str(f1)) != compute_file_hash(str(f2))


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Database infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatabase:
    @pytest.fixture(autouse=True)
    def reset_engine(self):
        """Reset cached engines before each test."""
        import src.persistence.database as db_module
        original_async = db_module._async_engine
        original_sync = db_module._sync_engine
        db_module._async_engine = None
        db_module._sync_engine = None
        yield
        db_module._async_engine = original_async
        db_module._sync_engine = original_sync

    def test_get_async_engine_creates_engine(self) -> None:
        from src.persistence.database import get_async_engine

        with patch("src.persistence.database.create_async_engine") as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine
            engine = get_async_engine()

        assert engine is mock_engine
        mock_create.assert_called_once()

    def test_get_async_engine_singleton(self) -> None:
        from src.persistence.database import get_async_engine

        with patch("src.persistence.database.create_async_engine") as mock_create:
            mock_create.return_value = MagicMock()
            e1 = get_async_engine()
            e2 = get_async_engine()

        assert e1 is e2
        mock_create.assert_called_once()  # solo una vez

    def test_async_url_conversion(self) -> None:
        """La URL psycopg debe convertirse a asyncpg."""
        from src.persistence.database import get_async_engine
        from src.config.settings import get_settings

        captured_url = {}

        def capture_create(url, **kwargs):
            captured_url["url"] = url
            return MagicMock()

        with patch("src.persistence.database.create_async_engine", side_effect=capture_create):
            get_async_engine()

        assert "asyncpg" in captured_url.get("url", "")
        assert "psycopg" not in captured_url.get("url", "")

    @pytest.mark.asyncio
    async def test_transaction_context_manager_commits(self) -> None:
        from src.persistence.database import transaction

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("src.persistence.database.get_session_factory", return_value=mock_factory):
            async with transaction() as db:
                assert db is mock_session

        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_transaction_rolls_back_on_exception(self) -> None:
        from src.persistence.database import transaction

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("src.persistence.database.get_session_factory", return_value=mock_factory):
            with pytest.raises(ValueError):
                async with transaction() as db:
                    raise ValueError("error de prueba")

        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_health_check_returns_ok_on_success(self) -> None:
        from src.persistence.database import check_database_connectivity

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock()
        mock_session.close = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch("src.persistence.database.get_session_factory", return_value=mock_factory):
            result = await check_database_connectivity()

        assert result["database"] == "ok"

    @pytest.mark.asyncio
    async def test_health_check_returns_error_on_failure(self) -> None:
        from src.persistence.database import check_database_connectivity

        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(
            side_effect=Exception("connection refused")
        )

        with patch("src.persistence.database.get_session_factory", return_value=mock_factory):
            result = await check_database_connectivity()

        assert result["database"] == "error"
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Checkpointer
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckpointer:
    def test_get_checkpointer_dev_uses_sqlite(self) -> None:
        from src.persistence.checkpointer import get_checkpointer

        with patch("src.persistence.checkpointer.get_sqlite_checkpointer") as mock_sqlite, \
             patch("src.persistence.checkpointer.get_settings") as mock_settings:
            mock_settings.return_value.is_production = False
            mock_settings.return_value.app_env = "development"
            mock_settings.return_value.storage_dir.return_value = MagicMock()
            get_checkpointer()

        mock_sqlite.assert_called_once()

    def test_get_checkpointer_prod_uses_postgres(self) -> None:
        from src.persistence.checkpointer import get_checkpointer

        with patch("src.persistence.checkpointer.get_postgres_checkpointer") as mock_pg, \
             patch("src.persistence.checkpointer.get_settings") as mock_settings:
            mock_settings.return_value.is_production = True
            mock_settings.return_value.app_env = "production"
            get_checkpointer()

        mock_pg.assert_called_once()

    def test_force_sqlite_overrides_production(self) -> None:
        from src.persistence.checkpointer import get_checkpointer

        with patch("src.persistence.checkpointer.get_sqlite_checkpointer") as mock_sqlite, \
             patch("src.persistence.checkpointer.get_postgres_checkpointer") as mock_pg, \
             patch("src.persistence.checkpointer.get_settings") as mock_settings:
            mock_settings.return_value.is_production = True
            mock_settings.return_value.storage_dir = MagicMock()
            get_checkpointer(force_sqlite=True)

        mock_sqlite.assert_called_once()
        mock_pg.assert_not_called()

    def test_missing_sqlite_dep_raises_clear_error(self, tmp_path) -> None:
        from src.persistence.checkpointer import get_sqlite_checkpointer

        with patch("builtins.__import__", side_effect=lambda name, *a, **k: (
            (_ for _ in ()).throw(ImportError("not installed"))
            if "langgraph.checkpoint.sqlite" in name
            else __import__(name, *a, **k)
        )):
            with pytest.raises(ImportError) as exc_info:
                get_sqlite_checkpointer(tmp_path / "test.db")

        assert "pip install" in str(exc_info.value)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Repository functions (con DB mockeada)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionRepoFunctions:
    @pytest.mark.asyncio
    async def test_create_session_adds_to_db(self) -> None:
        from src.persistence.repositories.session_repo import create_session

        mock_db = AsyncMock(spec=AsyncSession)
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        session = await create_session(
            mock_db,
            user_identifier="test@example.com",
            title="Mi sesión",
        )

        mock_db.add.assert_called_once()
        mock_db.flush.assert_called_once()
        assert session.user_identifier == "test@example.com"
        assert session.title == "Mi sesión"

    @pytest.mark.asyncio
    async def test_create_message_adds_to_db(self) -> None:
        from src.persistence.repositories.session_repo import create_message

        mock_db = AsyncMock(spec=AsyncSession)
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        session_id = uuid.uuid4()
        msg = await create_message(
            mock_db,
            session_id=session_id,
            role="assistant",
            content="Según el artículo 1...",
            reflection_score=0.88,
            iteration_count=1,
        )

        mock_db.add.assert_called_once()
        assert msg.role == "assistant"
        assert msg.reflection_score == 0.88

    @pytest.mark.asyncio
    async def test_create_query_log_detects_reformulation(self) -> None:
        from src.persistence.repositories.query_repo import create_query_log

        mock_db = AsyncMock(spec=AsyncSession)
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        log_entry = await create_query_log(
            mock_db,
            session_id=None,
            message_id=None,
            query_text="pregunta original",
            active_query="pregunta reformulada",  # diferente → was_reformulated=True
            reflection_score=0.75,
        )

        assert log_entry.was_reformulated is True
        assert log_entry.active_query == "pregunta reformulada"

    @pytest.mark.asyncio
    async def test_create_query_log_no_reformulation(self) -> None:
        from src.persistence.repositories.query_repo import create_query_log

        mock_db = AsyncMock(spec=AsyncSession)
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        log_entry = await create_query_log(
            mock_db,
            session_id=None,
            message_id=None,
            query_text="misma query",
            active_query="misma query",  # igual → was_reformulated=False
            reflection_score=0.90,
        )

        assert log_entry.was_reformulated is False
        assert log_entry.active_query is None  # no se guarda si son iguales

    @pytest.mark.asyncio
    async def test_create_chunks_bulk_empty_list(self) -> None:
        from src.persistence.repositories.document_repo import create_chunks_bulk

        mock_db = AsyncMock(spec=AsyncSession)
        count = await create_chunks_bulk(mock_db, uuid.uuid4(), [])

        assert count == 0
        mock_db.add_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_chunks_bulk_truncates_content(self) -> None:
        from src.persistence.repositories.document_repo import create_chunks_bulk

        mock_db = AsyncMock(spec=AsyncSession)
        mock_db.add_all = MagicMock()
        mock_db.flush = AsyncMock()

        long_content = "A" * 5000  # mayor que el límite de 2000

        chunks_data = [{
            "chroma_id": "doc::0",
            "chunk_index": 0,
            "content_preview": long_content,
            "content_length": len(long_content),
            "article_number": "1",
        }]

        await create_chunks_bulk(mock_db, uuid.uuid4(), chunks_data)

        # Verificar que el Chunk creado tiene content_preview truncado
        call_args = mock_db.add_all.call_args[0][0]
        assert len(call_args[0].content_preview) <= 2000
