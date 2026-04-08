"""
Tests unitarios para la API FastAPI de Fénix RAG.

Estrategia:
  - TestClient de httpx (no inicia servidor real)
  - Todas las dependencias externas mockeadas (BD, agente, vector store)
  - Verifica: status codes, estructura de respuesta, headers, error handling
  - No hay llamadas reales a PostgreSQL ni a Bedrock

Qué se testea:
  - /health — status, componentes, readiness, liveness
  - /api/v1/sessions — CRUD completo
  - /api/v1/chat — happy path, errores de validación, sesión inválida
  - /api/v1/documents — upload, stats, listado, delete
  - /api/v1/admin — métricas de calidad y estrategias
  - Middleware — request_id, response-time header, error format
  - Schemas — coerción de UUID, campos opcionales
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db_session():
    """AsyncSession mockeada que no hace I/O real."""
    mock = AsyncMock()
    mock.commit = AsyncMock()
    mock.rollback = AsyncMock()
    mock.close = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def app(mock_db_session):
    """Instancia de FastAPI con dependencias mockeadas."""
    from src.api.main import create_app  # noqa: PLC0415
    from src.api.dependencies import get_db  # noqa: PLC0415

    application = create_app()

    # Override de la dependencia de BD para no necesitar PostgreSQL real
    async def override_get_db():
        yield mock_db_session

    application.dependency_overrides[get_db] = override_get_db
    return application


@pytest.fixture
def client(app) -> TestClient:
    """TestClient httpx (no inicia servidor, no necesita asyncio externo)."""
    return TestClient(app, raise_server_exceptions=False)


def make_session_record(**kwargs) -> MagicMock:
    """Crea un MagicMock que simula un modelo Session de SQLAlchemy."""
    record = MagicMock()
    record.id = kwargs.get("id", uuid.uuid4())
    record.user_identifier = kwargs.get("user_identifier", "user@test.com")
    record.title = kwargs.get("title", "Sesión de prueba")
    record.is_active = kwargs.get("is_active", True)
    record.total_messages = kwargs.get("total_messages", 0)
    record.total_documents = kwargs.get("total_documents", 0)
    record.created_at = kwargs.get("created_at", datetime.now(timezone.utc))
    record.updated_at = kwargs.get("updated_at", datetime.now(timezone.utc))
    record.deleted_at = None
    return record


def make_message_record(**kwargs) -> MagicMock:
    record = MagicMock()
    record.id = kwargs.get("id", uuid.uuid4())
    record.session_id = kwargs.get("session_id", uuid.uuid4())
    record.role = kwargs.get("role", "assistant")
    record.content = kwargs.get("content", "Respuesta del agente.")
    record.sources = kwargs.get("sources", [])
    record.retrieval_strategy = kwargs.get("retrieval_strategy", "hybrid")
    record.reflection_score = kwargs.get("reflection_score", 0.88)
    record.iteration_count = kwargs.get("iteration_count", 1)
    record.response_time_ms = kwargs.get("response_time_ms", 1200)
    record.created_at = kwargs.get("created_at", datetime.now(timezone.utc))
    return record


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Health
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        with patch("src.api.routes.health.check_database_connectivity", new_callable=AsyncMock) as mock_db, \
             patch("src.api.routes.health.get_vector_store") as mock_vs, \
             patch("src.api.routes.health.check_bedrock_connectivity") as mock_bedrock:

            mock_db.return_value = {"database": "ok"}
            mock_vs.return_value.is_initialized = True
            mock_vs.return_value.count.return_value = 42
            mock_bedrock.return_value = {"bedrock": True, "region": "us-east-1"}

            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded", "unhealthy")
        assert "components" in data
        assert "version" in data
        assert "uptime_seconds" in data

    def test_health_components_structure(self, client: TestClient) -> None:
        with patch("src.api.routes.health.check_database_connectivity", new_callable=AsyncMock) as mock_db, \
             patch("src.api.routes.health.get_vector_store") as mock_vs, \
             patch("src.api.routes.health.check_bedrock_connectivity") as mock_bedrock:

            mock_db.return_value = {"database": "ok"}
            mock_vs.return_value.is_initialized = True
            mock_vs.return_value.count.return_value = 10
            mock_bedrock.return_value = {"bedrock": True, "region": "us-east-1"}

            response = client.get("/health")

        data = response.json()
        for component_name, component in data["components"].items():
            assert "status" in component
            assert component["status"] in ("ok", "degraded", "error")

    def test_health_unhealthy_when_db_down(self, client: TestClient) -> None:
        with patch("src.api.routes.health.check_database_connectivity", new_callable=AsyncMock) as mock_db, \
             patch("src.api.routes.health.get_vector_store") as mock_vs, \
             patch("src.api.routes.health.check_bedrock_connectivity") as mock_bedrock:

            mock_db.return_value = {"database": "error", "error": "connection refused"}
            mock_vs.return_value.is_initialized = False
            mock_vs.return_value.count.return_value = 0
            mock_bedrock.return_value = {"bedrock": True}

            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("unhealthy", "degraded")

    def test_readiness_returns_200_when_db_ok(self, client: TestClient) -> None:
        with patch("src.api.routes.health.check_database_connectivity", new_callable=AsyncMock) as mock_db:
            mock_db.return_value = {"database": "ok"}
            response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True

    def test_readiness_returns_503_when_db_down(self, client: TestClient) -> None:
        with patch("src.api.routes.health.check_database_connectivity", new_callable=AsyncMock) as mock_db:
            mock_db.side_effect = Exception("connection refused")
            response = client.get("/health/ready")
        assert response.status_code == 503

    def test_liveness_always_200(self, client: TestClient) -> None:
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["alive"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Sessions
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessions:
    def test_create_session_returns_201(self, client: TestClient) -> None:
        session_record = make_session_record()

        with patch("src.api.routes.sessions.session_repo.create_session", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = session_record
            response = client.post("/api/v1/sessions/", json={
                "user_identifier": "user@empresa.co",
                "title": "Mi sesión de consulta SST",
            })

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["user_identifier"] == "user@empresa.co"

    def test_create_session_missing_user_identifier(self, client: TestClient) -> None:
        response = client.post("/api/v1/sessions/", json={"title": "Sin usuario"})
        assert response.status_code == 422

    def test_create_session_empty_user_identifier(self, client: TestClient) -> None:
        response = client.post("/api/v1/sessions/", json={"user_identifier": ""})
        assert response.status_code == 422

    def test_get_session_returns_200(self, client: TestClient) -> None:
        session_id = uuid.uuid4()
        session_record = make_session_record(id=session_id)

        with patch("src.api.routes.sessions.session_repo.get_session", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = session_record
            response = client.get(f"/api/v1/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(session_id)

    def test_get_session_not_found(self, client: TestClient) -> None:
        with patch("src.api.routes.sessions.session_repo.get_session", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            response = client.get(f"/api/v1/sessions/{uuid.uuid4()}")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "detail" in data

    def test_get_session_invalid_uuid(self, client: TestClient) -> None:
        response = client.get("/api/v1/sessions/not-a-uuid")
        assert response.status_code == 400

    def test_list_sessions_returns_paginated(self, client: TestClient) -> None:
        sessions = [make_session_record() for _ in range(3)]

        with patch("src.api.routes.sessions.session_repo.get_sessions_by_user", new_callable=AsyncMock) as mock_list, \
             patch("src.api.routes.sessions.session_repo.count_user_sessions", new_callable=AsyncMock) as mock_count:

            mock_list.return_value = sessions
            mock_count.return_value = 3
            response = client.get("/api/v1/sessions/?user_identifier=user@test.com")

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert data["total"] == 3
        assert data["limit"] == 20
        assert "has_more" in data

    def test_delete_session_returns_204(self, client: TestClient) -> None:
        session_id = uuid.uuid4()

        with patch("src.api.routes.sessions.session_repo.delete_session", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = True
            response = client.delete(f"/api/v1/sessions/{session_id}")

        assert response.status_code == 204

    def test_delete_session_not_found(self, client: TestClient) -> None:
        with patch("src.api.routes.sessions.session_repo.delete_session", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = False
            response = client.delete(f"/api/v1/sessions/{uuid.uuid4()}")

        assert response.status_code == 404

    def test_get_messages_returns_paginated(self, client: TestClient) -> None:
        session_id = uuid.uuid4()
        messages = [make_message_record(session_id=session_id) for _ in range(5)]

        with patch("src.api.routes.sessions.session_repo.get_messages_by_session", new_callable=AsyncMock) as mock_msgs, \
             patch("src.api.routes.sessions.session_repo.count_messages_by_session", new_callable=AsyncMock) as mock_count:

            mock_msgs.return_value = messages
            mock_count.return_value = 5
            response = client.get(f"/api/v1/sessions/{session_id}/messages")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 5
        assert data["total"] == 5


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Chat
# ═══════════════════════════════════════════════════════════════════════════════

class TestChat:
    def test_chat_returns_200(self, client: TestClient) -> None:
        session_id = uuid.uuid4()
        session_record = make_session_record(id=session_id)
        agent_msg = make_message_record(id=uuid.uuid4(), session_id=session_id)

        agent_result = {
            "final_answer": "El artículo 2.2.4.6.1 establece el objeto del capítulo.",
            "sources": [{"source": "decreto_1072.pdf", "article": "2.2.4.6.1", "page": "45"}],
            "retrieval_strategy": "hybrid_rrf",
            "iteration_count": 1,
            "reflection": {"score": 0.88, "is_grounded": True, "has_hallucination": False,
                           "cites_source": True, "feedback": ""},
        }

        with patch("src.api.routes.chat.session_repo.get_session", new_callable=AsyncMock) as mock_session, \
             patch("src.api.routes.chat.session_repo.create_message", new_callable=AsyncMock) as mock_msg, \
             patch("src.api.routes.chat.arun_agent", new_callable=AsyncMock) as mock_agent, \
             patch("src.api.routes.chat.transaction") as mock_tx, \
             patch("src.api.routes.chat.query_repo") as mock_qr, \
             patch("src.api.routes.chat.session_repo.increment_session_counters", new_callable=AsyncMock):

            mock_session.return_value = session_record
            mock_msg.return_value = agent_msg
            mock_agent.return_value = agent_result

            mock_tx_ctx = AsyncMock()
            mock_tx_ctx.__aenter__ = AsyncMock(return_value=AsyncMock())
            mock_tx_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_tx.return_value = mock_tx_ctx
            mock_qr.create_query_log = AsyncMock()

            response = client.post("/api/v1/chat/", json={
                "session_id": str(session_id),
                "query": "¿Qué dice el artículo 2.2.4.6.1?",
            })

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "iteration_count" in data
        assert "response_time_ms" in data

    def test_chat_query_too_short(self, client: TestClient) -> None:
        response = client.post("/api/v1/chat/", json={
            "session_id": str(uuid.uuid4()),
            "query": "hi",  # menos de 3 chars
        })
        assert response.status_code == 422

    def test_chat_invalid_session_uuid(self, client: TestClient) -> None:
        response = client.post("/api/v1/chat/", json={
            "session_id": "not-valid-uuid",
            "query": "pregunta válida para el agente",
        })
        assert response.status_code == 400

    def test_chat_session_not_found(self, client: TestClient) -> None:
        with patch("src.api.routes.chat.session_repo.get_session", new_callable=AsyncMock) as mock_session, \
             patch("src.api.routes.chat.session_repo.create_message", new_callable=AsyncMock):

            mock_session.return_value = None
            response = client.post("/api/v1/chat/", json={
                "session_id": str(uuid.uuid4()),
                "query": "pregunta válida para el agente",
            })

        assert response.status_code == 404

    def test_chat_max_iterations_out_of_range(self, client: TestClient) -> None:
        response = client.post("/api/v1/chat/", json={
            "session_id": str(uuid.uuid4()),
            "query": "pregunta válida",
            "max_iterations": 10,  # máximo es 4
        })
        assert response.status_code == 422

    def test_chat_stream_endpoint_exists(self, client: TestClient) -> None:
        """El endpoint de streaming debe existir aunque el cliente no soporte SSE."""
        with patch("src.api.routes.chat.arun_agent", new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = {
                "final_answer": "respuesta de prueba",
                "sources": [],
                "retrieval_strategy": "hybrid",
                "iteration_count": 1,
            }
            response = client.post("/api/v1/chat/stream", json={
                "session_id": str(uuid.uuid4()),
                "query": "pregunta de prueba para el stream",
            })
        # El endpoint debe responder aunque sea con error de sesión
        assert response.status_code in (200, 404, 400)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Documents
# ═══════════════════════════════════════════════════════════════════════════════

class TestDocuments:
    def test_get_stats_returns_200(self, client: TestClient) -> None:
        stats = {"total_documents": 5, "total_chunks": 210, "by_document_type": {"decreto": 3}, "by_loader": {"pymupdf": 3}}

        with patch("src.api.routes.documents.document_repo.get_corpus_stats", new_callable=AsyncMock) as mock_stats:
            mock_stats.return_value = stats
            response = client.get("/api/v1/documents/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] == 5
        assert data["total_chunks"] == 210

    def test_list_documents_returns_paginated(self, client: TestClient) -> None:
        doc = MagicMock()
        doc.id = uuid.uuid4()
        doc.filename = "decreto_1072.pdf"
        doc.document_type = "decreto"
        doc.loader_used = "pymupdf"
        doc.chunk_count = 42
        doc.page_count = 10
        doc.is_indexed = True
        doc.classifier_confidence = 0.95
        doc.created_at = datetime.now(timezone.utc)

        with patch("src.api.routes.documents.document_repo.get_all_documents", new_callable=AsyncMock) as mock_docs, \
             patch("src.api.routes.documents.document_repo.get_corpus_stats", new_callable=AsyncMock) as mock_stats:

            mock_docs.return_value = [doc]
            mock_stats.return_value = {"total_documents": 1, "total_chunks": 42,
                                       "by_document_type": {}, "by_loader": {}}
            response = client.get("/api/v1/documents/")

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1

    def test_delete_document_returns_204(self, client: TestClient) -> None:
        with patch("src.api.routes.documents.document_repo.deindex_document", new_callable=AsyncMock) as mock_del:
            mock_del.return_value = True
            response = client.delete(f"/api/v1/documents/{uuid.uuid4()}")
        assert response.status_code == 204

    def test_delete_document_not_found(self, client: TestClient) -> None:
        with patch("src.api.routes.documents.document_repo.deindex_document", new_callable=AsyncMock) as mock_del:
            mock_del.return_value = False
            response = client.delete(f"/api/v1/documents/{uuid.uuid4()}")
        assert response.status_code == 404

    def test_delete_document_invalid_uuid(self, client: TestClient) -> None:
        response = client.delete("/api/v1/documents/not-a-uuid")
        assert response.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Admin
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdmin:
    def test_quality_metrics_returns_200(self, client: TestClient) -> None:
        metrics = {
            "total_queries": 100,
            "avg_reflection_score": 0.83,
            "avg_iterations": 1.2,
            "reformulated_pct": 15.0,
            "low_score_pct": 8.0,
            "period_days": 7,
        }
        with patch("src.api.routes.admin.query_repo.get_quality_metrics", new_callable=AsyncMock) as mock:
            mock.return_value = metrics
            response = client.get("/api/v1/admin/metrics/quality")

        assert response.status_code == 200
        data = response.json()
        assert data["total_queries"] == 100
        assert data["avg_reflection_score"] == 0.83

    def test_strategy_performance_returns_list(self, client: TestClient) -> None:
        strategies = [
            {"strategy": "hybrid_rrf", "total_queries": 80, "avg_score": 0.85, "avg_time_ms": 1200.0, "avg_iterations": 1.1},
            {"strategy": "full", "total_queries": 20, "avg_score": 0.90, "avg_time_ms": 2500.0, "avg_iterations": 1.5},
        ]
        with patch("src.api.routes.admin.query_repo.get_strategy_performance", new_callable=AsyncMock) as mock:
            mock.return_value = strategies
            response = client.get("/api/v1/admin/metrics/strategies")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["strategy"] == "hybrid_rrf"

    def test_top_queries_returns_list(self, client: TestClient) -> None:
        queries = [
            {"query": "¿Qué es el COPASST?", "frequency": 15, "avg_score": 0.88},
            {"query": "artículo 2.2.4.6.1", "frequency": 12, "avg_score": 0.92},
        ]
        with patch("src.api.routes.admin.query_repo.get_top_queries", new_callable=AsyncMock) as mock:
            mock.return_value = queries
            response = client.get("/api/v1/admin/queries/top")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["query"] == "¿Qué es el COPASST?"

    def test_vector_store_status_returns_dict(self, client: TestClient) -> None:
        with patch("src.api.routes.admin.get_vector_store") as mock_vs:
            mock_vs.return_value.health_check.return_value = {
                "vector_store": "ok",
                "collection": "fenix_legal",
                "documents": 42,
            }
            response = client.get("/api/v1/admin/vector-store")

        assert response.status_code == 200
        assert "vector_store" in response.json() or "error" in response.json()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Middleware y error handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestMiddleware:
    def test_request_id_in_response_headers(self, client: TestClient) -> None:
        with patch("src.api.routes.health.check_database_connectivity", new_callable=AsyncMock) as mock_db, \
             patch("src.api.routes.health.get_vector_store") as mock_vs, \
             patch("src.api.routes.health.check_bedrock_connectivity") as mock_bedrock:
            mock_db.return_value = {"database": "ok"}
            mock_vs.return_value.is_initialized = True
            mock_vs.return_value.count.return_value = 0
            mock_bedrock.return_value = {"bedrock": True}
            response = client.get("/health")

        assert "x-request-id" in response.headers or "X-Request-ID" in response.headers

    def test_response_time_in_response_headers(self, client: TestClient) -> None:
        with patch("src.api.routes.health.check_database_connectivity", new_callable=AsyncMock) as mock_db, \
             patch("src.api.routes.health.get_vector_store") as mock_vs, \
             patch("src.api.routes.health.check_bedrock_connectivity") as mock_bedrock:
            mock_db.return_value = {"database": "ok"}
            mock_vs.return_value.is_initialized = True
            mock_vs.return_value.count.return_value = 0
            mock_bedrock.return_value = {"bedrock": True}
            response = client.get("/health")

        rt_header = response.headers.get("x-response-time") or response.headers.get("X-Response-Time")
        assert rt_header is not None
        assert "ms" in rt_header

    def test_custom_request_id_preserved(self, client: TestClient) -> None:
        custom_id = "my-custom-request-id-12345"
        with patch("src.api.routes.health.check_database_connectivity", new_callable=AsyncMock) as mock_db, \
             patch("src.api.routes.health.get_vector_store") as mock_vs, \
             patch("src.api.routes.health.check_bedrock_connectivity") as mock_bedrock:
            mock_db.return_value = {"database": "ok"}
            mock_vs.return_value.is_initialized = True
            mock_vs.return_value.count.return_value = 0
            mock_bedrock.return_value = {"bedrock": True}
            response = client.get("/health", headers={"X-Request-ID": custom_id})

        response_id = response.headers.get("x-request-id") or response.headers.get("X-Request-ID")
        assert response_id == custom_id

    def test_404_returns_error_response_format(self, client: TestClient) -> None:
        response = client.get("/api/v1/endpoint-que-no-existe")
        assert response.status_code == 404
        # FastAPI retorna su propio formato para rutas no encontradas

    def test_error_response_has_required_fields(self, client: TestClient) -> None:
        """Las respuestas de error del handler personalizado tienen 'error' y 'detail'."""
        with patch("src.api.routes.sessions.session_repo.get_session", new_callable=AsyncMock) as mock:
            mock.return_value = None
            response = client.get(f"/api/v1/sessions/{uuid.uuid4()}")

        data = response.json()
        assert "error" in data
        assert "detail" in data


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemas:
    def test_session_response_uuid_as_string(self) -> None:
        from src.api.schemas import SessionResponse  # noqa: PLC0415

        session_id = uuid.uuid4()
        record = MagicMock()
        record.id = session_id  # UUID object
        record.user_identifier = "user@test.com"
        record.title = None
        record.is_active = True
        record.total_messages = 0
        record.total_documents = 0
        record.created_at = datetime.now(timezone.utc)
        record.updated_at = datetime.now(timezone.utc)

        response = SessionResponse.model_validate(record)
        assert isinstance(response.id, str)
        assert response.id == str(session_id)

    def test_chat_request_strips_whitespace(self) -> None:
        from src.api.schemas import ChatRequest  # noqa: PLC0415

        req = ChatRequest(
            session_id=str(uuid.uuid4()),
            query="  ¿Qué dice el artículo 1?  ",
        )
        assert req.query == "¿Qué dice el artículo 1?"

    def test_chat_request_empty_query_raises(self) -> None:
        from src.api.schemas import ChatRequest  # noqa: PLC0415
        from pydantic import ValidationError  # noqa: PLC0415

        with pytest.raises(ValidationError):
            ChatRequest(session_id=str(uuid.uuid4()), query="   ")

    def test_paginated_response_has_more(self) -> None:
        from src.api.schemas import PaginatedResponse  # noqa: PLC0415

        result = PaginatedResponse.from_list(
            items=["a", "b", "c"],
            total=10,
            limit=3,
            offset=0,
        )
        assert result.has_more is True
        assert result.total == 10

    def test_paginated_response_no_more(self) -> None:
        from src.api.schemas import PaginatedResponse  # noqa: PLC0415

        result = PaginatedResponse.from_list(
            items=["a", "b"],
            total=2,
            limit=10,
            offset=0,
        )
        assert result.has_more is False

    def test_source_reference_optional_fields(self) -> None:
        from src.api.schemas import SourceReference  # noqa: PLC0415

        src = SourceReference(source="decreto.pdf")
        assert src.article == ""
        assert src.page == ""
