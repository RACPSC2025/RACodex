"""
Tests unitarios para el módulo agent de Fénix RAG.

Qué se testea:
  - AgentState: construcción, initial_state
  - DocumentClassifierSkill: clasificación por reglas, fallback
  - QueryPlannerSkill: detección de complejidad, extracción de artículos
  - AnswerValidatorSkill: detección de conclusiones genéricas, numerales
  - Nodos del grafo: document_router, generation (mockeados)
  - Graph routing: edges condicionales
  - run_agent: integración con mocks
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.state import AgentState, IngestionPlan, initial_state


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_doc(content: str = "texto legal", source: str = "decreto.pdf") -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "article_number": "1", "page": "1", "chunk_index": "0"},
    )


def make_mime_result(
    mime_type: str = "application/pdf",
    is_supported: bool = True,
) -> MagicMock:
    m = MagicMock()
    m.mime_type = mime_type
    m.is_supported = is_supported
    m.is_pdf = mime_type == "application/pdf"
    m.is_word = "wordprocessingml" in mime_type
    m.is_excel = "spreadsheetml" in mime_type or "ms-excel" in mime_type
    m.is_image = mime_type.startswith("image/")
    m.label = mime_type
    return m


def make_quality_result(is_native: bool = True, pages: int = 10, confidence: float = 0.9) -> MagicMock:
    q = MagicMock()
    q.is_native = is_native
    q.is_scanned = not is_native
    q.confidence = confidence
    q.total_pages = pages
    q.quality_label = "native_high_confidence" if is_native else "scanned_high_confidence"
    return q


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: AgentState
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentState:
    def test_initial_state_has_required_fields(self) -> None:
        state = initial_state("¿Qué dice el artículo 1?")
        assert "messages" in state
        assert "user_query" in state
        assert "active_query" in state
        assert state["user_query"] == "¿Qué dice el artículo 1?"

    def test_initial_state_messages_has_human_message(self) -> None:
        state = initial_state("query de prueba")
        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], HumanMessage)
        assert state["messages"][0].content == "query de prueba"

    def test_initial_state_empty_lists(self) -> None:
        state = initial_state("query")
        assert state["uploaded_files"] == []
        assert state["ingestion_plans"] == []
        assert state["ingested_documents"] == []
        assert state["retrieval_results"] == []

    def test_initial_state_counters(self) -> None:
        state = initial_state("query", max_iterations=3)
        assert state["iteration_count"] == 0
        assert state["max_iterations"] == 3

    def test_initial_state_with_files(self) -> None:
        files = ["/tmp/decreto.pdf", "/tmp/contrato.docx"]
        state = initial_state("query", uploaded_files=files)
        assert len(state["uploaded_files"]) == 2

    def test_initial_state_session_id(self) -> None:
        state = initial_state("query", session_id="session-123")
        assert state["session_id"] == "session-123"

    def test_initial_state_no_error(self) -> None:
        state = initial_state("query")
        assert state["error"] is None

    def test_ingestion_plan_typed_dict(self) -> None:
        plan = IngestionPlan(
            loader_type="pymupdf",
            cleaner_profile="legal_colombia",
            requires_ocr=False,
            document_type="decreto",
            source_path="/tmp/decreto.pdf",
            mime_type="application/pdf",
            confidence=0.95,
            reasoning="PDF nativo detectado",
        )
        assert plan["loader_type"] == "pymupdf"
        assert plan["confidence"] == 0.95


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: DocumentClassifierSkill
# ═══════════════════════════════════════════════════════════════════════════════

class TestDocumentClassifierSkill:
    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        from src.agent.skills.document_classifier import DocumentClassifierSkill
        classifier = DocumentClassifierSkill()
        with pytest.raises(FileNotFoundError):
            classifier.classify(tmp_path / "no_existe.pdf")

    def test_classifies_excel_correctly(self, tmp_path: Path) -> None:
        from src.agent.skills.document_classifier import DocumentClassifierSkill
        f = tmp_path / "data.xlsx"
        f.write_bytes(b"PK\x03\x04")

        classifier = DocumentClassifierSkill(use_llm_fallback=False)
        mime = make_mime_result("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        with patch("src.agent.skills.document_classifier.detect_mime", return_value=mime), \
             patch("src.agent.skills.document_classifier.analyze_pdf_quality", return_value=None):
            plan = classifier.classify(f)

        assert plan["loader_type"] == "excel"
        assert plan["document_type"] == "excel"
        assert plan["requires_ocr"] is False

    def test_classifies_word_correctly(self, tmp_path: Path) -> None:
        from src.agent.skills.document_classifier import DocumentClassifierSkill
        f = tmp_path / "contrato.docx"
        f.write_bytes(b"PK\x03\x04")

        classifier = DocumentClassifierSkill(use_llm_fallback=False)
        mime = make_mime_result(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        with patch("src.agent.skills.document_classifier.detect_mime", return_value=mime), \
             patch("src.agent.skills.document_classifier.analyze_pdf_quality", return_value=None):
            plan = classifier.classify(f)

        assert plan["loader_type"] == "word"

    def test_classifies_native_pdf_as_pymupdf(self, tmp_path: Path) -> None:
        from src.agent.skills.document_classifier import DocumentClassifierSkill
        f = tmp_path / "decreto.pdf"
        f.write_bytes(b"%PDF-1.4")

        classifier = DocumentClassifierSkill(use_llm_fallback=False)
        mime = make_mime_result("application/pdf")
        quality = make_quality_result(is_native=True, pages=10, confidence=0.92)

        with patch("src.agent.skills.document_classifier.detect_mime", return_value=mime), \
             patch("src.agent.skills.document_classifier.analyze_pdf_quality", return_value=quality):
            plan = classifier.classify(f)

        assert plan["loader_type"] == "pymupdf"
        assert plan["requires_ocr"] is False

    def test_classifies_scanned_pdf_as_ocr(self, tmp_path: Path) -> None:
        from src.agent.skills.document_classifier import DocumentClassifierSkill
        f = tmp_path / "escaneado.pdf"
        f.write_bytes(b"%PDF-1.4")

        classifier = DocumentClassifierSkill(use_llm_fallback=False)
        mime = make_mime_result("application/pdf")
        quality = make_quality_result(is_native=False, pages=5, confidence=0.88)

        with patch("src.agent.skills.document_classifier.detect_mime", return_value=mime), \
             patch("src.agent.skills.document_classifier.analyze_pdf_quality", return_value=quality):
            plan = classifier.classify(f)

        assert plan["loader_type"] == "ocr"
        assert plan["requires_ocr"] is True

    def test_classifies_image_as_ocr(self, tmp_path: Path) -> None:
        from src.agent.skills.document_classifier import DocumentClassifierSkill
        f = tmp_path / "foto.jpg"
        f.write_bytes(b"\xff\xd8\xff")

        classifier = DocumentClassifierSkill(use_llm_fallback=False)
        mime = make_mime_result("image/jpeg")

        with patch("src.agent.skills.document_classifier.detect_mime", return_value=mime), \
             patch("src.agent.skills.document_classifier.analyze_pdf_quality", return_value=None):
            plan = classifier.classify(f)

        assert plan["loader_type"] == "ocr"
        assert plan["requires_ocr"] is True

    def test_large_pdf_uses_docling(self, tmp_path: Path) -> None:
        from src.agent.skills.document_classifier import DocumentClassifierSkill
        f = tmp_path / "decreto_largo.pdf"
        f.write_bytes(b"%PDF-1.4")

        classifier = DocumentClassifierSkill(use_llm_fallback=False)
        mime = make_mime_result("application/pdf")
        quality = make_quality_result(is_native=True, pages=100, confidence=0.90)

        with patch("src.agent.skills.document_classifier.detect_mime", return_value=mime), \
             patch("src.agent.skills.document_classifier.analyze_pdf_quality", return_value=quality):
            plan = classifier.classify(f)

        assert plan["loader_type"] == "docling"

    def test_decreto_1072_uses_specific_profile(self, tmp_path: Path) -> None:
        from src.agent.skills.document_classifier import DocumentClassifierSkill
        f = tmp_path / "decreto_1072_2015.pdf"
        f.write_bytes(b"%PDF-1.4")

        classifier = DocumentClassifierSkill(use_llm_fallback=False)
        mime = make_mime_result("application/pdf")
        quality = make_quality_result(is_native=True, pages=20, confidence=0.95)

        with patch("src.agent.skills.document_classifier.detect_mime", return_value=mime), \
             patch("src.agent.skills.document_classifier.analyze_pdf_quality", return_value=quality):
            plan = classifier.classify(f)

        assert plan["cleaner_profile"] == "decreto_1072"

    def test_fallback_plan_on_mime_error(self, tmp_path: Path) -> None:
        from src.agent.skills.document_classifier import DocumentClassifierSkill
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4")

        classifier = DocumentClassifierSkill(use_llm_fallback=False)

        with patch("src.agent.skills.document_classifier.detect_mime",
                   side_effect=Exception("error MIME")):
            plan = classifier.classify(f)

        assert plan["confidence"] == 0.0
        assert "Error" in plan["reasoning"]

    def test_classify_many_handles_errors(self, tmp_path: Path) -> None:
        from src.agent.skills.document_classifier import DocumentClassifierSkill
        files = [tmp_path / "existe.pdf", tmp_path / "no_existe.pdf"]
        (tmp_path / "existe.pdf").write_bytes(b"%PDF-1.4")

        classifier = DocumentClassifierSkill(use_llm_fallback=False)
        mime = make_mime_result("application/pdf")
        quality = make_quality_result()

        with patch("src.agent.skills.document_classifier.detect_mime", return_value=mime), \
             patch("src.agent.skills.document_classifier.analyze_pdf_quality", return_value=quality):
            plans = classifier.classify_many(files)

        assert len(plans) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: QueryPlannerSkill
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueryPlannerSkill:
    def test_simple_query_no_llm(self) -> None:
        from src.agent.skills.query_planner import QueryPlannerSkill
        planner = QueryPlannerSkill()
        plan = planner.plan("¿Qué es el COPASST?")
        assert plan.complexity == "simple"
        assert plan.sub_queries == ["¿Qué es el COPASST?"]
        assert plan.use_planner is False

    def test_article_number_query_simple(self) -> None:
        from src.agent.skills.query_planner import QueryPlannerSkill
        planner = QueryPlannerSkill()
        plan = planner.plan("artículo 2.2.4.6.1")
        assert plan.complexity == "simple"
        assert "2.2.4.6.1" in plan.article_numbers

    def test_comparison_query_is_compound(self) -> None:
        from src.agent.skills.query_planner import QueryPlannerSkill
        planner = QueryPlannerSkill()
        query = "¿Cuál es la diferencia entre el artículo 22 y el artículo 23?"
        plan = planner._assess_complexity(query)
        assert plan == "compound"

    def test_long_query_is_complex(self) -> None:
        from src.agent.skills.query_planner import QueryPlannerSkill
        planner = QueryPlannerSkill()
        query = " ".join(["palabra"] * 35)  # 35 palabras
        complexity = planner._assess_complexity(query)
        assert complexity == "complex"

    def test_extract_articles_from_query(self) -> None:
        from src.agent.skills.query_planner import QueryPlannerSkill
        planner = QueryPlannerSkill()
        articles = planner._extract_articles("artículo 2.2.4.6.1 y artículo 2.2.4.6.15")
        assert "2.2.4.6.1" in articles
        assert "2.2.4.6.15" in articles

    def test_extract_articles_deduplicates(self) -> None:
        from src.agent.skills.query_planner import QueryPlannerSkill
        planner = QueryPlannerSkill()
        articles = planner._extract_articles("artículo 1 y artículo 1 nuevamente")
        assert articles.count("1") == 1

    def test_llm_fallback_on_complex_query(self) -> None:
        """En queries complejas, el LLM se invoca (mockeado)."""
        from src.agent.skills.query_planner import QueryPlannerSkill
        import json

        planner = QueryPlannerSkill()
        mock_llm_response = json.dumps({
            "sub_queries": ["sub-query 1", "sub-query 2"],
            "strategy": "hybrid",
            "expected_sources": ["decreto"],
            "complexity": "compound",
        })

        with patch.object(planner, "_plan_with_llm") as mock_plan:
            mock_plan.return_value = MagicMock(
                original_query="q",
                sub_queries=["sub-query 1", "sub-query 2"],
                strategy="hybrid",
                complexity="compound",
                article_numbers=[],
                use_planner=True,
                expected_sources=["decreto"],
            )
            query = "¿Diferencias entre artículo 22 y artículo 23 sobre capacitación?"
            plan = planner.plan(query)

        assert len(plan.sub_queries) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: AnswerValidatorSkill
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnswerValidatorSkill:
    def test_empty_answer_is_invalid(self) -> None:
        from src.agent.skills.answer_validator import AnswerValidatorSkill
        v = AnswerValidatorSkill(use_llm_validation=False)
        result = v.validate("", [])
        assert result.is_valid is False

    def test_generic_conclusion_detected(self) -> None:
        from src.agent.skills.answer_validator import AnswerValidatorSkill
        v = AnswerValidatorSkill(use_llm_validation=False)
        answer = "El artículo dice X. Estos principios buscan garantizar el bienestar."
        result = v.validate(answer, [make_doc("El artículo dice X.")])
        assert result.is_valid is False
        assert len(result.violations) > 0

    def test_clean_answer_is_valid(self) -> None:
        from src.agent.skills.answer_validator import AnswerValidatorSkill
        v = AnswerValidatorSkill(use_llm_validation=False)
        answer = "ARTÍCULO 1. El empleador debe implementar el SG-SST."
        result = v.validate(answer, [make_doc("ARTÍCULO 1. El empleador debe implementar el SG-SST.")])
        assert result.is_valid is True
        assert result.violations == []

    def test_excessive_numerals_detected(self) -> None:
        from src.agent.skills.answer_validator import AnswerValidatorSkill
        v = AnswerValidatorSkill(use_llm_validation=False)
        # Contexto tiene hasta numeral 2, respuesta hasta 8
        context_doc = make_doc("1. Primero\n2. Segundo")
        answer = "1. Primero\n2. Segundo\n3. Tercero\n4. Cuarto\n5. Quinto\n6. Sexto\n7. Séptimo\n8. Octavo"
        result = v.validate(answer, [context_doc])
        assert result.is_valid is False
        assert any("umeral" in v.lower() for v in result.violations)

    def test_not_found_response_is_valid(self) -> None:
        from src.agent.skills.answer_validator import AnswerValidatorSkill
        v = AnswerValidatorSkill(use_llm_validation=False)
        answer = "No encontré información relevante en los documentos para responder esta pregunta."
        result = v.validate(answer, [])
        assert result.is_valid is True

    def test_sanitized_answer_returned(self) -> None:
        from src.agent.skills.answer_validator import AnswerValidatorSkill
        v = AnswerValidatorSkill(use_llm_validation=False)
        answer = "respuesta correcta sin problemas"
        result = v.validate(answer, [make_doc(answer)])
        assert result.sanitized_answer == answer


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Nodos del grafo
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphNodes:
    def test_document_router_no_uploads(self) -> None:
        from src.agent.nodes.all_nodes import document_router_node
        state = initial_state("query")
        result = document_router_node(state)
        assert result["ingestion_plans"] == []
        assert result["route"] == "retrieval"

    def test_document_router_with_uploads(self, tmp_path: Path) -> None:
        from src.agent.nodes.all_nodes import document_router_node
        f = tmp_path / "decreto.pdf"
        f.write_bytes(b"%PDF-1.4")

        state = initial_state("query", uploaded_files=[str(f)])

        mime = make_mime_result("application/pdf")
        quality = make_quality_result()

        with patch("src.agent.skills.document_classifier.detect_mime", return_value=mime), \
             patch("src.agent.skills.document_classifier.analyze_pdf_quality", return_value=quality):
            result = document_router_node(state)

        assert len(result["ingestion_plans"]) == 1
        assert result["route"] == "ingestion"

    def test_generation_node_empty_docs(self) -> None:
        from src.agent.nodes.all_nodes import generation_node

        state = initial_state("¿Qué es el COPASST?")
        state["retrieval_results"] = []

        mock_llm = MagicMock()
        mock_llm.return_value = "No encontré información relevante."

        with patch("src.agent.nodes.all_nodes.get_llm") as mock_get_llm:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "No encontré información."
            mock_get_llm.return_value.__or__ = MagicMock(return_value=mock_chain)

            # Patchear la cadena completa
            with patch("src.agent.nodes.all_nodes.StrOutputParser"):
                with patch("src.agent.nodes.all_nodes.GENERATION_PROMPT") as mock_prompt:
                    mock_prompt.__or__ = MagicMock(return_value=MagicMock(
                        __or__=MagicMock(return_value=MagicMock(
                            invoke=MagicMock(return_value="No encontré información.")
                        ))
                    ))
                    result = generation_node(state)

        assert "draft_answer" in result
        assert "sources" in result

    def test_reflection_node_max_iterations(self) -> None:
        from src.agent.nodes.all_nodes import reflection_node

        state = initial_state("query")
        state["draft_answer"] = "respuesta de prueba"
        state["retrieval_results"] = [make_doc()]
        state["iteration_count"] = 2  # >= max_iterations
        state["max_iterations"] = 2

        result = reflection_node(state)
        assert result["route"] == "END"
        assert result["final_answer"] == "respuesta de prueba"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Graph routing
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphRouting:
    def test_route_after_router_with_uploads(self) -> None:
        from src.agent.graph import route_after_router
        state = initial_state("query")
        state["route"] = "ingestion"
        assert route_after_router(state) == "ingestion"

    def test_route_after_router_no_uploads(self) -> None:
        from src.agent.graph import route_after_router
        state = initial_state("query")
        state["route"] = "retrieval"
        assert route_after_router(state) == "retrieval"

    def test_route_after_reflection_end(self) -> None:
        from src.agent.graph import route_after_reflection
        from langgraph.graph import END
        state = initial_state("query")
        state["route"] = "END"
        assert route_after_reflection(state) == END

    def test_route_after_reflection_retry(self) -> None:
        from src.agent.graph import route_after_reflection
        state = initial_state("query")
        state["route"] = "retrieval"
        assert route_after_reflection(state) == "retrieval"

    def test_route_after_ingestion_success(self) -> None:
        from src.agent.graph import route_after_ingestion
        state = initial_state("query")
        state["ingested_documents"] = [make_doc()]
        state["error"] = None
        assert route_after_ingestion(state) == "retrieval"

    def test_route_after_ingestion_fatal_error(self) -> None:
        from src.agent.graph import route_after_ingestion
        from langgraph.graph import END
        state = initial_state("query")
        state["ingested_documents"] = []
        state["error"] = "error fatal"
        assert route_after_ingestion(state) == END

    def test_build_graph_compiles(self) -> None:
        """El grafo debe compilar sin errores."""
        from src.agent.graph import build_graph
        graph = build_graph(with_tools=False)
        assert graph is not None

    def test_build_graph_with_tools_compiles(self) -> None:
        from src.agent.graph import build_graph
        graph = build_graph(with_tools=True)
        assert graph is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: run_agent (integración con mocks)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunAgent:
    def test_run_agent_returns_answer(self) -> None:
        from src.agent.graph import run_agent

        expected_state = {
            "final_answer": "El COPASST es el Comité Paritario de Seguridad.",
            "sources": [{"source": "decreto.pdf", "article": "1", "page": "1"}],
            "reflection": None,
            "iteration_count": 1,
            "retrieval_strategy": "hybrid",
            "ingestion_plans": [],
        }

        with patch("src.agent.graph.get_graph") as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = expected_state
            mock_get_graph.return_value = mock_graph

            result = run_agent("¿Qué es el COPASST?", session_id="test-001")

        assert result["final_answer"] == "El COPASST es el Comité Paritario de Seguridad."
        assert len(result["sources"]) == 1

    def test_run_agent_handles_exception(self) -> None:
        from src.agent.graph import run_agent

        with patch("src.agent.graph.get_graph") as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = Exception("Error inesperado")
            mock_get_graph.return_value = mock_graph

            result = run_agent("query")

        assert "error" in result
        assert result["final_answer"] != ""

    def test_run_agent_fallback_to_draft_answer(self) -> None:
        from src.agent.graph import run_agent

        state_no_final = {
            "draft_answer": "respuesta draft",
            "final_answer": "",
            "sources": [],
            "reflection": None,
            "iteration_count": 1,
            "retrieval_strategy": "",
            "ingestion_plans": [],
        }

        with patch("src.agent.graph.get_graph") as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = state_no_final
            mock_get_graph.return_value = mock_graph

            result = run_agent("query")

        assert result["final_answer"] == "respuesta draft"

    @pytest.mark.asyncio
    async def test_arun_agent_returns_answer(self) -> None:
        from src.agent.graph import arun_agent

        expected_state = {
            "final_answer": "Respuesta async.",
            "sources": [],
            "reflection": None,
            "iteration_count": 1,
        }

        with patch("src.agent.graph.get_graph") as mock_get_graph:
            mock_graph = MagicMock()
            mock_graph.ainvoke = AsyncMock(return_value=expected_state)
            mock_get_graph.return_value = mock_graph

            result = await arun_agent("¿Qué es SST?")

        assert result["final_answer"] == "Respuesta async."
