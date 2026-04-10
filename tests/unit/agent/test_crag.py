"""Tests del CRAG skill — Corrective RAG grading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_core.documents import Document

from src.agent.skills.crag import (
    DocumentGrade,
    grade_documents,
    rewrite_query_for_reretrieval,
    grade_documents_node,
    route_after_grading,
)


# ─── grade_documents ────────────────────────────────────────────────────────

class TestGradeDocuments:
    """Tests de la evaluacion de calidad de documentos."""

    def test_empty_documents_returns_incorrect(self):
        grade = grade_documents("test query", [])
        assert grade.quality == "incorrect"
        assert grade.score == 0.0

    @patch("src.agent.skills.crag.get_llm")
    def test_grading_calls_llm(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = DocumentGrade(
            quality="correct",
            score=0.85,
            reasoning="Relevant documents",
        )
        mock_llm.with_structured_output.return_value = mock_chain
        mock_get_llm.return_value = mock_llm

        docs = [
            Document(
                page_content="relevant content about the topic",
                metadata={"source": "doc.pdf", "chunk_index": 0},
            ),
        ]

        grade = grade_documents("test query", docs)

        mock_get_llm.assert_called_once()
        mock_chain.invoke.assert_called_once()
        assert grade.quality == "correct"

    @patch("src.agent.skills.crag.get_llm")
    def test_grading_fallback_on_llm_error(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM error")
        mock_llm.with_structured_output.return_value = mock_chain
        mock_get_llm.return_value = mock_llm

        docs = [
            Document(
                page_content="some content",
                metadata={"source": "doc.pdf"},
            ),
        ]

        grade = grade_documents("test query", docs)

        # Fallback debe ser ambiguous
        assert grade.quality == "ambiguous"
        assert grade.score == 0.5


# ─── rewrite_query_for_reretrieval ─────────────────────────────────────────

class TestRewriteQuery:
    """Tests de reescritura de queries para re-retrieval."""

    @patch("src.agent.skills.crag.get_llm")
    def test_rewrite_for_ambiguous(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Rewritten query with more detail")
        mock_get_llm.return_value = mock_llm

        grade = DocumentGrade(
            quality="ambiguous",
            score=0.5,
            reasoning="Partially relevant",
        )

        result = rewrite_query_for_reretrieval("original query", grade)
        assert "Rewritten query" in result

    @patch("src.agent.skills.crag.get_llm")
    def test_rewrite_for_incorrect_uses_stepback(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Broader general query")
        mock_get_llm.return_value = mock_llm

        grade = DocumentGrade(
            quality="incorrect",
            score=0.1,
            reasoning="Not relevant",
        )

        result = rewrite_query_for_reretrieval("original query", grade)
        assert "Broader" in result

    @patch("src.agent.skills.crag.get_llm")
    def test_rewrite_fallback_on_error(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        mock_get_llm.return_value = mock_llm

        grade = DocumentGrade(quality="ambiguous", score=0.5, reasoning="test")

        result = rewrite_query_for_reretrieval("original query", grade)
        assert result == "original query"


# ─── grade_documents_node ──────────────────────────────────────────────────

class TestGradeDocumentsNode:
    """Tests del nodo de grading para LangGraph."""

    @patch("src.agent.skills.crag.grade_documents")
    def test_node_returns_correct_route(self, mock_grade):
        mock_grade.return_value = DocumentGrade(
            quality="correct",
            score=0.85,
            reasoning="Good docs",
        )

        state = {
            "user_query": "test",
            "retrieval_results": [Document(page_content="content")],
        }

        result = grade_documents_node(state)

        assert result["route"] == "generation"
        assert result["doc_quality"] == "correct"
        assert result["grade_score"] == 0.85

    @patch("src.agent.skills.crag.grade_documents")
    def test_node_rewrites_query_on_ambiguous(self, mock_grade):
        mock_grade.return_value = DocumentGrade(
            quality="ambiguous",
            score=0.5,
            reasoning="Partially relevant",
        )

        with patch("src.agent.skills.crag.rewrite_query_for_reretrieval") as mock_rewrite:
            mock_rewrite.return_value = "rewritten query"

            state = {
                "user_query": "test",
                "retrieval_results": [Document(page_content="content")],
            }

            result = grade_documents_node(state)

            assert result["route"] == "retrieval"
            assert result["active_query"] == "rewritten query"

    def test_node_handles_empty_docs(self):
        state = {
            "user_query": "test",
            "retrieval_results": [],
        }

        result = grade_documents_node(state)

        assert result["doc_quality"] == "incorrect"
        assert result["route"] == "retrieval"


# ─── route_after_grading ──────────────────────────────────────────────────

class TestRouteAfterGrading:
    """Tests del routing condicional post-grading."""

    def test_correct_routes_to_generation(self):
        state = {"route": "generation"}
        assert route_after_grading(state) == "generation"

    def test_ambiguous_routes_to_retrieval(self):
        state = {"route": "retrieval"}
        assert route_after_grading(state) == "retrieval"

    def test_default_routes_to_generation(self):
        state = {}
        assert route_after_grading(state) == "generation"
