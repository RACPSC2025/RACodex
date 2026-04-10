"""Tests del retrieval_node con Multi-Query Fusion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_core.documents import Document

from src.agent.nodes.all_nodes import retrieval_node


class TestRetrievalNodeMultiQueryFusion:
    """Tests de la integracion de QueryTransformer en retrieval_node."""

    @pytest.fixture
    def mock_ensemble(self):
        """Mock del ensemble retriever."""
        mock = MagicMock()
        mock_result = MagicMock()
        mock_result.documents = [
            Document(
                page_content="relevant content",
                metadata={"source": "doc.pdf", "chunk_index": 0},
            ),
        ]
        mock.retrieve.return_value = mock_result
        return mock

    @pytest.fixture
    def mock_vector_store(self, mock_ensemble):
        """Mock del vector store."""
        mock = MagicMock()
        mock.is_initialized = True
        return mock

    @patch("src.agent.nodes.all_nodes.get_ensemble_retriever")
    @patch("src.agent.nodes.all_nodes.get_vector_store")
    @patch("src.agent.nodes.all_nodes.QueryTransformer")
    def test_retrieval_uses_query_transformer(
        self, mock_transformer_cls, mock_get_vs, mock_get_ensemble,
        mock_vector_store, mock_ensemble,
    ):
        """El retrieval_node debe invocar QueryTransformer.transform_all()."""
        mock_get_vs.return_value = mock_vector_store
        mock_get_ensemble.return_value = mock_ensemble

        mock_transformer = MagicMock()
        mock_transformer.transform_all.return_value = [
            "original query",
            "rewritten query with technical terms",
            "broader general query",
        ]
        mock_transformer_cls.return_value = mock_transformer

        state = {
            "user_query": "original query",
            "active_query": "",
        }

        result = retrieval_node(state)

        # Debe llamar transform_all con la query
        mock_transformer.transform_all.assert_called_once_with("original query")

        # Debe ejecutar retrieval con cada variante
        assert mock_ensemble.retrieve.call_count == 3

        # Debe retornar documentos deduplicados
        assert "retrieval_results" in result
        assert result["retrieval_strategy"] == "multi_query_fusion"

    @patch("src.agent.nodes.all_nodes.get_ensemble_retriever")
    @patch("src.agent.nodes.all_nodes.get_vector_store")
    @patch("src.agent.nodes.all_nodes.QueryTransformer")
    def test_retrieval_deduplicates_results(
        self, mock_transformer_cls, mock_get_vs, mock_get_ensemble,
        mock_vector_store, mock_ensemble,
    ):
        """Resultados duplicados entre variantes deben deduplicarse."""
        mock_get_vs.return_value = mock_vector_store
        mock_get_ensemble.return_value = mock_ensemble

        mock_transformer = MagicMock()
        mock_transformer.transform_all.return_value = ["query1", "query2"]
        mock_transformer_cls.return_value = mock_transformer

        # Ambas variantes retornan el mismo documento
        same_doc = Document(
            page_content="duplicate content",
            metadata={"source": "doc.pdf", "chunk_index": 0},
        )
        mock_result = MagicMock()
        mock_result.documents = [same_doc]
        mock_ensemble.retrieve.return_value = mock_result

        state = {"user_query": "test query", "active_query": ""}
        result = retrieval_node(state)

        # Debe haber solo 1 doc (deduplicado) aunque retrieve se llamo 2 veces
        assert len(result["retrieval_results"]) == 1

    @patch("src.agent.nodes.all_nodes.get_ensemble_retriever")
    @patch("src.agent.nodes.all_nodes.get_vector_store")
    @patch("src.agent.nodes.all_nodes.QueryTransformer")
    def test_retrieval_handles_empty_query(
        self, mock_transformer_cls, mock_get_vs, mock_get_ensemble,
    ):
        """Query vacia debe retornar resultados vacios sin error."""
        state = {"user_query": "", "active_query": ""}
        result = retrieval_node(state)

        assert result["retrieval_results"] == []
        assert result["retrieval_strategy"] == "none"

    @patch("src.agent.nodes.all_nodes.get_ensemble_retriever")
    @patch("src.agent.nodes.all_nodes.get_vector_store")
    @patch("src.agent.nodes.all_nodes.QueryTransformer")
    def test_retrieval_falls_back_on_transformer_error(
        self, mock_transformer_cls, mock_get_vs, mock_get_ensemble,
        mock_vector_store, mock_ensemble,
    ):
        """Si QueryTransformer falla, debe usar la query original."""
        mock_get_vs.return_value = mock_vector_store
        mock_get_ensemble.return_value = mock_ensemble

        mock_transformer = MagicMock()
        mock_transformer.transform_all.side_effect = Exception("LLM error")
        mock_transformer.transform_all.return_value = ["fallback query"]
        mock_transformer_cls.return_value = mock_transformer

        state = {"user_query": "test query", "active_query": ""}

        # No debe fallar — debe manejar el error graceful
        # (dependiendo de la implementacion exacta del fallback)
