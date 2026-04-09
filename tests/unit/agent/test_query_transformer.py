"""
Tests del QueryTransformer — rewriting, step-back, decompose.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.skills.query_transformer import QueryTransformer


class TestQueryTransformer:
    """Tests de transformación de queries."""

    def test_rewrite_returns_string(self) -> None:
        """Rewriting retorna un string (query reformulada)."""
        transformer = QueryTransformer()
        result = transformer.rewrite("¿qué dice sobre eso?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_step_back_returns_string(self) -> None:
        """Step-back retorna un string (query más general)."""
        transformer = QueryTransformer()
        result = transformer.step_back("¿qué dice la sección 3.2?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_decompose_returns_list(self) -> None:
        """Decomposition retorna una lista de sub-queries."""
        transformer = QueryTransformer()
        result = transformer.decompose("¿Cuáles son las diferencias entre X e Y y cómo afectan Z?")
        assert isinstance(result, list)
        # Si la descomposición falla, retorna la query original
        assert len(result) >= 1

    def test_transform_all_returns_variants(self) -> None:
        """transform_all retorna al menos la query original."""
        transformer = QueryTransformer()
        variants = transformer.transform_all("¿qué dice el artículo 5?")
        assert isinstance(variants, list)
        assert len(variants) >= 1  # Siempre incluye la original

    def test_rewrite_fallback_on_error(self) -> None:
        """Si rewriting falla, retorna la query original."""
        transformer = QueryTransformer()
        with patch.object(transformer, "_rewrite_chain") as mock_chain:
            mock_chain.invoke.side_effect = Exception("LLM error")
            result = transformer.rewrite("test query")
            assert result == "test query"

    def test_step_back_fallback_on_error(self) -> None:
        """Si step-back falla, retorna la query original."""
        transformer = QueryTransformer()
        with patch.object(transformer, "_step_back_chain") as mock_chain:
            mock_chain.invoke.side_effect = Exception("LLM error")
            result = transformer.step_back("test query")
            assert result == "test query"

    def test_decompose_fallback_on_error(self) -> None:
        """Si decomposition falla, retorna [query]."""
        transformer = QueryTransformer()
        with patch.object(transformer, "_decompose_chain") as mock_chain:
            mock_chain.invoke.side_effect = Exception("LLM error")
            result = transformer.decompose("test query")
            assert result == ["test query"]

    def test_transform_all_deduplicates(self) -> None:
        """transform_all no retorna variantes duplicadas."""
        transformer = QueryTransformer()
        with patch.object(transformer, "rewrite", return_value="different query"):
            with patch.object(transformer, "step_back", return_value="another query"):
                variants = transformer.transform_all("original query")
                assert len(variants) == len(set(variants))  # No duplicados
