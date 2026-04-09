"""
Tests del SemanticRouter — clasificación de queries por categoría.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.skills.semantic_router import (
    SemanticRouter,
    get_semantic_router,
    clear_semantic_router_cache,
    RoutingResult,
)


# ─── Inicialización ───────────────────────────────────────────────────────────

class TestSemanticRouterInit:
    """Tests de inicialización del router."""

    def test_router_ready_after_init(self) -> None:
        clear_semantic_router_cache()
        router = get_semantic_router()
        assert router._ready is True

    def test_centroids_built(self) -> None:
        clear_semantic_router_cache()
        router = get_semantic_router()
        assert len(router._centroids) == 6  # 6 categorías

    def test_singleton_pattern(self) -> None:
        clear_semantic_router_cache()
        r1 = get_semantic_router()
        r2 = get_semantic_router()
        assert r1 is r2


# ─── Clasificación de Queries ────────────────────────────────────────────────

class TestQueryClassification:
    """Tests de clasificación de queries."""

    def test_classify_fact_query(self) -> None:
        router = get_semantic_router()
        result = router.classify("¿Qué dice el artículo 5?")
        assert isinstance(result, RoutingResult)
        assert result.category in router._centroids
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_obligation_query(self) -> None:
        router = get_semantic_router()
        result = router.classify("¿Cuáles son las obligaciones del empleador?")
        assert result.category in router._centroids

    def test_classify_consequence_query(self) -> None:
        router = get_semantic_router()
        result = router.classify("¿Qué pasa si no cumplo con la normativa?")
        assert result.category in router._centroids

    def test_all_scores_present(self) -> None:
        router = get_semantic_router()
        result = router.classify("¿Me puedes ayudar?")
        assert len(result.all_scores) == len(router._centroids)

    def test_empty_query(self) -> None:
        router = get_semantic_router()
        result = router.classify("")
        # No debe fallar — retorna algún resultado
        assert isinstance(result, RoutingResult)


# ─── Cosine Similarity ───────────────────────────────────────────────────────

class TestCosineSimilarity:
    """Tests del cálculo de cosine similarity."""

    def test_identical_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert SemanticRouter._cosine_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert SemanticRouter._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector(self) -> None:
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        assert SemanticRouter._cosine_similarity(a, b) == pytest.approx(0.0)
