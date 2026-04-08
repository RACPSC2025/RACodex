"""
Ensemble Retriever — combina todas las estrategias con pesos dinámicos.

El ensemble es el punto de entrada principal del módulo de retrieval.
El agente (Fase 4) llama a EnsembleRetriever.retrieve() y este decide
qué combinación de estrategias usar según el tipo de query.

Estrategias disponibles:
  "vector"       — Solo búsqueda vectorial (rápida, buena semántica)
  "bm25"         — Solo BM25 (exacta, buena para siglas y números)
  "hybrid"       — Vector + BM25 con RRF (mejor balance)
  "hierarchical" — Summary + Detail (mejor para preguntas complejas)
  "full"         — hybrid + hierarchical + reranking (máxima calidad)

Selección automática por tipo de query (heurística):
  - Query con números de artículo ("2.2.4.6.1") → hybrid (BM25 importante)
  - Query corta (< 5 palabras)                 → hybrid
  - Query larga y compleja                     → full (con jerarquía)
  - Default                                    → hybrid

El EnsembleRetriever puede configurarse para usar siempre una estrategia
específica (útil para evaluación y debugging).
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from langchain_core.documents import Document

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.retrieval.base import BaseRetriever, RetrievalQuery, RetrievalResult
from src.retrieval.reranker import Reranker, get_reranker
from src.retrieval.vector_store import VectorStore

log = get_logger(__name__)

# Regex para detectar números de artículo en la query
_ARTICLE_NUMBER_RE = re.compile(r"\d+\.\d+", re.IGNORECASE)


class RetrievalStrategy(StrEnum):
    VECTOR       = "vector"
    BM25         = "bm25"
    HYBRID       = "hybrid"
    HIERARCHICAL = "hierarchical"
    FULL         = "full"
    AUTO         = "auto"


class EnsembleRetriever(BaseRetriever):
    """
    Retriever orquestador que combina múltiples estrategias.

    Cada estrategia es un retriever independiente. El ensemble
    decide cuál usar (o combinarlos) según la query y la configuración.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        strategy: RetrievalStrategy | str = RetrievalStrategy.AUTO,
        use_reranking: bool = True,
        reranker: Reranker | None = None,
        top_k: int | None = None,
    ) -> None:
        """
        Args:
            vector_store: VectorStore inicializado (requerido).
            strategy: Estrategia de retrieval. AUTO selecciona automáticamente.
            use_reranking: Si True, aplica FlashRank al resultado final.
            reranker: Instancia de Reranker. None = usa singleton.
            top_k: Override del top_k de settings.
        """
        settings = get_settings()
        self._vector_store = vector_store
        self._strategy = RetrievalStrategy(strategy)
        self._use_reranking = use_reranking
        self._reranker = reranker or get_reranker()
        self._top_k = top_k or settings.retrieval_top_k
        self._rerank_top_k = settings.retrieval_rerank_top_k

        # Retrievers lazy-initialized para no crear conexiones innecesarias
        self._bm25: Any | None = None
        self._hybrid: Any | None = None
        self._hierarchical: Any | None = None

    @property
    def retriever_type(self) -> str:
        return f"ensemble_{self._strategy}"

    def is_ready(self) -> bool:
        return self._vector_store.is_initialized

    # ── Lazy retriever getters ────────────────────────────────────────────────

    def _get_bm25(self) -> Any:
        if self._bm25 is None:
            from src.retrieval.bm25_retriever import BM25Retriever  # noqa: PLC0415
            self._bm25 = BM25Retriever(vector_store=self._vector_store)
        return self._bm25

    def _get_hybrid(self) -> Any:
        if self._hybrid is None:
            from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: PLC0415
            self._hybrid = HybridRetriever(
                vector_store=self._vector_store,
                bm25_retriever=self._get_bm25(),
            )
        return self._hybrid

    def _get_hierarchical(self) -> Any:
        if self._hierarchical is None:
            from src.retrieval.hierarchical_retriever import HierarchicalRetriever  # noqa: PLC0415
            self._hierarchical = HierarchicalRetriever(
                detail_store=self._vector_store,
            )
        return self._hierarchical

    # ── Selección de estrategia ───────────────────────────────────────────────

    def _select_strategy(self, query: RetrievalQuery) -> RetrievalStrategy:
        """
        Selecciona la estrategia óptima para la query dada.

        Heurística basada en características de la query:
          - Números de artículo → hybrid (BM25 captura exactos)
          - Query corta (< 5 tokens) → hybrid
          - Query larga compleja → full
          - Default → hybrid
        """
        text = query.text

        # Detectar números de artículo (2.2.4.6.1, artículo 15, etc.)
        if _ARTICLE_NUMBER_RE.search(text):
            log.debug("strategy_selected_article_number", query=text[:60])
            return RetrievalStrategy.HYBRID

        word_count = len(text.split())

        if word_count < 5:
            return RetrievalStrategy.HYBRID

        if word_count >= 20:
            return RetrievalStrategy.FULL

        return RetrievalStrategy.HYBRID

    # ── Execution por estrategia ──────────────────────────────────────────────

    def _execute_strategy(
        self,
        strategy: RetrievalStrategy,
        query: RetrievalQuery,
    ) -> list[Document]:
        """Ejecuta la estrategia seleccionada y retorna documentos candidatos."""

        # Para las estrategias que lo necesiten, buscar más candidatos
        fetch_query = RetrievalQuery(
            text=query.text,
            top_k=self._top_k * 2 if self._use_reranking else self._top_k,
            filters=query.filters,
            rerank=False,
        )

        if strategy == RetrievalStrategy.VECTOR:
            docs = self._vector_store.similarity_search(fetch_query)

        elif strategy == RetrievalStrategy.BM25:
            result = self._get_bm25().retrieve(fetch_query)
            docs = result.documents

        elif strategy == RetrievalStrategy.HYBRID:
            result = self._get_hybrid().retrieve(fetch_query)
            docs = result.documents

        elif strategy == RetrievalStrategy.HIERARCHICAL:
            result = self._get_hierarchical().retrieve(fetch_query)
            docs = result.documents

        elif strategy == RetrievalStrategy.FULL:
            # Combinar hybrid + hierarchical y deduplicar
            hybrid_result = self._get_hybrid().retrieve(fetch_query)
            hier_result = self._get_hierarchical().retrieve(fetch_query)

            seen: set[str] = set()
            docs = []
            for doc in hybrid_result.documents + hier_result.documents:
                doc_id = self._doc_id(doc)
                if doc_id not in seen:
                    seen.add(doc_id)
                    docs.append(doc)

        else:
            docs = self._vector_store.similarity_search(fetch_query)

        return docs

    def _doc_id(self, doc: Document) -> str:
        source = doc.metadata.get("source", "")
        chunk_idx = doc.metadata.get("chunk_index", "")
        if source and chunk_idx != "":
            return f"{source}::{chunk_idx}"
        import hashlib  # noqa: PLC0415
        return hashlib.md5(doc.page_content[:200].encode()).hexdigest()

    # ── Retrieval principal ───────────────────────────────────────────────────

    def _retrieve(self, query: RetrievalQuery) -> list[Document]:
        """Orquesta la estrategia seleccionada y aplica reranking final."""

        # 1. Seleccionar estrategia
        if self._strategy == RetrievalStrategy.AUTO:
            active_strategy = self._select_strategy(query)
        else:
            active_strategy = self._strategy

        log.info(
            "ensemble_strategy",
            strategy=active_strategy,
            query=query.text[:80],
        )

        # 2. Ejecutar retrieval
        candidates = self._execute_strategy(active_strategy, query)

        if not candidates:
            log.warning("ensemble_no_candidates", strategy=active_strategy)
            return []

        log.debug("ensemble_candidates", count=len(candidates), strategy=active_strategy)

        # 3. Reranking final (si está habilitado)
        if self._use_reranking and query.rerank and len(candidates) > self._rerank_top_k:
            reranked = self._reranker.rerank(
                query=query.text,
                documents=candidates,
                top_k=self._rerank_top_k,
            )
            log.debug(
                "ensemble_reranked",
                before=len(candidates),
                after=len(reranked),
            )
            return reranked

        return candidates[:self._top_k]

    # ── Métricas de evaluación ────────────────────────────────────────────────

    def evaluate(
        self,
        eval_queries: list[dict],
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
    ) -> dict:
        """
        Evalúa el retriever con queries anotadas.

        Args:
            eval_queries: Lista de {"query": str, "relevant_chunks": [str]}
            strategy: Estrategia a evaluar.

        Returns:
            Dict con métricas: hit_rate, mrr, mean_docs.

        Para evaluación detallada, ver scripts/eval_retrieval.py
        """
        hits = 0
        reciprocal_ranks = []
        total_docs = 0

        for item in eval_queries:
            query = RetrievalQuery(text=item["query"], top_k=self._top_k)
            try:
                result = self.retrieve(query)
                docs = result.documents
                total_docs += len(docs)

                relevant = set(item.get("relevant_chunks", []))
                found = False
                for rank, doc in enumerate(docs, start=1):
                    chunk_id = doc.metadata.get("chunk_index", "")
                    if str(chunk_id) in relevant:
                        if not found:
                            hits += 1
                            reciprocal_ranks.append(1.0 / rank)
                            found = True

                if not found:
                    reciprocal_ranks.append(0.0)

            except Exception as exc:
                log.error("eval_query_failed", query=item["query"][:60], error=str(exc))
                reciprocal_ranks.append(0.0)

        n = len(eval_queries)
        return {
            "strategy": strategy,
            "total_queries": n,
            "hit_rate": round(hits / n, 4) if n > 0 else 0.0,
            "mrr": round(sum(reciprocal_ranks) / n, 4) if n > 0 else 0.0,
            "mean_docs_returned": round(total_docs / n, 1) if n > 0 else 0.0,
        }


def get_ensemble_retriever(
    vector_store: VectorStore,
    **kwargs,
) -> EnsembleRetriever:
    """Factory function para EnsembleRetriever."""
    return EnsembleRetriever(vector_store=vector_store, **kwargs)
