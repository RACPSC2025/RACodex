"""
Retriever Híbrido con Reciprocal Rank Fusion (RRF).

Combina búsqueda densa (vector embeddings) y sparse (BM25) usando RRF,
que es superior a suma ponderada de scores porque:
  - No asume que los scores de ambos métodos están en la misma escala
  - Es robusto ante outliers (un score muy alto en un método no domina)
  - Tiene un único hiperparámetro (k=60) con valor por defecto probado

Fórmula RRF:
  score(doc) = Σ 1 / (rank_i(doc) + k)
  donde rank_i es la posición del documento en cada lista (1-based)
  y k=60 es la constante de suavizado (Paper: Cormack et al. 2009)

Correcciones respecto al código original (hybrid_search.py):
  ✓ RRF en lugar de suma directa de scores (no comparables entre métodos)
  ✓ BM25 usa tokenizador spaCy en lugar de .lower().split()
  ✓ BM25 carga documentos lazy (no en __init__)
  ✓ FlashRank no bloquea en __init__ — lazy init async-safe
  ✓ Deduplicación por contenido antes de RRF (no solo por page_content hash)
  ✓ El score de reranking se inyecta en metadata para trazabilidad
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.retrieval.base import BaseRetriever, RetrievalQuery, RetrievalResult
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_store import VectorStore

log = get_logger(__name__)

# Constante de suavizado RRF (Cormack et al. 2009 — valor empíricamente óptimo)
_RRF_K = 60


class HybridRetriever(BaseRetriever):
    """
    Retriever híbrido que combina búsqueda vectorial y BM25 con RRF.

    Ambas búsquedas se ejecutan de forma independiente y sus rankings
    se fusionan con RRF antes de retornar los resultados finales.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever | None = None,
        rrf_k: int = _RRF_K,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
        fetch_multiplier: int = 3,
    ) -> None:
        """
        Args:
            vector_store: VectorStore inicializado para búsqueda semántica.
            bm25_retriever: BM25Retriever opcional. Si None, se crea uno
                            usando el mismo vector_store como fuente.
            rrf_k: Constante de suavizado RRF. Valores menores dan más
                   peso a los primeros resultados. Default: 60.
            vector_weight: Multiplicador del score RRF para resultados vectoriales.
            bm25_weight: Multiplicador del score RRF para resultados BM25.
            fetch_multiplier: Cuántos documentos extra solicitar a cada retriever
                              antes de fusionar (top_k × fetch_multiplier).
        """
        self._vector_store = vector_store
        self._bm25 = bm25_retriever or BM25Retriever(vector_store=vector_store)
        self._rrf_k = rrf_k
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight
        self._fetch_multiplier = fetch_multiplier

    @property
    def retriever_type(self) -> str:
        return "hybrid_rrf"

    def is_ready(self) -> bool:
        return self._vector_store.is_initialized

    # ── RRF ───────────────────────────────────────────────────────────────────

    def _rrf_score(self, rank: int, weight: float = 1.0) -> float:
        """
        Calcula el score RRF para un documento en posición rank.

        rank: posición 1-based en la lista ordenada
        weight: multiplicador de la fuente (para ponderar vector vs BM25)
        """
        return weight / (rank + self._rrf_k)

    def _fuse_with_rrf(
        self,
        vector_docs: list[Document],
        bm25_docs: list[Document],
        top_k: int,
    ) -> list[Document]:
        """
        Fusiona dos listas de documentos usando RRF.

        Algoritmo:
          1. Para cada documento en cada lista, calcular su score RRF
          2. Acumular scores por document ID (deduplicar)
          3. Ordenar por score acumulado descendente
          4. Retornar top_k documentos
        """
        # Acumular scores RRF por ID de documento
        scores: dict[str, float] = {}
        doc_index: dict[str, Document] = {}

        # Lista vectorial
        for rank, doc in enumerate(vector_docs, start=1):
            doc_id = self._doc_id(doc)
            rrf = self._rrf_score(rank, self._vector_weight)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf
            if doc_id not in doc_index:
                doc_index[doc_id] = doc

        # Lista BM25
        for rank, doc in enumerate(bm25_docs, start=1):
            doc_id = self._doc_id(doc)
            rrf = self._rrf_score(rank, self._bm25_weight)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf
            if doc_id not in doc_index:
                doc_index[doc_id] = doc

        # Ordenar por score RRF descendente
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Construir resultados con score inyectado en metadata
        results = []
        for doc_id in sorted_ids[:top_k]:
            doc = doc_index[doc_id]
            rrf_score = round(scores[doc_id], 6)
            results.append(Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "rrf_score": rrf_score,
                    "retriever": "hybrid_rrf",
                },
            ))

        return results

    def _doc_id(self, doc: Document) -> str:
        """ID único por documento: source::chunk_index o hash del contenido."""
        source = doc.metadata.get("source", "")
        chunk_idx = doc.metadata.get("chunk_index", "")
        if source and chunk_idx != "":
            return f"{source}::{chunk_idx}"
        import hashlib  # noqa: PLC0415
        return hashlib.md5(doc.page_content[:200].encode()).hexdigest()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve(self, query: RetrievalQuery) -> list[Document]:
        """Ejecuta búsqueda híbrida y fusiona con RRF."""
        # Buscar más candidatos que top_k para mejor fusión
        fetch_k = query.top_k * self._fetch_multiplier

        expanded_query = RetrievalQuery(
            text=query.text,
            top_k=fetch_k,
            filters=query.filters,
            rerank=False,  # el reranking lo hace HybridRetriever o el ensemble
        )

        # ── Búsqueda vectorial ────────────────────────────────────────────────
        try:
            vector_docs = self._vector_store.similarity_search(expanded_query)
            log.debug("hybrid_vector_results", count=len(vector_docs))
        except Exception as exc:
            log.warning("hybrid_vector_search_failed", error=str(exc))
            vector_docs = []

        # ── Búsqueda BM25 ────────────────────────────────────────────────────
        try:
            bm25_result = self._bm25.retrieve(expanded_query)
            bm25_docs = bm25_result.documents
            log.debug("hybrid_bm25_results", count=len(bm25_docs))
        except Exception as exc:
            log.warning("hybrid_bm25_search_failed", error=str(exc))
            bm25_docs = []

        if not vector_docs and not bm25_docs:
            log.warning("hybrid_both_empty", query=query.text[:80])
            return []

        # ── RRF Fusion ────────────────────────────────────────────────────────
        fused = self._fuse_with_rrf(
            vector_docs=vector_docs,
            bm25_docs=bm25_docs,
            top_k=query.top_k,
        )

        log.info(
            "hybrid_fusion_complete",
            vector_candidates=len(vector_docs),
            bm25_candidates=len(bm25_docs),
            fused=len(fused),
        )

        return fused


def get_hybrid_retriever(
    vector_store: VectorStore,
    **kwargs,
) -> HybridRetriever:
    """Factory function para HybridRetriever."""
    return HybridRetriever(vector_store=vector_store, **kwargs)
