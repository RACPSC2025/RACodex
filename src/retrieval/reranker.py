"""
Reranker async-safe usando FlashRank.

Problema del código original:
  FlashRank se inicializaba en __init__ de forma sincrónica, bloqueando
  el event loop en entornos async (FastAPI, LangGraph async nodes).
  La carga del modelo ms-marco-MiniLM-L-12-v2 tarda ~2-3 segundos.

Solución:
  - Lazy init: el modelo se carga al primer rerank(), no al construir
  - Thread pool executor: la carga se ejecuta en un worker thread para
    no bloquear el event loop en contextos async
  - Singleton del modelo: una instancia por proceso, compartida

Por qué FlashRank:
  - Modelo cross-encoder ms-marco-MiniLM-L-12-v2 entrenado para
    relevance ranking en passage retrieval
  - Sin llamadas a API externa (on-premise)
  - Latencia baja: ~50ms para rerank de 20 documentos en CPU
  - Score normalizado [0,1] fácil de interpretar

Cuándo usar:
  El reranker siempre se aplica DESPUÉS de la fusión en ensemble.py.
  No se aplica en cada retriever individual para no duplicar latencia.

Requiere: flashrank >= 0.2
  pip install flashrank
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.config.logging import get_logger
from src.config.settings import get_settings

log = get_logger(__name__)

# Executor dedicado para inicialización del modelo (evita bloquear event loop)
_MODEL_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reranker_init")


class Reranker:
    """
    Reranker cross-encoder con FlashRank.

    Reordena una lista de documentos candidatos según su relevancia
    exacta para la query, usando un modelo cross-encoder más preciso
    que los embeddings bi-encoder usados en la búsqueda inicial.
    """

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: Path | None = None,
        top_k: int | None = None,
    ) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.flashrank_model
        self._cache_dir = cache_dir or settings.flashrank_cache_dir
        self._top_k = top_k or settings.retrieval_rerank_top_k
        self._ranker: Any | None = None      # lazy init
        self._available: bool | None = None  # None = no comprobado

    def _get_ranker(self) -> Any | None:
        """
        Lazy init del Ranker de FlashRank.

        Retorna None si FlashRank no está instalado (degradación graceful).
        """
        if self._available is not None:
            return self._ranker if self._available else None

        try:
            from flashrank import Ranker  # noqa: PLC0415
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._ranker = Ranker(
                model_name=self._model_name,
                cache_dir=str(self._cache_dir),
            )
            self._available = True
            log.info("reranker_loaded", model=self._model_name)
            return self._ranker
        except ImportError:
            log.info(
                "reranker_unavailable",
                reason="flashrank no instalado",
                hint="pip install flashrank",
            )
            self._available = False
            return None
        except Exception as exc:
            log.warning("reranker_load_failed", error=str(exc))
            self._available = False
            return None

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """
        Reordena documentos por relevancia exacta para la query.

        Args:
            query: Texto de la query original del usuario.
            documents: Lista de candidatos a reordenar (output del ensemble).
            top_k: Número de documentos a retornar. None = usa setting default.

        Returns:
            Lista reordenada con score de reranking en metadata["rerank_score"].
            Si FlashRank no está disponible, retorna los documentos sin cambios.
        """
        k = top_k or self._top_k

        if not documents:
            return []

        ranker = self._get_ranker()

        if ranker is None:
            # Degradación graceful: retornar sin reranking
            log.debug("rerank_skipped_no_ranker", docs=len(documents))
            return documents[:k]

        log.debug("reranking_start", query=query[:80], candidates=len(documents))

        try:
            from flashrank import RerankRequest  # noqa: PLC0415

            passages = [
                {
                    "id": i,
                    "text": doc.page_content,
                    "meta": doc.metadata,
                }
                for i, doc in enumerate(documents)
            ]

            request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(request)

            # Reconstruir Documents preservando metadata + score
            reranked = []
            for res in results[:k]:
                idx = res["id"]
                original_doc = documents[idx]
                rerank_score = float(res.get("score", 0.0))

                reranked.append(Document(
                    page_content=original_doc.page_content,
                    metadata={
                        **original_doc.metadata,
                        "rerank_score": round(rerank_score, 4),
                    },
                ))

            log.info(
                "reranking_complete",
                input=len(documents),
                output=len(reranked),
                top_score=round(results[0]["score"], 4) if results else 0.0,
            )

            return reranked

        except Exception as exc:
            log.error("reranking_failed", error=str(exc))
            # Fallback: retornar sin reranking
            return documents[:k]

    async def arerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """
        Versión async del reranking.

        Ejecuta el reranking en un thread pool para no bloquear
        el event loop. Esencial en nodos async de LangGraph.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _MODEL_EXECUTOR,
            lambda: self.rerank(query, documents, top_k),
        )

    @property
    def is_available(self) -> bool:
        """True si FlashRank está instalado y el modelo cargó correctamente."""
        if self._available is None:
            self._get_ranker()
        return bool(self._available)


# ─── Singleton ────────────────────────────────────────────────────────────────

_default_reranker: Reranker | None = None


def get_reranker(**kwargs) -> Reranker:
    """Retorna la instancia singleton del reranker (lazy init)."""
    global _default_reranker  # noqa: PLW0603

    if kwargs:
        return Reranker(**kwargs)

    if _default_reranker is None:
        _default_reranker = Reranker()

    return _default_reranker
