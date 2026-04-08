"""
Wrapper de Chroma para Fénix RAG.

Responsabilidades:
  - Gestión del ciclo de vida de colecciones (crear, abrir, borrar)
  - Indexación de documentos con deduplicación por source+chunk_index
  - Búsqueda semántica con filtros de metadata
  - Health check del vector store

Por qué wrappear Chroma en lugar de usarlo directo:
  - Centraliza la configuración (persist_directory, embeddings)
  - Agrega logging estructurado con trazabilidad de operaciones
  - Permite cambiar a PGVector en Fase 6 sin tocar el código de retrieval
  - Maneja el estado (colección abierta vs recién creada)

Modelo de colecciones:
  - `fenix_legal`          — colección principal (documentos procesados)
  - `fenix_legal_summary`  — summaries para retrieval jerárquico (Fase 3.5)
  Ambas viven en el mismo persist_directory y comparten embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config.logging import get_logger
from src.config.providers import get_embeddings
from src.config.settings import get_settings
from src.retrieval.base import RetrievalQuery, VectorStoreNotInitializedError

log = get_logger(__name__)


class VectorStore:
    """
    Wrapper de Chroma con gestión de ciclo de vida de colecciones.

    Thread-safe para lecturas concurrentes. La indexación debe
    realizarse de forma secuencial o con locks externos.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: Path | None = None,
    ) -> None:
        settings = get_settings()
        self._collection_name = collection_name or settings.chroma_collection_name
        self._persist_dir = persist_directory or settings.chroma_persist_dir
        self._store: Chroma | None = None

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def is_initialized(self) -> bool:
        return self._store is not None

    # ── Ciclo de vida ─────────────────────────────────────────────────────────

    def open_or_create(self) -> "VectorStore":
        """
        Abre la colección existente o la crea si no existe.

        Idempotente — llamar múltiples veces es seguro.

        Returns:
            self para chaining.
        """
        if self._store is not None:
            return self

        self._persist_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "vector_store_opening",
            collection=self._collection_name,
            persist_dir=str(self._persist_dir),
        )

        self._store = Chroma(
            collection_name=self._collection_name,
            embedding_function=get_embeddings(),
            persist_directory=str(self._persist_dir),
        )

        count = self._safe_count()
        log.info(
            "vector_store_ready",
            collection=self._collection_name,
            existing_docs=count,
        )

        return self

    def delete_collection(self) -> None:
        """
        Elimina la colección completa (todos los documentos).

        Útil para re-indexación completa o en tests.
        """
        if self._store is None:
            self.open_or_create()

        log.warning(
            "vector_store_deleting",
            collection=self._collection_name,
        )

        self._store.delete_collection()
        self._store = None

        log.info("vector_store_deleted", collection=self._collection_name)

    def _require_store(self) -> Chroma:
        """Retorna el store o lanza error si no está inicializado."""
        if self._store is None:
            raise VectorStoreNotInitializedError(
                f"VectorStore '{self._collection_name}' no está inicializado. "
                "Llama a open_or_create() primero.",
            )
        return self._store

    def _safe_count(self) -> int:
        """Retorna el número de documentos sin lanzar si está vacío."""
        try:
            if self._store is None:
                return 0
            return self._store._collection.count()
        except Exception:
            return 0

    # ── Indexación ────────────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: list[Document],
        *,
        batch_size: int = 100,
        deduplicate: bool = True,
    ) -> int:
        """
        Indexa documentos en el vector store.

        Args:
            documents: Lista de Documents a indexar.
            batch_size: Documentos por lote (evita timeout en Chroma).
            deduplicate: Si True, asigna IDs deterministas por source+chunk_index
                         para que re-indexar el mismo documento actualice en vez
                         de duplicar.

        Returns:
            Número de documentos efectivamente indexados.
        """
        store = self._require_store()

        if not documents:
            log.warning("add_documents_empty_list")
            return 0

        # Asignar IDs deterministas para deduplicación
        if deduplicate:
            ids = self._make_ids(documents)
        else:
            ids = None

        total = len(documents)
        indexed = 0

        log.info(
            "indexing_start",
            collection=self._collection_name,
            total=total,
            batch_size=batch_size,
            deduplicate=deduplicate,
        )

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size] if ids else None

            try:
                store.add_documents(documents=batch, ids=batch_ids)
                indexed += len(batch)
                log.debug(
                    "indexing_batch_complete",
                    batch=f"{i // batch_size + 1}",
                    docs_indexed=indexed,
                    total=total,
                )
            except Exception as exc:
                log.error(
                    "indexing_batch_failed",
                    batch_start=i,
                    error=str(exc),
                )
                raise

        log.info(
            "indexing_complete",
            collection=self._collection_name,
            indexed=indexed,
            total_in_store=self._safe_count(),
        )

        return indexed

    def _make_ids(self, documents: list[Document]) -> list[str]:
        """
        Genera IDs deterministas para deduplicación.

        ID = "{source}::{chunk_index}" normalizado.
        Si el documento no tiene source/chunk_index en metadata,
        usa el hash del contenido.
        """
        import hashlib

        ids = []
        for doc in documents:
            source = doc.metadata.get("source", "")
            chunk_idx = doc.metadata.get("chunk_index", "")

            if source and chunk_idx != "":
                raw_id = f"{source}::{chunk_idx}"
            else:
                # Fallback: hash del contenido
                raw_id = hashlib.md5(doc.page_content.encode()).hexdigest()

            # Chroma requiere IDs sin caracteres especiales
            safe_id = raw_id.replace(" ", "_").replace("/", "_").replace("\\", "_")
            ids.append(safe_id[:512])  # límite de longitud de Chroma

        return ids

    # ── Búsqueda ──────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: RetrievalQuery,
    ) -> list[Document]:
        """
        Búsqueda semántica con filtros opcionales de metadata.

        Args:
            query: RetrievalQuery con texto, top_k y filtros.

        Returns:
            Lista de Documents ordenados por similitud (más similar primero).
        """
        store = self._require_store()

        # Construir filtro de Chroma desde metadata filters
        where_filter = self._build_filter(query.filters)

        try:
            if where_filter:
                results = store.similarity_search(
                    query=query.text,
                    k=query.top_k,
                    filter=where_filter,
                )
            else:
                results = store.similarity_search(
                    query=query.text,
                    k=query.top_k,
                )
        except Exception as exc:
            log.error(
                "similarity_search_failed",
                collection=self._collection_name,
                query=query.text[:80],
                error=str(exc),
            )
            raise

        return results

    def similarity_search_with_score(
        self,
        query: RetrievalQuery,
    ) -> list[tuple[Document, float]]:
        """
        Búsqueda semántica retornando documentos con scores de distancia.

        Los scores de Chroma son distancias coseno: menor = más similar.
        Se normalizan a [0,1] donde 1 = máxima similitud.
        """
        store = self._require_store()
        where_filter = self._build_filter(query.filters)

        try:
            if where_filter:
                raw_results = store.similarity_search_with_score(
                    query=query.text,
                    k=query.top_k,
                    filter=where_filter,
                )
            else:
                raw_results = store.similarity_search_with_score(
                    query=query.text,
                    k=query.top_k,
                )
        except Exception as exc:
            raise

        # Normalizar: score de Chroma es distancia L2 o coseno dependiendo de config.
        # Convertir a similitud [0,1]: sim = 1 / (1 + distance)
        normalized = [
            (doc, round(1.0 / (1.0 + distance), 4))
            for doc, distance in raw_results
        ]

        return normalized

    def _build_filter(self, filters: dict[str, str]) -> dict | None:
        """
        Convierte filtros de metadata a formato de Chroma.

        Chroma espera: {"key": {"$eq": "value"}} para igualdad.
        Para múltiples filtros: {"$and": [{...}, {...}]}
        """
        if not filters:
            return None

        conditions = [
            {key: {"$eq": value}}
            for key, value in filters.items()
            if value  # ignorar filtros con valor vacío
        ]

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ── Utilidades ────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Número de documentos en la colección."""
        if self._store is None:
            return 0
        return self._safe_count()

    def get_raw_collection(self) -> Any:
        """
        Acceso al objeto Collection interno de Chroma.

        Necesario para BM25Retriever que carga todos los documentos
        para construir su índice en memoria.
        """
        return self._require_store()._collection

    def health_check(self) -> dict[str, Any]:
        """Estado del vector store para endpoint /health."""
        try:
            count = self._safe_count()
            return {
                "vector_store": "ok",
                "collection": self._collection_name,
                "documents": count,
                "persist_dir": str(self._persist_dir),
            }
        except Exception as exc:
            return {
                "vector_store": "error",
                "collection": self._collection_name,
                "error": str(exc),
            }


# ─── Instancias singleton ──────────────────────────────────────────────────────

_default_store: VectorStore | None = None
_summary_store: VectorStore | None = None


def get_vector_store(collection_name: str | None = None) -> VectorStore:
    """
    Retorna la instancia singleton del vector store principal.

    Llama a open_or_create() automáticamente en el primer acceso.
    """
    global _default_store  # noqa: PLW0603

    if collection_name:
        # Colección específica — sin caché singleton
        return VectorStore(collection_name=collection_name).open_or_create()

    if _default_store is None:
        _default_store = VectorStore().open_or_create()

    return _default_store


def get_summary_store() -> VectorStore:
    """Retorna el vector store de summaries (para retrieval jerárquico)."""
    global _summary_store  # noqa: PLW0603

    if _summary_store is None:
        settings = get_settings()
        _summary_store = VectorStore(
            collection_name=f"{settings.chroma_collection_name}_summary"
        ).open_or_create()

    return _summary_store


def reset_vector_store_cache() -> None:
    """Resetea el caché de stores (útil en tests)."""
    global _default_store, _summary_store  # noqa: PLW0603
    _default_store = None
    _summary_store = None
