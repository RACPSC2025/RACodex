"""
Interfaces base para el módulo de retrieval de Fénix RAG.

Diseño:
  - `RetrieverProtocol`  — contrato estructural (duck-typing, no herencia obligatoria)
  - `BaseRetriever`      — ABC con comportamiento compartido y logging
  - `RetrievalResult`    — resultado tipado que viaja por el pipeline
  - `RetrievalQuery`     — query enriquecida con filtros y parámetros
  - Excepciones propias con contexto suficiente para logging y retry

Principio central:
  Ningún retriever sabe de los otros. La composición ocurre en
  `ensemble.py` usando el patrón Composite. Esto permite testear
  cada retriever de forma aislada y combinarlos sin acoplamiento.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from langchain_core.documents import Document

from src.config.logging import get_logger

log = get_logger(__name__)


# ─── Excepciones del dominio de retrieval ─────────────────────────────────────

class RetrievalError(Exception):
    """Error base del pipeline de retrieval."""

    def __init__(
        self,
        message: str,
        query: str = "",
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.query = query
        self.cause = cause

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.query:
            parts.append(f"query={self.query[:80]!r}")
        if self.cause:
            parts.append(f"cause={type(self.cause).__name__}: {self.cause}")
        return " | ".join(parts)


class RetrieverUnavailableError(RetrievalError):
    """Una dependencia requerida por el retriever no está disponible."""


class VectorStoreNotInitializedError(RetrievalError):
    """Se intentó buscar antes de indexar documentos en el vector store."""


# ─── Query enriquecida ────────────────────────────────────────────────────────

@dataclass
class RetrievalQuery:
    """
    Query con metadatos de contexto para retrieval filtrado.

    Los filtros se propagan a Chroma para retrieval pre-filtrado
    (más eficiente que filtrar post-retrieval en lotes grandes).
    """
    text: str                               # query del usuario, tal cual
    top_k: int = 10                         # documentos a recuperar
    filters: dict[str, str] = field(default_factory=dict)  # metadata filters
    collection_name: str | None = None      # None = colección por defecto
    rerank: bool = True                     # aplicar reranking al resultado
    min_score: float = 0.0                  # score mínimo (0.0 = sin filtro)

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("RetrievalQuery.text no puede estar vacío.")
        self.text = self.text.strip()


# ─── Resultado tipado ─────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    Resultado de una operación de retrieval.

    Encapsula los documentos recuperados junto con trazabilidad
    del retriever que los produjo y métricas de la operación.
    """
    query: str
    documents: list[Document] = field(default_factory=list)
    retriever_used: str = ""
    elapsed_ms: float = 0.0
    total_candidates: int = 0     # candidatos antes de reranking/filtrado

    @property
    def count(self) -> int:
        return len(self.documents)

    @property
    def is_empty(self) -> bool:
        return len(self.documents) == 0

    def top(self, n: int) -> list[Document]:
        """Retorna los n primeros documentos."""
        return self.documents[:n]

    def __repr__(self) -> str:
        return (
            f"RetrievalResult("
            f"retriever={self.retriever_used!r}, "
            f"docs={self.count}, "
            f"candidates={self.total_candidates}, "
            f"elapsed={self.elapsed_ms:.1f}ms)"
        )


# ─── Protocol (structural subtyping) ─────────────────────────────────────────

@runtime_checkable
class RetrieverProtocol(Protocol):
    """Contrato estructural para cualquier retriever de Fénix RAG."""

    @property
    def retriever_type(self) -> str:
        """Identificador único. Ej: 'vector', 'bm25', 'hybrid', 'hierarchical'."""
        ...

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Ejecuta una búsqueda y retorna documentos rankeados."""
        ...

    def is_ready(self) -> bool:
        """True si el retriever tiene documentos indexados y está listo."""
        ...


# ─── ABC base ─────────────────────────────────────────────────────────────────

class BaseRetriever(ABC):
    """
    Clase base abstracta con comportamiento compartido para todos los retrievers.

    Implementa logging estructurado, medición de latencia y manejo
    consistente de errores. Las subclases implementan `_retrieve`.
    """

    @property
    @abstractmethod
    def retriever_type(self) -> str:
        ...

    @abstractmethod
    def _retrieve(self, query: RetrievalQuery) -> list[Document]:
        """
        Lógica de retrieval específica del retriever.

        Las subclases implementan este método. `retrieve()` (público)
        añade timing, logging y manejo de errores alrededor de este.
        """
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        ...

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """
        Ejecuta retrieval con logging y medición de latencia.

        Este método es el punto de entrada público. No debe ser
        sobreescrito por subclases — sobreescribir `_retrieve`.
        """
        import time

        if not self.is_ready():
            raise VectorStoreNotInitializedError(
                f"Retriever {self.retriever_type!r} no está inicializado. "
                "Indexa documentos antes de buscar.",
                query=query.text,
            )

        log.debug(
            "retrieval_start",
            retriever=self.retriever_type,
            query=query.text[:80],
            top_k=query.top_k,
            filters=query.filters,
        )

        start = time.perf_counter()
        try:
            documents = self._retrieve(query)
        except (VectorStoreNotInitializedError, RetrieverUnavailableError):
            raise
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            log.error(
                "retrieval_failed",
                retriever=self.retriever_type,
                query=query.text[:80],
                elapsed_ms=f"{elapsed:.1f}",
                error=str(exc),
            )
            raise RetrievalError(
                f"Error en retriever {self.retriever_type!r}: {exc}",
                query=query.text,
                cause=exc,
            ) from exc

        elapsed = (time.perf_counter() - start) * 1000

        result = RetrievalResult(
            query=query.text,
            documents=documents,
            retriever_used=self.retriever_type,
            elapsed_ms=elapsed,
            total_candidates=len(documents),
        )

        log.info(
            "retrieval_complete",
            retriever=self.retriever_type,
            docs=result.count,
            elapsed_ms=f"{elapsed:.1f}",
        )

        return result

    def __repr__(self) -> str:
        ready = "ready" if self.is_ready() else "not_ready"
        return f"{self.__class__.__name__}(type={self.retriever_type!r}, {ready})"
