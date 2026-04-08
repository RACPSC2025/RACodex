"""
Interfaces base para el sistema de ingestion de Fénix RAG.

Diseño:
  - `LoaderProtocol` — structural subtyping (Protocol) para duck-typing
  - `BasePDFLoader`  — ABC con comportamiento compartido (load_multiple robusto)
  - `IngestionResult` — resultado tipado que viaja por el pipeline
  - Excepciones propias con contexto suficiente para logging y retry

Correcciones respecto al código original:
  - `load_multiple` ahora reporta resultados parciales, no silencia errores
  - Separación clara entre error irrecuperable (FileNotFoundError) y
    error de procesamiento (IngestionError) — el pipeline decide si reintenta
  - Tipado estricto en todos los métodos públicos
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from langchain_core.documents import Document

from src.config.logging import get_logger

log = get_logger(__name__)


# ─── Excepciones del dominio de ingestion ────────────────────────────────────

class IngestionError(Exception):
    """Error base del pipeline de ingestion. Incluye contexto procesable."""

    def __init__(self, message: str, path: Path | None = None, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.path = path
        self.cause = cause

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.path:
            parts.append(f"path={self.path}")
        if self.cause:
            parts.append(f"cause={type(self.cause).__name__}: {self.cause}")
        return " | ".join(parts)


class UnsupportedFormatError(IngestionError):
    """El formato del archivo no está soportado por ningún loader registrado."""


class DocumentQualityError(IngestionError):
    """El documento existe pero tiene calidad insuficiente para procesarse."""


class LoaderUnavailableError(IngestionError):
    """Una dependencia opcional requerida por el loader no está instalada."""


# ─── Resultado tipado del pipeline ────────────────────────────────────────────

@dataclass
class IngestionResult:
    """
    Resultado de procesar un documento.

    Separa el éxito parcial del fallo total:
      - `documents` tiene los chunks exitosos (puede ser vacío)
      - `errors` acumula los errores no fatales
      - `success` es False solo si NO se produjo ningún chunk
    """
    source: Path
    documents: list[Document] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    loader_used: str = ""
    pages_processed: int = 0

    @property
    def success(self) -> bool:
        return len(self.documents) > 0

    @property
    def chunk_count(self) -> int:
        return len(self.documents)

    def add_error(self, error: str) -> None:
        self.errors.append(error)
        log.warning("ingestion_partial_error", source=str(self.source), error=error)

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAILED"
        return (
            f"IngestionResult({status} | {self.source.name} | "
            f"chunks={self.chunk_count} | errors={len(self.errors)} | "
            f"loader={self.loader_used})"
        )


# ─── Protocol (structural subtyping) ─────────────────────────────────────────

@runtime_checkable
class LoaderProtocol(Protocol):
    """
    Contrato estructural para cualquier loader.

    Usar Protocol en lugar de (solo) ABC permite que loaders de terceros
    funcionen sin heredar de nuestra clase base.
    """

    @property
    def loader_type(self) -> str:
        """Identificador único del loader. Ej: 'pymupdf', 'ocr', 'docling'."""
        ...

    def load(self, path: Path) -> list[Document]:
        """Carga un único documento y retorna sus chunks como Documents."""
        ...

    def supports(self, path: Path, mime_type: str) -> bool:
        """Indica si este loader puede procesar el archivo dado."""
        ...


# ─── ABC con comportamiento compartido ───────────────────────────────────────

class BaseLoader(ABC):
    """
    Clase base abstracta para todos los loaders de Fénix RAG.

    Implementa `load_multiple` con:
      - Reporte de errores por archivo (no falla todo el lote)
      - Logging estructurado de progreso
      - Retorno de `IngestionResult` por archivo para trazabilidad

    Las subclases solo deben implementar `load`, `loader_type` y `supports`.
    """

    @property
    @abstractmethod
    def loader_type(self) -> str:
        """Identificador único del loader."""
        ...

    @abstractmethod
    def load(self, path: Path) -> list[Document]:
        """
        Carga un archivo y retorna sus chunks.

        Args:
            path: Ruta absoluta al archivo. Debe existir.

        Returns:
            Lista de Documents con page_content y metadata enriquecidos.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            IngestionError: Si el archivo existe pero no puede procesarse.
            LoaderUnavailableError: Si falta una dependencia opcional.
        """
        ...

    @abstractmethod
    def supports(self, path: Path, mime_type: str) -> bool:
        """
        Indica si este loader puede manejar el archivo dado.

        Args:
            path: Ruta al archivo.
            mime_type: MIME type detectado por python-magic (no por extensión).

        Returns:
            True si el loader puede procesar este archivo.
        """
        ...

    def load_multiple(
        self,
        paths: list[Path],
        *,
        stop_on_first_error: bool = False,
    ) -> list[IngestionResult]:
        """
        Carga múltiples archivos y retorna resultados individuales.

        Diferencia clave vs el código original:
          - No silencia errores: cada fallo queda registrado en IngestionResult
          - `stop_on_first_error=False` (default): procesa todos aunque haya fallos
          - `stop_on_first_error=True`: para en el primer error irrecuperable
          - Los chunks exitosos se retornan aunque otros archivos fallen

        Args:
            paths: Lista de rutas a procesar.
            stop_on_first_error: Si True, lanza la excepción del primer fallo.

        Returns:
            Lista de IngestionResult, uno por archivo (éxito o fracaso).
        """
        results: list[IngestionResult] = []
        total = len(paths)

        log.info(
            "batch_ingestion_started",
            loader=self.loader_type,
            total_files=total,
        )

        for idx, path in enumerate(paths, start=1):
            result = IngestionResult(source=path, loader_used=self.loader_type)
            log.debug(
                "processing_file",
                loader=self.loader_type,
                file=path.name,
                progress=f"{idx}/{total}",
            )

            try:
                path = Path(path)  # normalizar si viene como str
                if not path.exists():
                    raise FileNotFoundError(f"Archivo no encontrado: {path}")

                documents = self.load(path)
                result.documents = documents
                result.pages_processed = self._count_pages(documents)

                log.info(
                    "file_ingested",
                    loader=self.loader_type,
                    file=path.name,
                    chunks=result.chunk_count,
                )

            except FileNotFoundError as exc:
                msg = f"Archivo no encontrado: {path}"
                result.add_error(msg)
                if stop_on_first_error:
                    raise IngestionError(msg, path=path, cause=exc) from exc

            except LoaderUnavailableError:
                # Re-raise siempre: indica un problema de entorno, no de datos
                raise

            except IngestionError as exc:
                result.add_error(str(exc))
                if stop_on_first_error:
                    raise

            except Exception as exc:
                msg = f"Error inesperado procesando {path.name}: {exc}"
                result.add_error(msg)
                log.exception("unexpected_ingestion_error", file=str(path), error=str(exc))
                if stop_on_first_error:
                    raise IngestionError(msg, path=path, cause=exc) from exc

            results.append(result)

        successful = sum(1 for r in results if r.success)
        failed = total - successful
        log.info(
            "batch_ingestion_complete",
            loader=self.loader_type,
            total=total,
            successful=successful,
            failed=failed,
            total_chunks=sum(r.chunk_count for r in results),
        )

        return results

    def _count_pages(self, documents: list[Document]) -> int:
        """Estima páginas procesadas desde la metadata de los chunks."""
        pages = {
            doc.metadata.get("page")
            for doc in documents
            if doc.metadata.get("page") is not None
        }
        return len(pages)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loader_type={self.loader_type!r})"
