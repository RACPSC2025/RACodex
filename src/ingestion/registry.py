"""
LoaderRegistry — selección automática del loader óptimo.

Implementa el patrón Strategy: el registry elige la estrategia
(loader) correcta según el tipo MIME y la calidad del documento,
sin que el caller tenga que conocer los loaders internamente.

Flujo de selección:
  1. Se registran loaders con prioridad y condiciones de activación
  2. `select` detecta MIME + calidad y filtra candidatos
  3. El loader con mayor prioridad que satisface las condiciones gana
  4. Si ninguno aplica → UnsupportedFormatError claro

Los loaders se registran en `src/ingestion/__init__.py` al arrancar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.config.logging import get_logger
from src.ingestion.base import BaseLoader, UnsupportedFormatError
from src.ingestion.detectors.mime_detector import MimeDetectionResult, get_mime_detector
from src.ingestion.detectors.quality_detector import PDFQualityResult, get_quality_detector

log = get_logger(__name__)


# ─── Entrada del registry ─────────────────────────────────────────────────────

@dataclass
class LoaderEntry:
    """
    Registro de un loader con sus condiciones de activación.

    priority: mayor número = mayor prioridad (se evalúan de mayor a menor)
    condition: función que recibe (mime, quality_opt) y retorna True si aplica
    loader_factory: callable que construye la instancia del loader
    """
    loader_type: str
    priority: int
    condition: Callable[[MimeDetectionResult, PDFQualityResult | None], bool]
    loader_factory: Callable[[], BaseLoader]
    description: str = ""

    def matches(
        self,
        mime: MimeDetectionResult,
        quality: PDFQualityResult | None,
    ) -> bool:
        """Evalúa si este loader aplica para el archivo dado."""
        try:
            return self.condition(mime, quality)
        except Exception as exc:
            log.warning(
                "loader_condition_error",
                loader=self.loader_type,
                error=str(exc),
            )
            return False


# ─── Registry ─────────────────────────────────────────────────────────────────

class LoaderRegistry:
    """
    Registro central de loaders con selección automática.

    Thread-safe para lectura concurrente (los loaders se registran
    en startup y luego son solo lecturas).
    """

    def __init__(self) -> None:
        self._entries: list[LoaderEntry] = []

    def register(
        self,
        loader_type: str,
        condition: Callable[[MimeDetectionResult, PDFQualityResult | None], bool],
        loader_factory: Callable[[], BaseLoader],
        priority: int = 10,
        description: str = "",
    ) -> "LoaderRegistry":
        """
        Registra un loader con sus condiciones de activación.

        Args:
            loader_type: Identificador único (ej: "pymupdf", "ocr", "docling")
            condition: Función (mime, quality) → bool
            loader_factory: Callable que construye el loader (lazy init)
            priority: Mayor número = mayor prioridad. Default: 10
            description: Descripción legible del loader y cuándo aplica

        Returns:
            self para chaining: registry.register(...).register(...)
        """
        entry = LoaderEntry(
            loader_type=loader_type,
            priority=priority,
            condition=condition,
            loader_factory=loader_factory,
            description=description,
        )
        self._entries.append(entry)
        # Mantener ordenados por prioridad descendente
        self._entries.sort(key=lambda e: e.priority, reverse=True)

        log.debug(
            "loader_registered",
            loader_type=loader_type,
            priority=priority,
        )
        return self

    def select(self, path: Path) -> BaseLoader:
        """
        Selecciona el loader óptimo para el archivo dado.

        Detecta MIME y calidad automáticamente, evalúa condiciones
        de todos los loaders registrados y retorna el de mayor prioridad
        que aplique.

        Args:
            path: Ruta al archivo a procesar.

        Returns:
            Instancia del loader apropiado.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            UnsupportedFormatError: Si ningún loader puede manejar el archivo.
        """
        path = Path(path).resolve()

        # 1. Detección de MIME (puede lanzar UnsupportedFormatError)
        mime_detector = get_mime_detector()
        mime = mime_detector.detect(path)

        # 2. Análisis de calidad (solo para PDFs, opcional para otros)
        quality: PDFQualityResult | None = None
        if mime.is_pdf:
            try:
                quality_detector = get_quality_detector()
                quality = quality_detector.analyze(path)
            except Exception as exc:
                log.warning(
                    "quality_detection_skipped",
                    file=path.name,
                    reason=str(exc),
                )

        log.debug(
            "loader_selection_start",
            file=path.name,
            mime=mime.mime_type,
            quality=quality.quality_label if quality else "n/a",
            candidates=len(self._entries),
        )

        # 3. Evaluar candidatos (ya están ordenados por prioridad desc)
        for entry in self._entries:
            if entry.matches(mime, quality):
                loader = entry.loader_factory()
                log.info(
                    "loader_selected",
                    file=path.name,
                    loader=entry.loader_type,
                    priority=entry.priority,
                    reason=entry.description,
                )
                return loader

        raise UnsupportedFormatError(
            f"Ningún loader registrado puede procesar '{path.name}' "
            f"(MIME: {mime.mime_type}). "
            f"Loaders disponibles: {[e.loader_type for e in self._entries]}",
            path=path,
        )

    def select_and_load(self, path: Path) -> list:
        """
        Conveniencia: selecciona el loader y carga el documento en una llamada.

        Returns:
            Lista de Documents del documento procesado.
        """
        loader = self.select(path)
        return loader.load(path)

    def list_registered(self) -> list[dict]:
        """Retorna información de todos los loaders registrados (útil para debug/admin)."""
        return [
            {
                "loader_type": e.loader_type,
                "priority": e.priority,
                "description": e.description,
            }
            for e in self._entries
        ]

    def __repr__(self) -> str:
        types = [e.loader_type for e in self._entries]
        return f"LoaderRegistry(loaders={types})"


# ─── Condiciones reutilizables ────────────────────────────────────────────────

class Conditions:
    """
    Funciones de condición estándar para registrar loaders.

    Uso:
        registry.register(
            "pymupdf",
            condition=Conditions.native_pdf,
            loader_factory=lambda: PyMuPDFLoader(),
            priority=20,
        )
    """

    @staticmethod
    def native_pdf(
        mime: MimeDetectionResult,
        quality: PDFQualityResult | None,
    ) -> bool:
        """PDF con texto seleccionable — no necesita OCR."""
        return mime.is_pdf and quality is not None and quality.is_native

    @staticmethod
    def scanned_pdf(
        mime: MimeDetectionResult,
        quality: PDFQualityResult | None,
    ) -> bool:
        """PDF escaneado o fotografiado — requiere OCR."""
        return mime.is_pdf and (quality is None or quality.is_scanned)

    @staticmethod
    def any_pdf(
        mime: MimeDetectionResult,
        quality: PDFQualityResult | None,  # noqa: ARG004
    ) -> bool:
        """Cualquier PDF, nativo o escaneado."""
        return mime.is_pdf

    @staticmethod
    def word_document(
        mime: MimeDetectionResult,
        quality: PDFQualityResult | None,  # noqa: ARG004
    ) -> bool:
        """Documentos Word (.docx y .doc legacy)."""
        return mime.is_word

    @staticmethod
    def excel_spreadsheet(
        mime: MimeDetectionResult,
        quality: PDFQualityResult | None,  # noqa: ARG004
    ) -> bool:
        """Hojas de cálculo Excel (.xlsx y .xls legacy)."""
        return mime.is_excel

    @staticmethod
    def image_file(
        mime: MimeDetectionResult,
        quality: PDFQualityResult | None,  # noqa: ARG004
    ) -> bool:
        """Imágenes directas (JPEG, PNG, TIFF, etc.) — requieren OCR."""
        return mime.is_image

    @staticmethod
    def complex_pdf_heuristic(
        mime: MimeDetectionResult,
        quality: PDFQualityResult | None,
    ) -> bool:
        """
        PDFs nativos con características de complejidad (muchas páginas,
        posibles tablas). Heurística para decidir Docling vs PyMuPDF.

        Por ahora: PDFs nativos con más de 50 páginas se envían a Docling.
        Este umbral puede configurarse en settings en el futuro.
        """
        if not mime.is_pdf or quality is None or not quality.is_native:
            return False
        return quality.total_pages > 50


# ─── Instancia singleton ──────────────────────────────────────────────────────

_registry: LoaderRegistry | None = None


def get_registry() -> LoaderRegistry:
    """
    Retorna la instancia singleton del registry.

    La primera vez que se llama, el registry está vacío.
    Los loaders se registran en `src/ingestion/__init__.py`.
    """
    global _registry  # noqa: PLW0603
    if _registry is None:
        _registry = LoaderRegistry()
    return _registry
