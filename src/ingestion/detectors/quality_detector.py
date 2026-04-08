"""
Detector de calidad de documentos PDF.

Determina si un PDF contiene texto seleccionable (nativo) o
es una imagen escaneada/fotografía que requiere OCR.

Estrategia de detección (en orden de confiabilidad):
  1. Conteo de caracteres por página (método principal)
  2. Ratio de imágenes vs texto en la página
  3. Heurística de fuentes embebidas

Umbrales calibrados con el Decreto 1072 y documentos similares:
  - >= 50 chars/página → PDF nativo (texto seleccionable)
  - <  50 chars/página → Escaneado / requiere OCR
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.logging import get_logger
from src.config.settings import get_settings

log = get_logger(__name__)


# ─── Resultado de análisis de calidad ─────────────────────────────────────────

@dataclass(frozen=True)
class PDFQualityResult:
    """Resultado del análisis de calidad de un PDF."""
    path: Path
    is_native: bool             # True = tiene texto seleccionable
    is_scanned: bool            # True = requiere OCR
    avg_chars_per_page: float
    total_pages: int
    pages_with_text: int
    pages_without_text: int
    has_embedded_fonts: bool
    confidence: float           # 0.0 – 1.0, qué tan segura es la clasificación

    @property
    def requires_ocr(self) -> bool:
        """Alias semántico para mayor legibilidad en el pipeline."""
        return self.is_scanned

    @property
    def quality_label(self) -> str:
        if self.is_native and self.confidence >= 0.8:
            return "native_high_confidence"
        if self.is_native:
            return "native_low_confidence"
        if self.confidence >= 0.8:
            return "scanned_high_confidence"
        return "scanned_low_confidence"

    def __str__(self) -> str:
        return (
            f"{self.path.name} → {self.quality_label} | "
            f"avg_chars/page={self.avg_chars_per_page:.1f} | "
            f"pages_with_text={self.pages_with_text}/{self.total_pages}"
        )


# ─── Detector ─────────────────────────────────────────────────────────────────

class PDFQualityDetector:
    """
    Analiza la calidad textual de un PDF para decidir si necesita OCR.

    Usa PyMuPDF (fitz) que ya es dependencia del proyecto.
    No requiere dependencias adicionales.
    """

    def __init__(self, text_threshold: int | None = None) -> None:
        """
        Args:
            text_threshold: Chars/página mínimos para considerar PDF nativo.
                           Si None, usa PDF_TEXT_QUALITY_THRESHOLD de settings.
        """
        settings = get_settings()
        self.text_threshold = text_threshold or settings.pdf_text_quality_threshold

    def analyze(self, path: Path) -> PDFQualityResult:
        """
        Analiza la calidad textual de un PDF.

        Args:
            path: Ruta al PDF. Debe existir y ser un PDF válido.

        Returns:
            PDFQualityResult con clasificación y métricas.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si el archivo no es un PDF válido.
        """
        import fitz  # noqa: PLC0415 — lazy import, PyMuPDF

        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {path}")

        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise ValueError(f"No se pudo abrir como PDF: {path}") from exc

        total_pages = len(doc)
        char_counts: list[int] = []
        pages_with_text = 0
        pages_without_text = 0
        has_fonts = False

        try:
            for page in doc:
                text = page.get_text("text").strip()
                char_count = len(text)
                char_counts.append(char_count)

                if char_count >= self.text_threshold:
                    pages_with_text += 1
                else:
                    pages_without_text += 1

                # Detectar fuentes embebidas (indicador fuerte de PDF nativo)
                if not has_fonts and page.get_fonts():
                    has_fonts = True

        finally:
            doc.close()

        avg_chars = sum(char_counts) / total_pages if total_pages > 0 else 0.0

        # ── Clasificación con cálculo de confianza ─────────────────────────
        is_native = avg_chars >= self.text_threshold

        # Confianza basada en qué tan lejos estamos del umbral
        # y qué proporción de páginas tienen texto
        text_ratio = pages_with_text / total_pages if total_pages > 0 else 0.0

        if is_native:
            # Más lejos del umbral hacia arriba = más confianza
            distance_factor = min(avg_chars / (self.text_threshold * 3), 1.0)
            confidence = 0.5 + (0.3 * distance_factor) + (0.2 * text_ratio)
        else:
            # Más cerca de 0 chars = más seguro que está escaneado
            nearness_to_zero = 1.0 - min(avg_chars / self.text_threshold, 1.0)
            confidence = 0.5 + (0.3 * nearness_to_zero) + (0.2 * (1.0 - text_ratio))

        # Bonus de confianza si hay fuentes embebidas (indicador fuerte de nativo)
        if has_fonts and is_native:
            confidence = min(confidence + 0.15, 1.0)
        elif not has_fonts and not is_native:
            confidence = min(confidence + 0.1, 1.0)

        result = PDFQualityResult(
            path=path,
            is_native=is_native,
            is_scanned=not is_native,
            avg_chars_per_page=round(avg_chars, 2),
            total_pages=total_pages,
            pages_with_text=pages_with_text,
            pages_without_text=pages_without_text,
            has_embedded_fonts=has_fonts,
            confidence=round(confidence, 3),
        )

        log.info(
            "pdf_quality_analyzed",
            file=path.name,
            quality=result.quality_label,
            avg_chars_per_page=result.avg_chars_per_page,
            pages=f"{pages_with_text}/{total_pages}",
            has_fonts=has_fonts,
            confidence=result.confidence,
        )

        return result

    def is_scanned(self, path: Path) -> bool:
        """
        Método de conveniencia: retorna True si el PDF necesita OCR.

        Para uso rápido en condicionales sin necesitar el resultado completo.
        """
        return self.analyze(path).requires_ocr

    def classify_batch(self, paths: list[Path]) -> dict[str, list[Path]]:
        """
        Clasifica múltiples PDFs en nativos vs escaneados.

        Args:
            paths: Lista de rutas a PDFs.

        Returns:
            Dict con claves "native" y "scanned", cada una con su lista de paths.
        """
        result: dict[str, list[Path]] = {"native": [], "scanned": []}

        for path in paths:
            try:
                analysis = self.analyze(path)
                key = "native" if analysis.is_native else "scanned"
                result[key].append(path)
            except Exception as exc:
                log.warning(
                    "quality_detection_failed",
                    file=str(path),
                    error=str(exc),
                )
                # Por defecto: intentar con OCR (más seguro que omitir)
                result["scanned"].append(path)

        log.info(
            "batch_quality_classified",
            native=len(result["native"]),
            scanned=len(result["scanned"]),
        )
        return result


# ─── Instancia por defecto ────────────────────────────────────────────────────

_default_quality_detector: PDFQualityDetector | None = None


def get_quality_detector() -> PDFQualityDetector:
    """Retorna la instancia singleton del quality detector (lazy init)."""
    global _default_quality_detector  # noqa: PLW0603
    if _default_quality_detector is None:
        _default_quality_detector = PDFQualityDetector()
    return _default_quality_detector


def analyze_pdf_quality(path: Path | str) -> PDFQualityResult:
    """Función de conveniencia: analiza un PDF con el detector por defecto."""
    return get_quality_detector().analyze(Path(path))
