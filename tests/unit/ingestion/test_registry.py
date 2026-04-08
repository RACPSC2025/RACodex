"""
Tests unitarios para LoaderRegistry.

Verifica:
  - Registro y selección por prioridad
  - Condiciones estándar (Conditions.*)
  - Fallback correcto cuando ningún loader aplica
  - list_registered retorna info correcta
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.ingestion.base import BaseLoader, UnsupportedFormatError
from src.ingestion.detectors.mime_detector import MimeDetectionResult
from src.ingestion.detectors.quality_detector import PDFQualityResult
from src.ingestion.registry import Conditions, LoaderRegistry


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_mime(
    path: Path,
    mime_type: str = "application/pdf",
    is_supported: bool = True,
) -> MimeDetectionResult:
    label_map = {
        "application/pdf": "PDF",
        "image/jpeg": "Imagen JPEG",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "Word (.docx)",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "Excel (.xlsx)",
    }
    return MimeDetectionResult(
        path=path,
        mime_type=mime_type,
        is_supported=is_supported,
        label=label_map.get(mime_type, mime_type),
    )


def make_quality(
    path: Path,
    is_native: bool = True,
    total_pages: int = 10,
    confidence: float = 0.9,
) -> PDFQualityResult:
    return PDFQualityResult(
        path=path,
        is_native=is_native,
        is_scanned=not is_native,
        avg_chars_per_page=200.0 if is_native else 5.0,
        total_pages=total_pages,
        pages_with_text=total_pages if is_native else 0,
        pages_without_text=0 if is_native else total_pages,
        has_embedded_fonts=is_native,
        confidence=confidence,
    )


class StubLoader(BaseLoader):
    """Loader stub para tests."""
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def loader_type(self) -> str:
        return self.name

    def load(self, path: Path) -> list[Document]:
        return [Document(page_content=f"loaded by {self.name}", metadata={})]

    def supports(self, path: Path, mime_type: str) -> bool:
        return True


def make_registry_with_mocked_detection(
    tmp_path: Path,
    mime_type: str = "application/pdf",
    is_native: bool = True,
    total_pages: int = 10,
) -> tuple[LoaderRegistry, Path]:
    """Crea un registry con detección mockeada."""
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4")

    mime = make_mime(path, mime_type)
    quality = make_quality(path, is_native=is_native, total_pages=total_pages)

    registry = LoaderRegistry()
    return registry, path, mime, quality


# ─── Tests: Registro ──────────────────────────────────────────────────────────

class TestLoaderRegistration:
    def test_register_adds_entry(self) -> None:
        registry = LoaderRegistry()
        registry.register(
            "test_loader",
            condition=lambda m, q: True,
            loader_factory=lambda: StubLoader("test_loader"),
        )
        entries = registry.list_registered()
        assert any(e["loader_type"] == "test_loader" for e in entries)

    def test_register_returns_self_for_chaining(self) -> None:
        registry = LoaderRegistry()
        result = registry.register(
            "a", condition=lambda m, q: True, loader_factory=lambda: StubLoader("a")
        ).register(
            "b", condition=lambda m, q: False, loader_factory=lambda: StubLoader("b")
        )
        assert result is registry

    def test_entries_sorted_by_priority_descending(self) -> None:
        registry = LoaderRegistry()
        registry.register("low", condition=lambda m, q: True, loader_factory=lambda: StubLoader("low"), priority=5)
        registry.register("high", condition=lambda m, q: True, loader_factory=lambda: StubLoader("high"), priority=50)
        registry.register("mid", condition=lambda m, q: True, loader_factory=lambda: StubLoader("mid"), priority=20)

        entries = registry.list_registered()
        priorities = [e["priority"] for e in entries]
        assert priorities == sorted(priorities, reverse=True)


# ─── Tests: Selección ─────────────────────────────────────────────────────────

class TestLoaderSelection:
    def test_highest_priority_loader_selected(self, tmp_path: Path) -> None:
        registry, path, mime, quality = make_registry_with_mocked_detection(tmp_path)

        registry.register("low", condition=lambda m, q: True, loader_factory=lambda: StubLoader("low"), priority=5)
        registry.register("high", condition=lambda m, q: True, loader_factory=lambda: StubLoader("high"), priority=50)

        with patch("src.ingestion.registry.get_mime_detector") as mock_mime_det, \
             patch("src.ingestion.registry.get_quality_detector") as mock_qual_det:
            mock_mime_det.return_value.detect.return_value = mime
            mock_qual_det.return_value.analyze.return_value = quality

            loader = registry.select(path)

        assert loader.loader_type == "high"

    def test_condition_false_skips_loader(self, tmp_path: Path) -> None:
        registry, path, mime, quality = make_registry_with_mocked_detection(tmp_path)

        registry.register("never", condition=lambda m, q: False, loader_factory=lambda: StubLoader("never"), priority=100)
        registry.register("always", condition=lambda m, q: True, loader_factory=lambda: StubLoader("always"), priority=10)

        with patch("src.ingestion.registry.get_mime_detector") as mock_mime_det, \
             patch("src.ingestion.registry.get_quality_detector") as mock_qual_det:
            mock_mime_det.return_value.detect.return_value = mime
            mock_qual_det.return_value.analyze.return_value = quality

            loader = registry.select(path)

        assert loader.loader_type == "always"

    def test_no_matching_loader_raises(self, tmp_path: Path) -> None:
        registry, path, mime, quality = make_registry_with_mocked_detection(tmp_path)
        registry.register("never", condition=lambda m, q: False, loader_factory=lambda: StubLoader("never"))

        with patch("src.ingestion.registry.get_mime_detector") as mock_mime_det, \
             patch("src.ingestion.registry.get_quality_detector") as mock_qual_det:
            mock_mime_det.return_value.detect.return_value = mime
            mock_qual_det.return_value.analyze.return_value = quality

            with pytest.raises(UnsupportedFormatError):
                registry.select(path)

    def test_loader_factory_called_lazily(self, tmp_path: Path) -> None:
        """El factory no debe llamarse hasta que select() elija el loader."""
        registry, path, mime, quality = make_registry_with_mocked_detection(tmp_path)
        factory_calls = []

        def counting_factory() -> StubLoader:
            factory_calls.append(1)
            return StubLoader("counted")

        registry.register("counted", condition=lambda m, q: True, loader_factory=counting_factory)

        assert len(factory_calls) == 0  # no llamado aún

        with patch("src.ingestion.registry.get_mime_detector") as mock_mime_det, \
             patch("src.ingestion.registry.get_quality_detector") as mock_qual_det:
            mock_mime_det.return_value.detect.return_value = mime
            mock_qual_det.return_value.analyze.return_value = quality
            registry.select(path)

        assert len(factory_calls) == 1  # llamado exactamente una vez


# ─── Tests: Conditions ────────────────────────────────────────────────────────

class TestConditions:
    def test_native_pdf_true_for_native_pdf(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.pdf"
        mime = make_mime(path, "application/pdf")
        quality = make_quality(path, is_native=True)
        assert Conditions.native_pdf(mime, quality) is True

    def test_native_pdf_false_for_scanned(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.pdf"
        mime = make_mime(path, "application/pdf")
        quality = make_quality(path, is_native=False)
        assert Conditions.native_pdf(mime, quality) is False

    def test_scanned_pdf_true_for_scanned(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.pdf"
        mime = make_mime(path, "application/pdf")
        quality = make_quality(path, is_native=False)
        assert Conditions.scanned_pdf(mime, quality) is True

    def test_scanned_pdf_true_when_quality_none(self, tmp_path: Path) -> None:
        """Sin datos de calidad, asumir escaneado (más seguro)."""
        path = tmp_path / "doc.pdf"
        mime = make_mime(path, "application/pdf")
        assert Conditions.scanned_pdf(mime, None) is True

    def test_word_document_matches_docx(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.docx"
        mime = make_mime(
            path,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert Conditions.word_document(mime, None) is True

    def test_excel_spreadsheet_matches_xlsx(self, tmp_path: Path) -> None:
        path = tmp_path / "data.xlsx"
        mime = make_mime(
            path,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        assert Conditions.excel_spreadsheet(mime, None) is True

    def test_image_file_matches_jpeg(self, tmp_path: Path) -> None:
        path = tmp_path / "foto.jpg"
        mime = make_mime(path, "image/jpeg")
        assert Conditions.image_file(mime, None) is True

    def test_complex_pdf_heuristic_above_threshold(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.pdf"
        mime = make_mime(path, "application/pdf")
        quality = make_quality(path, is_native=True, total_pages=51)
        assert Conditions.complex_pdf_heuristic(mime, quality) is True

    def test_complex_pdf_heuristic_below_threshold(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.pdf"
        mime = make_mime(path, "application/pdf")
        quality = make_quality(path, is_native=True, total_pages=30)
        assert Conditions.complex_pdf_heuristic(mime, quality) is False

    def test_word_condition_false_for_pdf(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.pdf"
        mime = make_mime(path, "application/pdf")
        assert Conditions.word_document(mime, None) is False
