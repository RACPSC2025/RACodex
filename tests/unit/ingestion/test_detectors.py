"""
Tests unitarios para detectores de MIME type y calidad PDF.

Estrategia de testing:
  - MimeDetector: mockeamos python-magic para no requerir libmagic en CI
  - PDFQualityDetector: creamos PDFs mínimos reales con PyMuPDF
  - Verificamos clasificación correcta de nativo vs escaneado
"""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.base import UnsupportedFormatError
from src.ingestion.detectors.mime_detector import (
    MIME_LABELS,
    SUPPORTED_MIME_TYPES,
    MimeDetectionResult,
    MimeDetector,
    detect_mime,
)
from src.ingestion.detectors.quality_detector import (
    PDFQualityDetector,
    PDFQualityResult,
    analyze_pdf_quality,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def create_minimal_pdf(path: Path, content: str = "Texto de prueba.") -> Path:
    """Crea un PDF mínimo válido con texto usando PyMuPDF."""
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), content)
        doc.save(str(path))
        doc.close()
    except ImportError:
        # Fallback: PDF mínimo hardcodeado sin texto (simula escaneado)
        path.write_bytes(
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj "
            b"xref\n0 4\n0000000000 65535 f\ntrailer<</Root 1 0 R/Size 4>>\n%%EOF"
        )
    return path


# ─── Tests: MimeDetector ──────────────────────────────────────────────────────

class TestMimeDetector:
    """Todos los tests mockean python-magic para no requerir libmagic."""

    def _make_detector_with_mock(self, mime_to_return: str) -> tuple[MimeDetector, MagicMock]:
        detector = MimeDetector()
        mock_magic = MagicMock()
        mock_magic.from_file.return_value = mime_to_return
        detector._magic_instance = mock_magic
        return detector, mock_magic

    def test_pdf_detected_correctly(self, tmp_path: Path) -> None:
        f = tmp_path / "decreto.pdf"
        f.write_bytes(b"%PDF-1.4")

        detector, mock = self._make_detector_with_mock("application/pdf")
        result = detector.detect(f)

        assert result.mime_type == "application/pdf"
        assert result.is_pdf is True
        assert result.is_supported is True
        assert result.path == f.resolve()
        mock.from_file.assert_called_once_with(str(f.resolve()))

    def test_docx_detected_correctly(self, tmp_path: Path) -> None:
        f = tmp_path / "contrato.docx"
        f.write_bytes(b"PK\x03\x04")  # magic bytes de ZIP (docx es un ZIP)

        detector, _ = self._make_detector_with_mock(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        result = detector.detect(f)

        assert result.is_word is True
        assert result.is_supported is True

    def test_xlsx_detected_correctly(self, tmp_path: Path) -> None:
        f = tmp_path / "tabla.xlsx"
        f.write_bytes(b"PK\x03\x04")

        detector, _ = self._make_detector_with_mock(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        result = detector.detect(f)

        assert result.is_excel is True

    def test_image_pdf_jpeg_detected(self, tmp_path: Path) -> None:
        f = tmp_path / "escaneo.jpg"
        f.write_bytes(b"\xff\xd8\xff")

        detector, _ = self._make_detector_with_mock("image/jpeg")
        result = detector.detect(f)

        assert result.is_image is True
        assert result.is_supported is True

    def test_unsupported_mime_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00\x00\x00\x18ftyp")

        detector, _ = self._make_detector_with_mock("video/mp4")

        with pytest.raises(UnsupportedFormatError) as exc_info:
            detector.detect(f)

        assert "video/mp4" in str(exc_info.value)

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        detector = MimeDetector()
        with pytest.raises(FileNotFoundError):
            detector.detect(tmp_path / "no_existe.pdf")

    def test_mime_with_charset_param_is_normalized(self, tmp_path: Path) -> None:
        """'application/pdf; charset=utf-8' debe normalizarse a 'application/pdf'."""
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")

        detector, _ = self._make_detector_with_mock("application/pdf; charset=utf-8")
        result = detector.detect(f)

        assert result.mime_type == "application/pdf"

    def test_detect_many_skips_unsupported_by_default(self, tmp_path: Path) -> None:
        pdf = tmp_path / "ok.pdf"
        pdf.write_bytes(b"%PDF")
        mp4 = tmp_path / "video.mp4"
        mp4.write_bytes(b"\x00\x00\x00\x18")

        detector = MimeDetector()
        call_count = [0]

        def mock_from_file(path: str) -> str:
            call_count[0] += 1
            if "ok.pdf" in path:
                return "application/pdf"
            return "video/mp4"

        mock_magic = MagicMock()
        mock_magic.from_file.side_effect = mock_from_file
        detector._magic_instance = mock_magic

        results = detector.detect_many([pdf, mp4], skip_unsupported=True)

        assert len(results) == 1
        assert results[0].is_pdf is True

    def test_detect_many_raises_on_unsupported_when_skip_false(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.mp4"
        f.write_bytes(b"\x00")

        detector = MimeDetector()
        mock_magic = MagicMock()
        mock_magic.from_file.return_value = "video/mp4"
        detector._magic_instance = mock_magic

        with pytest.raises(UnsupportedFormatError):
            detector.detect_many([f], skip_unsupported=False)

    def test_missing_python_magic_gives_clear_error(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")

        detector = MimeDetector()
        # Forzar que el import falle
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            # El magic_instance ya existe si fue creado antes — lo reseteamos
            detector._magic_instance = None
            with pytest.raises(ImportError) as exc_info:
                detector.detect(f)
            assert "python-magic" in str(exc_info.value)


# ─── Tests: PDFQualityDetector ────────────────────────────────────────────────

class TestPDFQualityDetector:
    def test_native_pdf_classified_correctly(self, tmp_path: Path) -> None:
        """Un PDF con texto real debe clasificarse como nativo."""
        path = create_minimal_pdf(tmp_path / "nativo.pdf", content="A" * 200)
        detector = PDFQualityDetector(text_threshold=50)

        try:
            result = detector.analyze(path)
            assert result.is_native is True
            assert result.is_scanned is False
            assert result.requires_ocr is False
        except ImportError:
            pytest.skip("PyMuPDF no disponible en este entorno")

    def test_scanned_pdf_classified_correctly(self, tmp_path: Path) -> None:
        """Un PDF sin texto debe clasificarse como escaneado."""
        # PDF minimal sin texto
        path = tmp_path / "escaneado.pdf"
        path.write_bytes(
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj "
            b"xref\n0 4\n0000000000 65535 f \n"
            b"trailer<</Root 1 0 R/Size 4>>\n%%EOF"
        )
        detector = PDFQualityDetector(text_threshold=50)

        try:
            result = detector.analyze(path)
            assert result.is_scanned is True
            assert result.requires_ocr is True
        except ImportError:
            pytest.skip("PyMuPDF no disponible")

    def test_quality_result_has_all_fields(self, tmp_path: Path) -> None:
        path = create_minimal_pdf(tmp_path / "doc.pdf")
        detector = PDFQualityDetector(text_threshold=50)

        try:
            result = detector.analyze(path)
            assert isinstance(result.path, Path)
            assert isinstance(result.avg_chars_per_page, float)
            assert result.total_pages >= 1
            assert 0.0 <= result.confidence <= 1.0
        except ImportError:
            pytest.skip("PyMuPDF no disponible")

    def test_is_scanned_convenience_method(self, tmp_path: Path) -> None:
        path = create_minimal_pdf(tmp_path / "doc.pdf", content="A" * 300)
        detector = PDFQualityDetector(text_threshold=50)

        try:
            # Un PDF con texto no es escaneado
            assert detector.is_scanned(path) is False
        except ImportError:
            pytest.skip("PyMuPDF no disponible")

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        detector = PDFQualityDetector()
        with pytest.raises(FileNotFoundError):
            detector.analyze(tmp_path / "no_existe.pdf")

    def test_quality_label_values(self, tmp_path: Path) -> None:
        path = create_minimal_pdf(tmp_path / "doc.pdf", content="A" * 500)
        detector = PDFQualityDetector(text_threshold=50)

        try:
            result = detector.analyze(path)
            valid_labels = {
                "native_high_confidence",
                "native_low_confidence",
                "scanned_high_confidence",
                "scanned_low_confidence",
            }
            assert result.quality_label in valid_labels
        except ImportError:
            pytest.skip("PyMuPDF no disponible")

    def test_classify_batch_returns_correct_keys(self, tmp_path: Path) -> None:
        paths = [create_minimal_pdf(tmp_path / f"doc{i}.pdf") for i in range(2)]
        detector = PDFQualityDetector(text_threshold=50)

        try:
            result = detector.classify_batch(paths)
            assert "native" in result
            assert "scanned" in result
            assert isinstance(result["native"], list)
            assert isinstance(result["scanned"], list)
        except ImportError:
            pytest.skip("PyMuPDF no disponible")
