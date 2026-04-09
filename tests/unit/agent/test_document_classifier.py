"""
Tests del DocumentClassifierSkill — clasificación de documentos.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.skills.document_classifier import (
    DocumentClassifierSkill,
    get_document_classifier,
)


# ─── Inicialización ───────────────────────────────────────────────────────────

class TestDocumentClassifierInit:
    """Tests de inicialización del classifier."""

    def test_default_threshold(self) -> None:
        classifier = DocumentClassifierSkill()
        assert classifier._llm_threshold == 0.70

    def test_custom_threshold(self) -> None:
        classifier = DocumentClassifierSkill(llm_threshold=0.50)
        assert classifier._llm_threshold == 0.50

    def test_disable_llm_fallback(self) -> None:
        classifier = DocumentClassifierSkill(use_llm_fallback=False)
        assert classifier._use_llm_fallback is False

    def test_singleton_returns_instance(self) -> None:
        classifier = get_document_classifier()
        assert isinstance(classifier, DocumentClassifierSkill)


# ─── Clasificación por MIME ──────────────────────────────────────────────────

class TestDocumentClassifierByMime:
    """Tests de clasificación por tipo MIME."""

    def test_word_document(self, sample_docx_path: Path) -> None:
        classifier = DocumentClassifierSkill(use_llm_fallback=False)
        plan = classifier.classify(sample_docx_path)
        assert plan["loader_type"] == "word"
        assert plan["requires_ocr"] is False
        assert plan["confidence"] >= 0.90

    def test_excel_document(self, sample_xlsx_path: Path) -> None:
        classifier = DocumentClassifierSkill(use_llm_fallback=False)
        plan = classifier.classify(sample_xlsx_path)
        assert plan["loader_type"] == "excel"
        assert plan["requires_ocr"] is False
        assert plan["confidence"] >= 0.95

    def test_file_not_found(self) -> None:
        classifier = DocumentClassifierSkill()
        with pytest.raises(FileNotFoundError):
            classifier.classify("/nonexistent/file.pdf")

    def test_fallback_plan_on_failure(self) -> None:
        """Cuando la detección falla, retorna un fallback plan."""
        classifier = DocumentClassifierSkill()
        with patch("src.agent.skills.document_classifier.detect_mime") as mock_detect:
            mock_detect.side_effect = Exception("MIME error")
            plan = classifier.classify("/some/file.pdf")
            assert plan["confidence"] == 0.0
            assert "MIME detection failed" in plan["reasoning"]


# ─── Detección de tipo de documento ─────────────────────────────────────────

class TestDocumentTypeInference:
    """Tests de inferencia de tipo de documento."""

    def test_infer_documentation_from_name(self, tmp_path: Path) -> None:
        classifier = DocumentClassifierSkill()
        path = tmp_path / "getting_started_guide.pdf"
        doc_type = classifier._infer_doc_type_from_name(path)
        assert doc_type == "documentation"

    def test_infer_api_docs_from_name(self, tmp_path: Path) -> None:
        classifier = DocumentClassifierSkill()
        path = tmp_path / "api_endpoints_reference.pdf"
        doc_type = classifier._infer_doc_type_from_name(path)
        assert doc_type == "api_docs"

    def test_infer_architecture_from_name(self, tmp_path: Path) -> None:
        classifier = DocumentClassifierSkill()
        path = tmp_path / "system_architecture_design.pdf"
        doc_type = classifier._infer_doc_type_from_name(path)
        assert doc_type == "architecture"

    def test_infer_standard_when_no_signal(self, tmp_path: Path) -> None:
        classifier = DocumentClassifierSkill()
        path = tmp_path / "random_document.pdf"
        doc_type = classifier._infer_doc_type_from_name(path)
        assert doc_type == "standard"

    def test_infer_from_content(self, tmp_path: Path) -> None:
        classifier = DocumentClassifierSkill()
        path = tmp_path / "unknown.pdf"
        content = "This API endpoint accepts a request body and returns a 200 response."
        doc_type = classifier._infer_doc_type_from_name(path, content)
        assert doc_type == "api_docs"

    def test_infer_contract_from_content(self, tmp_path: Path) -> None:
        classifier = DocumentClassifierSkill()
        path = tmp_path / "unknown.pdf"
        content = "The parties agree to the following terms and conditions."
        doc_type = classifier._infer_doc_type_from_name(path, content)
        assert doc_type == "contract"


# ─── Detección de documento técnico ─────────────────────────────────────────

class TestTechnicalDocDetection:
    """Tests de detección de documentos técnicos."""

    def test_detects_by_name(self, tmp_path: Path) -> None:
        classifier = DocumentClassifierSkill()
        path = tmp_path / "api_documentation.pdf"
        assert classifier._looks_like_technical_doc(path) is True

    def test_detects_by_content(self, tmp_path: Path) -> None:
        classifier = DocumentClassifierSkill()
        path = tmp_path / "unknown.pdf"
        content = "This function takes a parameter and returns a value."
        assert classifier._looks_like_technical_doc(path, content) is True

    def test_not_technical(self, tmp_path: Path) -> None:
        classifier = DocumentClassifierSkill()
        path = tmp_path / "random.txt"
        assert classifier._looks_like_technical_doc(path) is False


# ─── classify_many ──────────────────────────────────────────────────────────

class TestClassifyMany:
    """Tests de clasificación múltiple."""

    def test_classify_many_returns_list(self, sample_pdf_path: Path, sample_docx_path: Path) -> None:
        classifier = DocumentClassifierSkill(use_llm_fallback=False)
        plans = classifier.classify_many([sample_pdf_path, sample_docx_path])
        assert isinstance(plans, list)
        assert len(plans) == 2

    def test_classify_many_continues_on_error(self, sample_pdf_path: Path) -> None:
        classifier = DocumentClassifierSkill()
        plans = classifier.classify_many([sample_pdf_path, "/nonexistent/file.pdf"])
        assert len(plans) == 2
        # El segundo debe ser un fallback plan
        assert plans[1]["confidence"] == 0.0
