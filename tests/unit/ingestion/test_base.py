"""
Tests unitarios para src/ingestion/base.py

Cubre:
  - IngestionResult: success, chunk_count, add_error
  - BaseLoader.load_multiple: éxito total, fallo parcial, stop_on_first_error
  - Excepciones del dominio
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.ingestion.base import (
    BaseLoader,
    IngestionError,
    IngestionResult,
    LoaderProtocol,
    LoaderUnavailableError,
    UnsupportedFormatError,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_doc(content: str = "texto", page: int = 1, source: str = "test.pdf") -> Document:
    return Document(page_content=content, metadata={"page": page, "source": source})


class GoodLoader(BaseLoader):
    """Loader que siempre funciona — retorna 3 chunks."""

    @property
    def loader_type(self) -> str:
        return "good_loader"

    def load(self, path: Path) -> list[Document]:
        return [make_doc(f"chunk_{i}", page=i) for i in range(1, 4)]

    def supports(self, path: Path, mime_type: str) -> bool:
        return True


class FailLoader(BaseLoader):
    """Loader que siempre lanza IngestionError."""

    @property
    def loader_type(self) -> str:
        return "fail_loader"

    def load(self, path: Path) -> list[Document]:
        raise IngestionError("error simulado", path=path)

    def supports(self, path: Path, mime_type: str) -> bool:
        return False


class AlternatingLoader(BaseLoader):
    """Loader que falla en archivos con índice par (simula lote mixto)."""

    def __init__(self) -> None:
        self._call_count = 0

    @property
    def loader_type(self) -> str:
        return "alternating_loader"

    def load(self, path: Path) -> list[Document]:
        self._call_count += 1
        if self._call_count % 2 == 0:
            raise IngestionError(f"fallo en llamada {self._call_count}", path=path)
        return [make_doc(f"ok_{self._call_count}")]

    def supports(self, path: Path, mime_type: str) -> bool:
        return True


# ─── Tests: IngestionResult ───────────────────────────────────────────────────

class TestIngestionResult:
    def test_success_true_when_has_documents(self, tmp_path: Path) -> None:
        result = IngestionResult(source=tmp_path / "a.pdf")
        result.documents = [make_doc()]
        assert result.success is True

    def test_success_false_when_no_documents(self, tmp_path: Path) -> None:
        result = IngestionResult(source=tmp_path / "a.pdf")
        assert result.success is False

    def test_chunk_count(self, tmp_path: Path) -> None:
        result = IngestionResult(source=tmp_path / "a.pdf")
        result.documents = [make_doc(), make_doc(), make_doc()]
        assert result.chunk_count == 3

    def test_add_error_accumulates(self, tmp_path: Path) -> None:
        result = IngestionResult(source=tmp_path / "a.pdf")
        result.add_error("error 1")
        result.add_error("error 2")
        assert len(result.errors) == 2
        assert "error 1" in result.errors

    def test_repr_contains_status(self, tmp_path: Path) -> None:
        result = IngestionResult(source=tmp_path / "a.pdf", loader_used="pymupdf")
        result.documents = [make_doc()]
        assert "OK" in repr(result)

    def test_repr_failed_when_no_docs(self, tmp_path: Path) -> None:
        result = IngestionResult(source=tmp_path / "a.pdf")
        assert "FAILED" in repr(result)


# ─── Tests: BaseLoader.load_multiple ──────────────────────────────────────────

class TestLoadMultiple:
    def test_all_success_returns_all_results(self, tmp_path: Path) -> None:
        files = [tmp_path / f"doc{i}.pdf" for i in range(3)]
        for f in files:
            f.write_bytes(b"%PDF-1.4")  # contenido mínimo

        loader = GoodLoader()
        results = loader.load_multiple(files)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert all(r.chunk_count == 3 for r in results)

    def test_partial_failure_continues(self, tmp_path: Path) -> None:
        """Debe procesar todos los archivos aunque uno falle."""
        files = [tmp_path / f"doc{i}.pdf" for i in range(4)]
        for f in files:
            f.write_bytes(b"%PDF-1.4")

        loader = AlternatingLoader()
        results = loader.load_multiple(files)

        assert len(results) == 4
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        assert len(successful) == 2
        assert len(failed) == 2

    def test_stop_on_first_error_raises(self, tmp_path: Path) -> None:
        files = [tmp_path / f"doc{i}.pdf" for i in range(3)]
        for f in files:
            f.write_bytes(b"%PDF-1.4")

        loader = FailLoader()

        with pytest.raises(IngestionError):
            loader.load_multiple(files, stop_on_first_error=True)

    def test_missing_file_adds_error_and_continues(self, tmp_path: Path) -> None:
        existing = tmp_path / "exists.pdf"
        existing.write_bytes(b"%PDF-1.4")
        missing = tmp_path / "does_not_exist.pdf"

        loader = GoodLoader()
        results = loader.load_multiple([existing, missing])

        assert len(results) == 2
        # El primer archivo tiene éxito
        assert results[0].success is True
        # El segundo falla con error descriptivo
        assert results[1].success is False
        assert len(results[1].errors) > 0

    def test_empty_input_returns_empty_list(self) -> None:
        loader = GoodLoader()
        results = loader.load_multiple([])
        assert results == []

    def test_loader_unavailable_always_propagates(self, tmp_path: Path) -> None:
        """LoaderUnavailableError debe salir siempre, sin importar stop_on_first_error."""
        class MissingDepLoader(BaseLoader):
            @property
            def loader_type(self) -> str:
                return "missing_dep"

            def load(self, path: Path) -> list[Document]:
                raise LoaderUnavailableError("easyocr no instalado")

            def supports(self, path: Path, mime_type: str) -> bool:
                return False

        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4")

        loader = MissingDepLoader()
        with pytest.raises(LoaderUnavailableError):
            loader.load_multiple([f], stop_on_first_error=False)

    def test_loader_used_set_in_result(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4")
        loader = GoodLoader()
        results = loader.load_multiple([f])
        assert results[0].loader_used == "good_loader"

    def test_page_count_estimated_from_metadata(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4")
        loader = GoodLoader()  # retorna 3 docs con page=1,2,3
        results = loader.load_multiple([f])
        assert results[0].pages_processed == 3


# ─── Tests: LoaderProtocol ────────────────────────────────────────────────────

class TestLoaderProtocol:
    def test_good_loader_satisfies_protocol(self) -> None:
        loader = GoodLoader()
        assert isinstance(loader, LoaderProtocol)

    def test_arbitrary_class_without_methods_fails_protocol(self) -> None:
        class NotALoader:
            pass

        assert not isinstance(NotALoader(), LoaderProtocol)


# ─── Tests: Excepciones ───────────────────────────────────────────────────────

class TestExceptions:
    def test_ingestion_error_str_includes_path(self, tmp_path: Path) -> None:
        path = tmp_path / "test.pdf"
        exc = IngestionError("algo salió mal", path=path)
        assert "test.pdf" in str(exc)

    def test_ingestion_error_str_includes_cause(self) -> None:
        cause = ValueError("causa raíz")
        exc = IngestionError("error wrapper", cause=cause)
        assert "ValueError" in str(exc)

    def test_unsupported_format_is_ingestion_error(self) -> None:
        exc = UnsupportedFormatError("formato no soportado")
        assert isinstance(exc, IngestionError)

    def test_loader_unavailable_is_ingestion_error(self) -> None:
        exc = LoaderUnavailableError("dependencia faltante")
        assert isinstance(exc, IngestionError)
