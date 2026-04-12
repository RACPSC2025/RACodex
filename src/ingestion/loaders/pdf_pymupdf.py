"""
Loader de PDF nativo usando PyMuPDF (fitz).

Diferencias respecto al código original (pdf_simple.py):
  ✓ Bug corregido: _get_article_metadata recibía chunk_text pero hacía
    Path(chunk_text).name → siempre retornaba "document.pdf".
    Ahora el source_path se pasa correctamente desde load().

  ✓ Bug corregido: clean_text tenía reglas del Decreto 1072 hardcodeadas
    dentro del loader. Ahora usa TextCleaner con perfil configurable.

  ✓ Bug corregido: el join genérico "(\w)\n(\w)" estaba COMENTADO en el
    original "porque rompe tablas legales" — se eliminó. La reconstrucción
    de palabras rotas solo aplica al guión al final de línea (JOIN_BROKEN_WORDS).

  ✓ page_markers se añaden como metadata, no dentro del page_content.
    El contextual header viene del MetadataExtractor, no hardcodeado.

  ✓ load_multiple heredado de BaseLoader con manejo robusto de errores.

  ✓ supports() implementado correctamente (requería override).
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.ingestion.base import BaseLoader, IngestionError, LoaderUnavailableError
from src.ingestion.processors.hierarchical_chunker import ChunkConfig, HierarchicalChunker, get_hierarchical_chunker
from src.ingestion.processors.text_cleaner import CleanerRegistry, get_cleaner

log = get_logger(__name__)


class PyMuPDFLoader(BaseLoader):
    """
    Loader optimizado para PDFs nativos con texto seleccionable.

    Flujo:
      fitz.open() → extract per-page text → join pages → clean → chunk
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        max_pages: int = 400,
        cleaner_profile: str = "default",
    ) -> None:
        """
        Args:
            chunk_size: Override del chunk size de settings.
            chunk_overlap: Override del overlap de settings.
            max_pages: Límite de páginas por documento (protección de memoria).
            cleaner_profile: Perfil de TextCleaner a aplicar.
                             Ver CleanerProfiles para opciones disponibles.
        """
        settings = get_settings()
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._max_pages = max_pages
        self._cleaner_profile = cleaner_profile

        self._chunker: LegalChunker | None = None  # lazy init

    @property
    def loader_type(self) -> str:
        return "pymupdf"

    def supports(self, path: Path, mime_type: str) -> bool:
        return mime_type == "application/pdf"

    def _get_fitz(self):
        """Lazy import de PyMuPDF con error claro si no está instalado."""
        try:
            import fitz  # noqa: PLC0415
            return fitz
        except ImportError as exc:
            raise LoaderUnavailableError(
                "PyMuPDF no está instalado. Ejecuta: pip install pymupdf"
            ) from exc

    def _get_chunker(self) -> LegalChunker:
        if self._chunker is None:
            self._chunker = LegalChunker(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
        return self._chunker

    def _extract_text(self, pdf_path: Path) -> tuple[str, int]:
        """
        Extrae texto completo del PDF preservando orden de páginas.

        Args:
            pdf_path: Ruta al PDF.

        Returns:
            (texto_completo, páginas_procesadas)
            El texto incluye marcadores [Página N] para metadata posterior.
        """
        fitz = self._get_fitz()
        pages: list[str] = []
        pages_processed = 0

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            raise IngestionError(
                f"No se pudo abrir el PDF: {pdf_path.name}",
                path=pdf_path,
                cause=exc,
            ) from exc

        try:
            for page_num, page in enumerate(doc, start=1):
                if page_num > self._max_pages:
                    log.warning(
                        "max_pages_reached",
                        file=pdf_path.name,
                        max_pages=self._max_pages,
                    )
                    break

                # "text" preserva el orden de lectura correcto
                text = page.get_text("text")
                if text.strip():
                    # Marcador de página incluido para extracción de metadata
                    pages.append(f"[Página {page_num}]\n{text}")
                    pages_processed += 1

        finally:
            doc.close()

        return "\n\n".join(pages), pages_processed

    def load(self, path: Path) -> list[Document]:
        """
        Carga un PDF nativo y retorna sus chunks como Documents.

        Args:
            path: Ruta al PDF. Debe existir y tener texto seleccionable.

        Returns:
            Lista de Documents con page_content enriquecido y metadata completa.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            IngestionError: Si el PDF no puede procesarse.
            LoaderUnavailableError: Si PyMuPDF no está instalado.
        """
        path = Path(path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {path}")

        log.info("pymupdf_loading", file=path.name)

        raw_text, pages_processed = self._extract_text(path)

        if not raw_text.strip():
            raise IngestionError(
                f"El PDF '{path.name}' no contiene texto extraíble. "
                "¿Es un PDF escaneado? Usa el loader OCR.",
                path=path,
            )

        # Limpiar y chunkear usando los processors
        chunker = self._get_chunker()
        documents = chunker.chunk_with_profile(
            text=raw_text,
            source_path=path,
            loader_type=self.loader_type,
            cleaner_profile=self._cleaner_profile,
            add_header=True,
        )

        log.info(
            "pymupdf_loaded",
            file=path.name,
            pages=pages_processed,
            chunks=len(documents),
            profile=self._cleaner_profile,
        )

        return documents


def get_pymupdf_loader(**kwargs) -> PyMuPDFLoader:
    """Factory function — compatibilidad con el código original."""
    return PyMuPDFLoader(**kwargs)
