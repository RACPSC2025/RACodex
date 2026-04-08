"""
Loader de PDF complejo usando IBM Docling.

Docling convierte PDFs a Markdown estructurado preservando:
  - Tablas como | col | col | (Markdown GFM)
  - Jerarquía de encabezados # / ## / ###
  - Listas numeradas y con viñetas
  - Columnas y layouts multi-columna

Cuándo usar Docling vs PyMuPDF:
  - Docling: PDFs con tablas de normas, layouts complejos, multi-columna,
             documentos con mezcla de texto e imágenes estructuradas.
  - PyMuPDF: Decretos y resoluciones de texto corrido sin tablas (más rápido).
  El DocumentClassifierSkill (Fase 4) decide automáticamente.

Chunking post-Docling:
  El Markdown producido por Docling ya tiene estructura explícita con
  encabezados. Usamos un MarkdownSplitter que respeta esos límites
  en lugar del LegalChunker orientado a ARTÍCULO/CAPÍTULO.

Requiere: docling>=2.0
  pip install docling
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.ingestion.base import BaseLoader, IngestionError, LoaderUnavailableError
from src.ingestion.processors.metadata_extractor import get_metadata_extractor
from src.ingestion.processors.text_cleaner import get_cleaner

log = get_logger(__name__)

# Headers de Markdown que actúan como límites de chunk en Docling output
_MARKDOWN_HEADERS = [
    ("#",   "h1"),
    ("##",  "h2"),
    ("###", "h3"),
]


class DoclingLoader(BaseLoader):
    """
    Loader de PDF complejo usando IBM Docling → Markdown estructurado.

    El pipeline completo:
      1. docling.DocumentConverter convierte PDF → DoclingDocument
      2. export_to_markdown() produce Markdown con tablas y encabezados
      3. MarkdownHeaderTextSplitter divide en secciones respetando headers
      4. Sub-chunking por tamaño en secciones largas
      5. MetadataExtractor enriquece cada chunk
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        max_pages: int = 400,
        num_workers: int | None = None,
    ) -> None:
        settings = get_settings()
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._max_pages = max_pages
        self._num_workers = num_workers or settings.docling_num_workers

        self._converter: object | None = None  # lazy

    @property
    def loader_type(self) -> str:
        return "docling"

    def supports(self, path: Path, mime_type: str) -> bool:
        return mime_type == "application/pdf"

    def _get_converter(self) -> object:
        """Lazy init de DocumentConverter con error descriptivo."""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter  # noqa: PLC0415
            except ImportError as exc:
                raise LoaderUnavailableError(
                    "docling no está instalado.\n"
                    "Ejecuta: pip install docling"
                ) from exc

            log.info("docling_initializing", workers=self._num_workers)
            self._converter = DocumentConverter()
            log.info("docling_ready")

        return self._converter

    def _convert_to_markdown(self, path: Path) -> str:
        """
        Convierte un PDF a Markdown usando Docling.

        Args:
            path: Ruta al PDF.

        Returns:
            String Markdown con estructura preservada.

        Raises:
            IngestionError: Si Docling falla al convertir el archivo.
        """
        converter = self._get_converter()

        log.info("docling_converting", file=path.name)

        try:
            result = converter.convert(str(path))
        except Exception as exc:
            raise IngestionError(
                f"Docling no pudo convertir '{path.name}'",
                path=path,
                cause=exc,
            ) from exc

        if result is None:
            raise IngestionError(
                f"Docling retornó resultado nulo para '{path.name}'",
                path=path,
            )

        try:
            markdown = result.document.export_to_markdown()
        except AttributeError as exc:
            raise IngestionError(
                f"No se pudo exportar a Markdown: {exc}",
                path=path,
                cause=exc,
            ) from exc

        if not markdown or not markdown.strip():
            raise IngestionError(
                f"Docling produjo Markdown vacío para '{path.name}'",
                path=path,
            )

        log.info(
            "docling_converted",
            file=path.name,
            markdown_chars=len(markdown),
        )

        return markdown

    def _split_markdown(self, markdown: str) -> list[str]:
        """
        Divide el Markdown en chunks respetando la jerarquía de headers.

        Estrategia de dos pasadas:
          1. MarkdownHeaderTextSplitter divide en secciones por # / ## / ###
          2. RecursiveCharacterTextSplitter sub-divide secciones que superan chunk_size
        """
        # Primera pasada: respetar headers
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_MARKDOWN_HEADERS,
            strip_headers=False,  # mantener encabezados en el contenido
            return_each_line=False,
        )

        header_docs = header_splitter.split_text(markdown)

        # Segunda pasada: sub-dividir secciones largas
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

        chunks: list[str] = []
        for doc in header_docs:
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            if len(content) <= self._chunk_size:
                chunks.append(content)
            else:
                sub_chunks = char_splitter.split_text(content)
                chunks.extend(sub_chunks)

        return [c for c in chunks if len(c.strip()) >= 50]

    def _build_documents(
        self,
        chunks: list[str],
        source_path: Path,
    ) -> list[Document]:
        """
        Construye Documents con metadata enriquecida desde chunks Markdown.

        Para Docling usamos el MetadataExtractor estándar pero con ajustes
        para contexto Markdown (puede no haber ARTÍCULO explícito).
        """
        extractor = get_metadata_extractor()
        documents: list[Document] = []

        for idx, chunk in enumerate(chunks):
            # El MetadataExtractor funciona igual sobre texto markdown/legal
            meta = extractor.extract(
                chunk_text=chunk,
                source_path=source_path,
                chunk_index=idx,
                loader_type=self.loader_type,
            )

            # Header contextual
            header = extractor.build_contextual_header(meta)
            content = header + chunk.strip() if header else chunk.strip()
            meta.chunk_size = len(content)

            documents.append(Document(
                page_content=content,
                metadata=meta.to_dict(),
            ))

        return documents

    def load(self, path: Path) -> list[Document]:
        """
        Convierte un PDF complejo a Documents usando Docling.

        Args:
            path: Ruta al PDF con tablas o layout complejo.

        Returns:
            Lista de Documents con Markdown estructurado y metadata.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            IngestionError: Si Docling falla.
            LoaderUnavailableError: Si docling no está instalado.
        """
        path = Path(path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {path}")

        log.info("docling_loading", file=path.name)

        # 1. Convertir a Markdown
        markdown = self._convert_to_markdown(path)

        # 2. Limpiar (usar perfil default — no aplicar reglas de ARTÍCULO
        #    porque Docling ya estructuró el documento con headers)
        cleaner = get_cleaner("default")
        clean_markdown = cleaner.clean(markdown)

        # 3. Dividir en chunks respetando estructura Markdown
        chunks = self._split_markdown(clean_markdown)

        if not chunks:
            raise IngestionError(
                f"Docling no produjo chunks para '{path.name}'",
                path=path,
            )

        # 4. Construir Documents con metadata
        documents = self._build_documents(chunks, source_path=path)

        log.info(
            "docling_loaded",
            file=path.name,
            markdown_chars=len(markdown),
            chunks=len(documents),
        )

        return documents


def get_docling_loader(**kwargs) -> DoclingLoader:
    """Factory function para DoclingLoader."""
    return DoclingLoader(**kwargs)
