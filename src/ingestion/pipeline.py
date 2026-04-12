"""
IngestionPipeline — orquesta el flujo completo: detect → load → clean → chunk → index.

Responsabilidad: convertir un archivo crudo en chunks indexados en ChromaDB.

Flujo:
  1. Detect MIME type (python-magic)
  2. Detect PDF quality (native vs scanned)
  3. Select optimal loader (LoaderRegistry)
  4. Load document(s)
  5. Clean text (CleanerRule profiles)
  6. Chunk (LegalHierarchicalChunker)
  7. Extract metadata (MetadataExtractor)
  8. Index into ChromaDB

Uso:
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    result = pipeline.ingest_file("/path/to/document.pdf")
    print(f"Ingested {result.pages_processed} pages, {len(result.documents)} chunks")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.config.logging import get_logger
from src.config.settings import get_settings

log = get_logger(__name__)


# ─── Resultado del pipeline ───────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Resultado completo de ingestar un archivo.

    Incluye tanto el resultado del loader como los metadatos
    de indexación en Chroma.
    """
    source_path: Path
    success: bool
    documents: list[Document] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    loader_used: str = ""
    document_type: str = ""
    mime_type: str = ""
    pages_processed: int = 0
    chunk_count: int = 0
    page_count: int = 0
    classifier_confidence: float = 0.0
    indexed: bool = False
    index_errors: list[str] = field(default_factory=list)


# ─── Pipeline principal ──────────────────────────────────────────────────────

class IngestionPipeline:
    """
    Orquestador del flujo de ingestión completo.

    Cada etapa es un método separado para facilitar testing
    y permitir reemplazo individual (ej: custom chunker).
    """

    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        cleaner_profile: str = "default",
        collection_name: str | None = None,
        augment_with_questions: bool = False,
        questions_per_chunk: int = 3,
    ) -> None:
        """
        Args:
            chunk_size: Tamaño de chunk (override de settings).
            chunk_overlap: Overlap entre chunks (override de settings).
            cleaner_profile: Perfil de limpieza ("default", "technical", "ocr_output").
            collection_name: Nombre de la colección Chroma (override de settings).
            augment_with_questions: Si True, genera preguntas por chunk durante ingestión.
            questions_per_chunk: Número de preguntas por chunk (si augment=True).
        """
        settings = get_settings()

        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.cleaner_profile = cleaner_profile
        self.collection_name = collection_name or settings.chroma_collection_name
        self.augment_with_questions = augment_with_questions
        self.questions_per_chunk = questions_per_chunk

        # Document type from IngestionPlan (propagated to chunker)
        self.document_type: str | None = None

        # Componentes lazy-initialized
        self._registry: Any | None = None
        self._vector_store: Any | None = None

    def ingest_file(self, plan: dict) -> PipelineResult:
        """
        Procesa un archivo completo a través del pipeline.

        Args:
            plan: IngestionPlan (dict) con la metadata extraida por el classifier.

        Returns:
            PipelineResult con chunks, metadatos y errores.
        """
        source = Path(plan.get("source_path", ""))
        self.document_type = plan.get("document_type", "unknown")
        self.cleaner_profile = plan.get("cleaner_profile", "default")
        
        # Obtenemos las detecciones provenientes del plan pre-clasificado
        mime_type = plan.get("mime_type", "application/octet-stream")
        confidence = plan.get("confidence", 1.0)
        loader_name = plan.get("loader_type")

        if not source.exists():
            return PipelineResult(
                source_path=source,
                success=False,
                errors=[f"Archivo no encontrado: {source}"],
            )

        log.info("pipeline_start", path=str(source), size=source.stat().st_size)

        result = PipelineResult(source_path=source, success=False)

        try:
            result.mime_type = mime_type
            log.debug("pipeline_using_plan_mime", mime=result.mime_type)

            # Ya no inferimos ni mimetype ni quality, y apuntamos al loader sugerido.
            if self._registry is None:
                from src.ingestion.registry import LoaderRegistry
                self._registry = LoaderRegistry() # Default instances 
            
            # Etapa 3: Instanciar loader
            # Asumimos que loader_name existe, de lo contrario usamos _select_loader para re-detectar
            if loader_name and hasattr(self._registry, "get_loader"):
                loader = self._registry.get_loader(loader_name)
            else:
                 # Fallback: re-detección (sólo debería activarse si falló extraer un plan válido)
                 loader = self._select_loader(source, result.mime_type, {"confidence": confidence})
                 
            if not loader:
                 raise ValueError(f"No se pudo inicializar un loader válido para {loader_name}")

            result.loader_used = type(loader).__name__
            result.classifier_confidence = confidence

            log.info(
                "pipeline_loader_selected",
                loader=result.loader_used,
                confidence=result.classifier_confidence,
            )

            # Etapa 4: Load document
            loaded_docs = self._load(source, loader)
            result.page_count = len(loaded_docs)

            log.debug(
                "pipeline_loaded",
                pages=len(loaded_docs),
                loader=result.loader_used,
            )

            # Etapa 5: Clean text
            cleaned_docs = self._clean(loaded_docs)

            # Etapa 6: Chunk
            chunked_docs = self._chunk(cleaned_docs)
            result.chunk_count = len(chunked_docs)

            # Etapa 7: Extract metadata
            enriched_docs = self._extract_metadata(chunked_docs, source)

            # Etapa 8: Index into Chroma
            index_errors = self._index(enriched_docs)
            result.indexed = len(index_errors) == 0
            result.index_errors = index_errors

            # Resultado final
            result.documents = enriched_docs
            result.errors = index_errors
            result.success = len(enriched_docs) > 0
            result.pages_processed = result.page_count
            result.document_type = self.document_type

            log.info(
                "pipeline_complete",
                path=str(source),
                chunks=result.chunk_count,
                pages=result.page_count,
                indexed=result.indexed,
                success=result.success,
            )

        except Exception as exc:
            log.error(
                "pipeline_failed",
                path=str(source),
                error=str(exc),
            )
            result.errors.append(f"Pipeline error: {str(exc)}")
            result.success = False

        return result

    # ── Etapa 1: MIME Type Detection ──────────────────────────────────────────

    def _detect_mime_type(self, source: Path) -> str:
        """Detecta el MIME type real del archivo (por bytes, no extensión)."""
        from src.ingestion.detectors.mime_detector import MimeDetector  # noqa: PLC0415

        detector = MimeDetector()
        detection = detector.detect(source)
        return detection.mime_type

    # ── Etapa 2: PDF Quality Detection ────────────────────────────────────────

    def _detect_quality(
        self,
        source: Path,
        mime_type: str,
    ) -> dict[str, Any]:
        """Detecta si un PDF es nativo o escaneado."""
        from src.ingestion.detectors.quality_detector import PDFQualityDetector  # noqa: PLC0415

        if not mime_type.startswith("application/pdf"):
            return {"confidence": 1.0, "page_count": 0, "document_type": "unknown"}

        detector = PDFQualityDetector()
        quality = detector.detect(source)

        return {
            "is_scanned": quality.is_scanned,
            "confidence": quality.confidence,
            "page_count": quality.page_count,
            "document_type": "pdf",
        }

    # ── Etapa 3: Loader Selection ─────────────────────────────────────────────

    def _select_loader(
        self,
        source: Path,
        mime_type: str,
        quality_info: dict[str, Any],
    ) -> Any:
        """Selecciona el loader óptimo según MIME type y calidad."""
        from src.ingestion.registry import LoaderRegistry  # noqa: PLC0415

        if self._registry is None:
            self._registry = LoaderRegistry()

        # El registry ya hace detección interna de MIME + quality
        loader = self._registry.select(source)

        return loader

    # ── Etapa 4: Load ─────────────────────────────────────────────────────────

    def _load(self, source: Path, loader: Any) -> list[Document]:
        """Carga el documento con el loader seleccionado."""
        result = loader.load_multiple([str(source)])

        if not result:
            from src.ingestion.base import IngestionError  # noqa: PLC0415
            raise IngestionError(
                f"Loader {type(loader).__name__} no produjo documentos para {source}",
                path=source,
            )

        docs = result[0].documents
        errors = result[0].errors

        if errors:
            log.warning("pipeline_load_warnings", errors=errors)

        if not docs:
            raise IngestionError(
                f"Documento vacío o no legible: {source}",
                path=source,
            )

        return docs

    # ── Etapa 5: Clean ────────────────────────────────────────────────────────

    def _clean(self, docs: list[Document]) -> list[Document]:
        """Aplica reglas de limpieza según el perfil configurado."""
        from src.ingestion.processors.text_cleaner import CleanerRegistry  # noqa: PLC0415

        cleaner = CleanerRegistry()
        cleaned = cleaner.clean(docs, profile=self.cleaner_profile)
        return cleaned

    # ── Etapa 6: Chunk ────────────────────────────────────────────────────────

    def _chunk(self, docs: list[Document]) -> list[Document]:
        """Aplica chunking adaptativo preservando estructura del documento."""
        from src.ingestion.processors.adaptive_chunker import AdaptiveChunker  # noqa: PLC0415

        # Usar AdaptiveChunker con document_type propagado del IngestionPlan
        return AdaptiveChunker.chunk(
            docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            document_type=self.document_type,
        )

    # ── Etapa 7: Metadata Extraction ──────────────────────────────────────────

    def _extract_metadata(
        self,
        docs: list[Document],
        source: Path,
    ) -> list[Document]:
        """Extrae metadatos estructurados y enriquece cada chunk."""
        from src.ingestion.processors.metadata_extractor import MetadataExtractor  # noqa: PLC0415

        extractor = MetadataExtractor()
        enriched = extractor.extract(docs, source=str(source))
        return enriched

    # ── Etapa 8: Index ────────────────────────────────────────────────────────

    def _index(self, docs: list[Document]) -> list[str]:
        """Indexa los chunks en ChromaDB. Retorna lista de errores."""
        from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415

        vs = get_vector_store()

        if not vs.is_initialized:
            vs.open_or_create()

        errors: list[str] = []

        # Augmentar con preguntas si está habilitado
        if self.augment_with_questions:
            from src.ingestion.processors.document_augmenter import augment_and_index  # noqa: PLC0415
            try:
                stats = augment_and_index(
                    chunks=docs,
                    vector_store=vs,
                    questions_per_chunk=self.questions_per_chunk,
                )
                log.info(
                    "index_with_augmentation",
                    original=stats["original_chunks"],
                    questions=stats["questions_added"],
                    total=stats["total_indexed"],
                )
            except Exception as exc:
                errors.append(f"Augmentation index error: {str(exc)}")
                # Fallback: indexar sin augmentación
                try:
                    vs.add_documents(docs)
                except Exception as exc2:
                    errors.append(f"Fallback index error: {str(exc2)}")
        else:
            try:
                vs.add_documents(docs)
            except Exception as exc:
                errors.append(f"Chroma index error: {str(exc)}")

        return errors


# ─── Factory function ─────────────────────────────────────────────────────────

def get_ingestion_pipeline(**kwargs) -> IngestionPipeline:
    """
    Factory para IngestionPipeline.

    Args kwargs: chunk_size, chunk_overlap, cleaner_profile, collection_name.

    Returns:
        IngestionPipeline configurado.
    """
    return IngestionPipeline(**kwargs)
