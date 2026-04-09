"""
Adaptive Chunking — Separadores específicos por tipo de documento.

Cada tipo de documento tiene su propia estructura y separadores naturales:
  - Decretos: "ARTÍCULO", "CAPÍTULO", "SECCIÓN", "PARÁGRAFO"
  - Contratos: "CLÁUSULA", "PARÁGRAFO", "OTROSÍ", "ANEXO"
  - Manuales técnicos: "##", "###", "1.", "1.1", "Nota:"
  - Políticas internas: "POLÍTICA", "PROCEDIMIENTO", "RESPONSABLE"

El chunker detecta el tipo de documento y aplica los separadores
correctos, mejorando significativamente la calidad de los chunks.

Uso:
    from src.ingestion.processors.adaptive_chunker import AdaptiveChunker

    chunker = AdaptiveChunker.detect_and_chunk(docs)
    # Detecta automáticamente el tipo y aplica separadores correctos
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.logging import get_logger
from src.config.settings import get_settings

log = get_logger(__name__)


# ─── Separadores por tipo de documento ────────────────────────────────────────

DOCUMENT_SEPARATORS: dict[str, list[str]] = {
    "decree": [
        "\n\nARTÍCULO",
        "\n\nArtículo",
        "\n\nCAPÍTULO",
        "\n\nSECCIÓN",
        "\n\nPARÁGRAFO",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    "contract": [
        "\n\nCLÁUSULA",
        "\n\nCláusula",
        "\n\nPARÁGRAFO",
        "\n\nOTROSÍ",
        "\n\nANEXO",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    "manual": [
        "\n## ",
        "\n### ",
        "\n#### ",
        "\n1.",
        "\n2.",
        "\nNota:",
        "\n\n",
        "\n",
        " ",
        "",
    ],
    "policy": [
        "\n\nPOLÍTICA",
        "\n\nPROCEDIMIENTO",
        "\n\nRESPONSABLE",
        "\n\nALCANCE",
        "\n\n",
        "\n",
        " ",
        "",
    ],
}


@dataclass
class DocumentTypeResult:
    """Resultado de la detección de tipo de documento."""

    doc_type: str  # "decree" | "contract" | "manual" | "policy" | "unknown"
    confidence: float
    indicators: list[str]


# ─── Detector de tipo ────────────────────────────────────────────────────────

def detect_document_type(text: str) -> DocumentTypeResult:
    """
    Detecta el tipo de documento basado en patrones de texto.

    Args:
        text: Contenido del documento (primeros 2000 chars suficientes).

    Returns:
        DocumentTypeResult con tipo, confianza e indicadores encontrados.
    """
    sample = text[:2000].lower()

    # Patrones por tipo
    patterns = {
        "decree": [r"art[íi]culo\s+\d", r"cap[íi]tulo\s+\w", r"par[áa]grafo", r"decreto\s+\d"],
        "contract": [r"cl[áa]usula\s+\w", r"otros[íi]", r"anexo\s+\w", r"partes\s+contratantes"],
        "manual": [r"^\s*##\s+\w", r"^\s*###\s+\w", r"^\s*\d+\.\s+\w", r"nota:"],
        "policy": [r"pol[íi]tica\s+\w", r"procedimiento\s+\w", r"responsable\s+\w", r"alcance\s+\w"],
    }

    scores: dict[str, int] = {}
    indicators: dict[str, list[str]] = {}

    for doc_type, type_patterns in patterns.items():
        matches = []
        for pattern in type_patterns:
            found = re.findall(pattern, sample, re.IGNORECASE | re.MULTILINE)
            if found:
                matches.extend(found[:3])  # Max 3 por patrón
        scores[doc_type] = len(matches)
        indicators[doc_type] = [m.strip() for m in matches]

    if not scores or max(scores.values()) == 0:
        return DocumentTypeResult(
            doc_type="unknown",
            confidence=0.0,
            indicators=[],
        )

    best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_type]
    total_score = sum(scores.values())

    # Confianza: proporción del score del mejor tipo vs total
    confidence = best_score / total_score if total_score > 0 else 0.0

    return DocumentTypeResult(
        doc_type=best_type,
        confidence=round(confidence, 4),
        indicators=indicators.get(best_type, []),
    )


# ─── Adaptive Chunker ────────────────────────────────────────────────────────

class AdaptiveChunker:
    """
    Chunker que detecta el tipo de documento y aplica separadores correctos.
    """

    @classmethod
    def chunk(cls, documents: list[Document], chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[Document]:
        """
        Chunking adaptativo para una lista de documentos.

        Cada documento se chunka con sus propios separadores según su tipo detectado.

        Args:
            documents: Documentos a chunkar.
            chunk_size: Override del tamaño de chunk.
            chunk_overlap: Override del overlap.

        Returns:
            Lista de documentos chunkeados.
        """
        settings = get_settings()
        size = chunk_size or settings.chunk_size
        overlap = chunk_overlap or settings.chunk_overlap

        all_chunks: list[Document] = []

        for doc in documents:
            doc_type_result = detect_document_type(doc.page_content)
            doc_type = doc_type_result.doc_type

            # Obtener separadores del tipo detectado
            separators = DOCUMENT_SEPARATORS.get(doc_type, DOCUMENT_SEPARATORS["decree"])

            # Crear splitter con separadores específicos
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                separators=separators,
                length_function=len,
                is_separator_regex=False,
            )

            # Chunkar documento
            chunks = splitter.split_text(doc.page_content)

            for i, chunk_text in enumerate(chunks):
                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "document_type": doc_type,
                        "chunk_index": i,
                        "chunking_confidence": doc_type_result.confidence,
                    },
                ))

            log.debug(
                "adaptive_chunking_complete",
                source=doc.metadata.get("source", "?"),
                doc_type=doc_type,
                chunks=len(chunks),
                confidence=doc_type_result.confidence,
            )

        log.info(
            "adaptive_chunking_all_complete",
            total_docs=len(documents),
            total_chunks=len(all_chunks),
        )

        return all_chunks

    @classmethod
    def detect_and_chunk(cls, documents: list[Document]) -> list[Document]:
        """Alias para `chunk()` — nombre más descriptivo."""
        return cls.chunk(documents)
