"""
Extractor de metadatos estructurados desde texto legal colombiano.

Enriquece cada chunk con metadatos extraídos por regex antes de indexarlo
en Chroma. Metadatos ricos mejoran la retrieval filtrada y la trazabilidad.

Metadatos que extrae:
  article_number   — "2.2.4.6.1", "15", "ÚNICO"
  article_title    — texto después del número hasta el primer punto
  chapter          — número/nombre de capítulo
  section          — número/nombre de sección
  paragraph        — número de parágrafo si hay PARÁGRAFO N en el chunk
  page             — número de página desde el marcador [Página N]
  document_type    — decreto / resolución / circular / ley / contrato
  year             — año del documento si aparece en el encabezado
  source_name      — nombre del archivo sin extensión
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


# ─── Estructura de metadatos ──────────────────────────────────────────────────

@dataclass
class LegalMetadata:
    """Metadatos estructurados extraídos de un chunk legal."""

    source: str = ""                # nombre del archivo
    path: str = ""                  # ruta completa
    loader: str = ""                # loader que produjo el chunk
    chunk_index: int = 0
    chunk_size: int = 0

    # Estructura del documento
    page: str = ""                  # número de página como string (Chroma no acepta int mixto)
    article_number: str = ""        # "2.2.4.6.1" | "15" | "ÚNICO"
    article_title: str = ""         # primer enunciado del artículo
    chapter: str = ""               # "I" | "1" | "GENERALIDADES"
    section: str = ""
    paragraph: str = ""             # "1" | "2" si hay PARÁGRAFO en el chunk

    # Clasificación del documento
    document_type: str = ""         # decreto | resolución | circular | ley | contrato
    document_number: str = ""       # "1072" en "Decreto 1072 de 2015"
    document_year: str = ""         # "2015"

    def to_dict(self) -> dict[str, str | int]:
        """
        Convierte a dict compatible con Chroma metadata.

        Chroma solo acepta str, int, float y bool en metadata.
        Convertimos todo a string para uniformidad.
        """
        raw = asdict(self)
        return {k: str(v) if v is not None else "" for k, v in raw.items()}


# ─── Extractor ────────────────────────────────────────────────────────────────

class MetadataExtractor:
    """
    Extrae metadatos estructurados desde texto de chunks legales.

    Usa exclusivamente regex — sin LLM. Es síncrono y determinista.
    Los metadatos enriquecidos mejoran la retrieval filtrada en Chroma.
    """

    # ── Patrones compilados (compilar una vez, reutilizar N veces) ────────────

    _RE_PAGE = re.compile(r"\[Página\s*(\d+)\]", re.IGNORECASE)

    _RE_ARTICLE = re.compile(
        r"ARTÍCULO\s+([\d\.]+(?:\s*°)?|ÚNICO|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO)",
        re.IGNORECASE,
    )

    _RE_ARTICLE_TITLE = re.compile(
        r"ARTÍCULO\s+[\d\.]+[°\s]+[\.–\-]?\s*([^\n\.]{5,80})",
        re.IGNORECASE,
    )

    _RE_CHAPTER = re.compile(
        r"CAPÍTULO\s+([IVXivx\d]+(?:\s*[–\-]\s*[A-ZÁÉÍÓÚ][^\n]{0,60})?)",
        re.IGNORECASE,
    )

    _RE_SECTION = re.compile(
        r"SECCIÓN\s+([\d]+(?:\s*[–\-]\s*[A-ZÁÉÍÓÚ][^\n]{0,60})?)",
        re.IGNORECASE,
    )

    _RE_PARAGRAPH = re.compile(
        r"PARÁGRAFO\s*(\d*)",
        re.IGNORECASE,
    )

    _RE_DECREE = re.compile(
        r"DECRETO\s+(?:NÚMERO\s+)?(\d+)\s+(?:DEL?\s+)?(?:DE\s+)?(\d{4})",
        re.IGNORECASE,
    )

    _RE_RESOLUTION = re.compile(
        r"RESOLUCIÓN\s+(?:NÚMERO\s+)?(\d+)\s+(?:DEL?\s+)?(?:DE\s+)?(\d{4})",
        re.IGNORECASE,
    )

    _RE_LAW = re.compile(
        r"LEY\s+(\d+)\s+(?:DEL?\s+)?(?:DE\s+)?(\d{4})",
        re.IGNORECASE,
    )

    _RE_CIRCULAR = re.compile(
        r"CIRCULAR\s+(?:EXTERNA\s+)?(\d+)\s+(?:DEL?\s+)?(?:DE\s+)?(\d{4})",
        re.IGNORECASE,
    )

    def extract(
        self,
        chunk_text: str,
        source_path: Path | None = None,
        chunk_index: int = 0,
        loader_type: str = "",
    ) -> LegalMetadata:
        """
        Extrae metadatos estructurados de un chunk de texto.

        Args:
            chunk_text: Texto del chunk (puede incluir header contextual).
            source_path: Ruta al archivo original (para source y path).
            chunk_index: Índice del chunk dentro del documento.
            loader_type: Identificador del loader que produjo el chunk.

        Returns:
            LegalMetadata con todos los campos extraídos que se encontraron.
        """
        meta = LegalMetadata(
            chunk_index=chunk_index,
            chunk_size=len(chunk_text),
            loader=loader_type,
        )

        if source_path is not None:
            meta.source = source_path.name
            meta.path = str(source_path)

        # Extraer cada campo en orden de especificidad
        meta.page = self._extract_page(chunk_text)
        meta.article_number = self._extract_article_number(chunk_text)
        meta.article_title = self._extract_article_title(chunk_text)
        meta.chapter = self._extract_chapter(chunk_text)
        meta.section = self._extract_section(chunk_text)
        meta.paragraph = self._extract_paragraph(chunk_text)

        doc_type, doc_number, doc_year = self._extract_document_info(chunk_text)
        meta.document_type = doc_type
        meta.document_number = doc_number
        meta.document_year = doc_year

        return meta

    # ── Extractores individuales ──────────────────────────────────────────────

    def _extract_page(self, text: str) -> str:
        match = self._RE_PAGE.search(text)
        return match.group(1) if match else ""

    def _extract_article_number(self, text: str) -> str:
        match = self._RE_ARTICLE.search(text)
        if not match:
            return ""
        return match.group(1).strip().rstrip("°").strip()

    def _extract_article_title(self, text: str) -> str:
        match = self._RE_ARTICLE_TITLE.search(text)
        if not match:
            return ""
        title = match.group(1).strip()
        # Limpiar puntuación residual al inicio
        title = re.sub(r"^[\.–\-\s]+", "", title)
        return title[:100]  # truncar si es muy largo

    def _extract_chapter(self, text: str) -> str:
        match = self._RE_CHAPTER.search(text)
        return match.group(1).strip() if match else ""

    def _extract_section(self, text: str) -> str:
        match = self._RE_SECTION.search(text)
        return match.group(1).strip() if match else ""

    def _extract_paragraph(self, text: str) -> str:
        match = self._RE_PARAGRAPH.search(text)
        if not match:
            return ""
        number = match.group(1).strip()
        return number if number else "1"

    def _extract_document_info(self, text: str) -> tuple[str, str, str]:
        """Retorna (document_type, document_number, document_year)."""
        for pattern, doc_type in [
            (self._RE_DECREE,     "decreto"),
            (self._RE_RESOLUTION, "resolución"),
            (self._RE_LAW,        "ley"),
            (self._RE_CIRCULAR,   "circular"),
        ]:
            match = pattern.search(text)
            if match:
                return doc_type, match.group(1), match.group(2)
        return "", "", ""

    def build_contextual_header(self, meta: LegalMetadata) -> str:
        """
        Construye el header contextual que se inyecta al inicio del chunk.

        Este header mejora el embedding porque el modelo ve el contexto
        completo incluso en chunks cortos. Patrón recomendado por
        Anthropic y OpenAI para RAG de alta precisión.

        Formato:
          [Fuente: documentacion.pdf | Art. 5 | Página: 45]
        """
        parts: list[str] = []

        if meta.source:
            parts.append(f"Fuente: {meta.source}")
        if meta.article_number:
            parts.append(f"Art. {meta.article_number}")
        elif meta.chapter:
            parts.append(f"Cap. {meta.chapter}")
        if meta.page:
            parts.append(f"Página: {meta.page}")
        if meta.document_type and meta.document_year:
            parts.append(f"{meta.document_type.title()} {meta.document_year}")

        if not parts:
            return ""

        return f"[{' | '.join(parts)}]\n"


# ─── Instancia por defecto ────────────────────────────────────────────────────

_default_extractor: MetadataExtractor | None = None


def get_metadata_extractor() -> MetadataExtractor:
    global _default_extractor  # noqa: PLW0603
    if _default_extractor is None:
        _default_extractor = MetadataExtractor()
    return _default_extractor
