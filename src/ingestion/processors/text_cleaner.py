"""
Limpieza de texto pluggable por dominio.

Diseño:
  - `CleanerRule`: una transformación atómica (regex + función de reemplazo)
  - `TextCleaner`: aplica una lista ordenada de reglas sobre un texto
  - `CleanerProfile`: perfil nombrado con reglas preconfiguradas para un dominio
  - `CleanerRegistry`: registro de perfiles por nombre

Por qué pluggable:
  El código original tenía reglas del Decreto 1072 hardcodeadas en PyMuPDFLoader.
  Si mañana procesas contratos laborales o circulares de la SFC, necesitas
  reglas distintas sin tocar el loader. El perfil se elige en IngestionPipeline.

Perfiles incluidos:
  "default"        — limpieza básica aplicable a cualquier documento
  "decreto_1072"   — reglas específicas para el Decreto 1072 de 2015
  "legal_colombia" — normativa colombiana genérica (decretos, resoluciones, circulares)
  "contract"       — contratos privados (cláusulas, parágrafos)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from src.config.logging import get_logger

log = get_logger(__name__)


# ─── Regla atómica ────────────────────────────────────────────────────────────

@dataclass
class CleanerRule:
    """
    Una transformación de texto atómica.

    name:        identificador legible para debugging y logging
    pattern:     regex compilado (o None si usa transform directamente)
    replacement: string o callable(match) → str  (usado con re.sub)
    transform:   función alternativa text → str para lógica sin regex
    flags:       flags de re (re.IGNORECASE, re.MULTILINE, etc.)
    """
    name: str
    pattern: str | None = None
    replacement: str | Callable = ""
    transform: Callable[[str], str] | None = None
    flags: int = re.IGNORECASE | re.MULTILINE

    def apply(self, text: str) -> str:
        """Aplica la regla al texto. Retorna texto transformado."""
        if self.transform is not None:
            return self.transform(text)
        if self.pattern is not None:
            return re.sub(self.pattern, self.replacement, text, flags=self.flags)
        return text


# ─── Cleaner ──────────────────────────────────────────────────────────────────

class TextCleaner:
    """
    Aplica una pipeline de reglas de limpieza sobre texto.

    Las reglas se aplican en orden: el output de cada regla
    es el input de la siguiente.
    """

    def __init__(self, rules: list[CleanerRule], profile_name: str = "custom") -> None:
        self.rules = rules
        self.profile_name = profile_name

    def clean(self, text: str) -> str:
        """
        Aplica todas las reglas en orden sobre el texto.

        Args:
            text: Texto de entrada (puede tener ruido de extracción PDF).

        Returns:
            Texto limpiado. Nunca retorna None aunque la entrada sea vacía.
        """
        if not text or not text.strip():
            return ""

        result = text
        for rule in self.rules:
            try:
                result = rule.apply(result)
            except Exception as exc:
                log.warning(
                    "cleaner_rule_failed",
                    rule=rule.name,
                    profile=self.profile_name,
                    error=str(exc),
                )
                # Continúa con el texto sin transformar si una regla falla

        return result.strip()

    def clean_many(self, texts: list[str]) -> list[str]:
        """Aplica clean() a una lista de textos."""
        return [self.clean(t) for t in texts]

    def add_rule(self, rule: CleanerRule) -> "TextCleaner":
        """Agrega una regla al final de la pipeline (mutación, útil para extensión)."""
        self.rules.append(rule)
        return self

    def __repr__(self) -> str:
        return f"TextCleaner(profile={self.profile_name!r}, rules={len(self.rules)})"


# ─── Reglas reutilizables ─────────────────────────────────────────────────────

class Rules:
    """Catálogo de reglas atómicas reutilizables entre perfiles."""

    # ── Ruido de extracción PDF ────────────────────────────────────────────────

    REMOVE_PAGE_MARKERS = CleanerRule(
        name="remove_page_markers",
        pattern=r"\[Página\s*\d+\]",
        replacement="",
    )

    COLLAPSE_BLANK_LINES = CleanerRule(
        name="collapse_blank_lines",
        pattern=r"\n{4,}",
        replacement="\n\n\n",
    )

    COLLAPSE_SPACES = CleanerRule(
        name="collapse_spaces",
        pattern=r" {2,}",
        replacement=" ",
    )

    REMOVE_HORIZONTAL_RULES = CleanerRule(
        name="remove_horizontal_rules",
        pattern=r"[_\-]{4,}",
        replacement="",
    )

    REMOVE_NULL_BYTES = CleanerRule(
        name="remove_null_bytes",
        transform=lambda t: t.replace("\x00", "").replace("\ufffd", ""),
    )

    # ── Ruido específico de documentos colombianos ────────────────────────────

    REMOVE_DAFP_HEADER = CleanerRule(
        name="remove_dafp_header",
        pattern=r"Departamento Administrativo de la Función Pública\s*",
        replacement="",
    )

    REMOVE_EVA_FOOTER = CleanerRule(
        name="remove_eva_footer",
        pattern=r"Decreto\s+\d+\s+de\s+\d+\s+Sector\s+\w+\s*\d*\s*EVA\s*[-–]\s*Gestor\s+Normativo",
        replacement="",
    )

    REMOVE_PAGE_NUMBERS_STANDALONE = CleanerRule(
        name="remove_standalone_page_numbers",
        # Líneas que solo contienen un número (número de página)
        pattern=r"^\s*\d{1,4}\s*$",
        replacement="",
        flags=re.MULTILINE,
    )

    REMOVE_REPEATED_HEADERS = CleanerRule(
        name="remove_repeated_headers",
        # Encabezados de página repetidos en documentos multi-página
        pattern=(
            r"(República de Colombia|Ministerio del Trabajo|"
            r"Ministerio de Salud|Congreso de la República)\s*"
        ),
        replacement="",
    )

    # ── Normalización de estructura legal ─────────────────────────────────────

    NORMALIZE_ARTICLE_KEYWORD = CleanerRule(
        name="normalize_article_keyword",
        # Unifica variantes: ARTÍCULO / Artículo / ARTICULO / Articulo
        pattern=r"\bARTICULO\b",
        replacement="ARTÍCULO",
        flags=re.IGNORECASE,
    )

    NORMALIZE_PARAGRAPH_KEYWORD = CleanerRule(
        name="normalize_paragraph_keyword",
        pattern=r"\bPARAGRAFO\b",
        replacement="PARÁGRAFO",
        flags=re.IGNORECASE,
    )

    ENSURE_NEWLINE_BEFORE_ARTICLE = CleanerRule(
        name="ensure_newline_before_article",
        # Asegura que cada ARTÍCULO empiece en línea nueva
        pattern=r"(?<!\n)(ARTÍCULO\s+[\d\.]+)",
        replacement=r"\n\1",
        flags=re.IGNORECASE,
    )

    ENSURE_NEWLINE_BEFORE_CHAPTER = CleanerRule(
        name="ensure_newline_before_chapter",
        pattern=r"(?<!\n)(CAPÍTULO\s+[IVXivx\d]+)",
        replacement=r"\n\n\1",
        flags=re.IGNORECASE,
    )

    # ── Reconstrucción de párrafos rotos por extracción PDF ──────────────────

    JOIN_BROKEN_WORDS = CleanerRule(
        name="join_broken_words",
        # Palabras partidas al final de línea con guión: "cons-\ntitución" → "constitución"
        pattern=r"(\w)-\n(\w)",
        replacement=r"\1\2",
        flags=0,  # sin IGNORECASE para no combinar líneas de artículos
    )

    # NOTA: NO usamos un join genérico "(\w)\n(\w)" → "\1 \2"
    # porque rompe la estructura de artículos y listas numeradas.
    # Ese era el bug comentado en el código original.

    # ── Limpieza de OCR ───────────────────────────────────────────────────────

    OCR_FIX_COMMON_ERRORS = CleanerRule(
        name="ocr_fix_common_errors",
        transform=lambda t: (
            t
            .replace("l.000", "1.000")  # l vs 1
            .replace("O.000", "0.000")  # O vs 0
            .replace(" ,", ",")         # espacio antes de coma
            .replace(" .", ".")         # espacio antes de punto
        ),
    )


# ─── Perfiles predefinidos ────────────────────────────────────────────────────

class CleanerProfiles:
    """Perfiles de limpieza para diferentes tipos de documentos."""

    @staticmethod
    def default() -> TextCleaner:
        """Limpieza básica aplicable a cualquier documento."""
        return TextCleaner(
            profile_name="default",
            rules=[
                Rules.REMOVE_NULL_BYTES,
                Rules.REMOVE_PAGE_MARKERS,
                Rules.REMOVE_HORIZONTAL_RULES,
                Rules.COLLAPSE_SPACES,
                Rules.COLLAPSE_BLANK_LINES,
            ],
        )

    @staticmethod
    def legal_colombia() -> TextCleaner:
        """
        Normativa colombiana genérica.
        Aplica a decretos, resoluciones, circulares, leyes.
        """
        return TextCleaner(
            profile_name="legal_colombia",
            rules=[
                # 1. Eliminar ruido antes de normalizar
                Rules.REMOVE_NULL_BYTES,
                Rules.REMOVE_PAGE_MARKERS,
                Rules.REMOVE_HORIZONTAL_RULES,
                Rules.REMOVE_PAGE_NUMBERS_STANDALONE,
                Rules.REMOVE_DAFP_HEADER,
                Rules.REMOVE_EVA_FOOTER,
                Rules.REMOVE_REPEATED_HEADERS,
                # 2. Normalizar vocabulario legal
                Rules.NORMALIZE_ARTICLE_KEYWORD,
                Rules.NORMALIZE_PARAGRAPH_KEYWORD,
                # 3. Preservar estructura
                Rules.ENSURE_NEWLINE_BEFORE_ARTICLE,
                Rules.ENSURE_NEWLINE_BEFORE_CHAPTER,
                # 4. Reparar palabras rotas (solo guión al final de línea)
                Rules.JOIN_BROKEN_WORDS,
                # 5. Normalización final de espacios
                Rules.COLLAPSE_SPACES,
                Rules.COLLAPSE_BLANK_LINES,
            ],
        )

    @staticmethod
    def decreto_1072() -> TextCleaner:
        """
        Perfil específico para el Decreto 1072 de 2015.
        Extiende legal_colombia con reglas adicionales para ese documento.
        """
        cleaner = CleanerProfiles.legal_colombia()
        cleaner.profile_name = "decreto_1072"

        # Reglas adicionales específicas del 1072
        cleaner.add_rule(CleanerRule(
            name="remove_decreto_1072_watermark",
            pattern=r"Decreto\s+1072\s+de\s+2015\s+Sector\s+Trabajo\s*",
            replacement="",
        ))
        return cleaner

    @staticmethod
    def contract() -> TextCleaner:
        """Contratos privados (cláusulas, parágrafos, otrosíes)."""
        return TextCleaner(
            profile_name="contract",
            rules=[
                Rules.REMOVE_NULL_BYTES,
                Rules.REMOVE_PAGE_MARKERS,
                Rules.REMOVE_HORIZONTAL_RULES,
                Rules.REMOVE_PAGE_NUMBERS_STANDALONE,
                Rules.JOIN_BROKEN_WORDS,
                CleanerRule(
                    name="normalize_clause_keyword",
                    pattern=r"\bCLAUSULA\b",
                    replacement="CLÁUSULA",
                    flags=re.IGNORECASE,
                ),
                CleanerRule(
                    name="ensure_newline_before_clause",
                    pattern=r"(?<!\n)(CLÁUSULA\s+[A-ZÁÉÍÓÚ\w]+)",
                    replacement=r"\n\n\1",
                    flags=re.IGNORECASE,
                ),
                Rules.NORMALIZE_PARAGRAPH_KEYWORD,
                Rules.COLLAPSE_SPACES,
                Rules.COLLAPSE_BLANK_LINES,
            ],
        )

    @staticmethod
    def ocr_output() -> TextCleaner:
        """Post-procesamiento de texto extraído por OCR."""
        return TextCleaner(
            profile_name="ocr_output",
            rules=[
                Rules.REMOVE_NULL_BYTES,
                Rules.OCR_FIX_COMMON_ERRORS,
                Rules.REMOVE_PAGE_MARKERS,
                Rules.REMOVE_HORIZONTAL_RULES,
                Rules.REMOVE_PAGE_NUMBERS_STANDALONE,
                Rules.JOIN_BROKEN_WORDS,
                Rules.NORMALIZE_ARTICLE_KEYWORD,
                Rules.NORMALIZE_PARAGRAPH_KEYWORD,
                Rules.ENSURE_NEWLINE_BEFORE_ARTICLE,
                Rules.COLLAPSE_SPACES,
                Rules.COLLAPSE_BLANK_LINES,
            ],
        )


# ─── Registry de perfiles ─────────────────────────────────────────────────────

class CleanerRegistry:
    """Registro de perfiles de limpieza por nombre."""

    _profiles: dict[str, Callable[[], TextCleaner]] = {
        "default": CleanerProfiles.default,
        "legal_colombia": CleanerProfiles.legal_colombia,
        "decreto_1072": CleanerProfiles.decreto_1072,
        "contract": CleanerProfiles.contract,
        "ocr_output": CleanerProfiles.ocr_output,
    }

    @classmethod
    def get(cls, profile_name: str) -> TextCleaner:
        """
        Obtiene una instancia fresca del perfil solicitado.

        Args:
            profile_name: Nombre del perfil. Ver CleanerProfiles.

        Returns:
            TextCleaner configurado con las reglas del perfil.

        Raises:
            KeyError: Si el perfil no existe.
        """
        if profile_name not in cls._profiles:
            available = list(cls._profiles.keys())
            raise KeyError(
                f"Perfil de limpieza desconocido: {profile_name!r}. "
                f"Disponibles: {available}"
            )
        return cls._profiles[profile_name]()

    @classmethod
    def register(cls, name: str, factory: Callable[[], TextCleaner]) -> None:
        """Registra un perfil personalizado."""
        cls._profiles[name] = factory
        log.debug("cleaner_profile_registered", name=name)

    @classmethod
    def available_profiles(cls) -> list[str]:
        return list(cls._profiles.keys())


def get_cleaner(profile: str = "legal_colombia") -> TextCleaner:
    """Función de conveniencia: obtiene un cleaner por nombre de perfil."""
    return CleanerRegistry.get(profile)
