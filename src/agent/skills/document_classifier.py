"""
DocumentClassifierSkill — elige la estrategia de ingestion óptima.

Responsabilidad única: dado un archivo, producir un IngestionPlan
que el pipeline de ingestion ejecutará sin tomar más decisiones.

Estrategia dual:
  1. Clasificación basada en reglas (determinista, sin LLM, latencia ~0ms)
     — Cubre el 90% de los casos con alta precisión
  2. Clasificación con LLM (para casos ambiguos)
     — Solo se invoca cuando la confianza de reglas < umbral

Por qué reglas primero:
  - Sin latencia de LLM para casos claros (PDF nativo = PyMuPDF)
  - Sin costo de tokens para clasificaciones simples
  - Reproducible y debuggeable: las reglas son explícitas y testeables
  - El LLM se reserva para ambigüedad real

Casos que requieren LLM:
  - PDF con mix de páginas nativas y escaneadas
  - Documentos sin extensión clara
  - PDFs con calidad borderline (avg_chars cerca del umbral)
  - Documentos en idiomas mixtos
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.agent.state import IngestionPlan
from src.config.logging import get_logger
from src.config.providers import get_llm
from src.ingestion.detectors.mime_detector import MimeDetectionResult, detect_mime
from src.ingestion.detectors.quality_detector import PDFQualityResult, analyze_pdf_quality

log = get_logger(__name__)

# Umbral de confianza para escalar al LLM
_LLM_FALLBACK_THRESHOLD = 0.70


class DocumentClassifierSkill:
    """
    Clasifica documentos y produce un IngestionPlan.

    La clasificación es la primera operación del pipeline de ingestion.
    Un plan incorrecto (ej: usar PyMuPDF en un PDF escaneado) produce
    chunks vacíos que el retrieval no puede usar.
    """

    def __init__(
        self,
        llm_threshold: float = _LLM_FALLBACK_THRESHOLD,
        use_llm_fallback: bool = True,
    ) -> None:
        self._llm_threshold = llm_threshold
        self._use_llm_fallback = use_llm_fallback

    def classify(self, file_path: Path | str) -> IngestionPlan:
        """
        Clasifica un documento y retorna un IngestionPlan.

        Args:
            file_path: Ruta al documento a clasificar.

        Returns:
            IngestionPlan con loader_type, cleaner_profile y flags.

        Raises:
            FileNotFoundError: Si el archivo no existe.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        log.info("classifying_document", file=path.name)

        # ── 1. Detección de MIME ──────────────────────────────────────────────
        try:
            mime_result = detect_mime(path)
        except Exception as exc:
            log.warning("mime_detection_failed", file=path.name, error=str(exc))
            return self._fallback_plan(path, reason=f"MIME detection failed: {exc}")

        # ── 2. Calidad PDF (solo si es PDF) ───────────────────────────────────
        quality_result: PDFQualityResult | None = None
        if mime_result.is_pdf:
            try:
                quality_result = analyze_pdf_quality(path)
            except Exception as exc:
                log.warning("quality_detection_failed", file=path.name, error=str(exc))

        # ── 3. Clasificación por reglas ───────────────────────────────────────
        plan, confidence = self._classify_by_rules(path, mime_result, quality_result)

        log.info(
            "classification_by_rules",
            file=path.name,
            loader=plan["loader_type"],
            confidence=confidence,
        )

        # ── 4. Escalar a LLM si confianza es baja ────────────────────────────
        if confidence < self._llm_threshold and self._use_llm_fallback:
            log.info(
                "classification_escalating_to_llm",
                file=path.name,
                rule_confidence=confidence,
            )
            try:
                plan = self._classify_with_llm(path, mime_result, quality_result)
            except Exception as exc:
                log.warning(
                    "llm_classification_failed",
                    file=path.name,
                    error=str(exc),
                    fallback="using_rule_based_plan",
                )
                # Mantener el plan de reglas si el LLM falla

        log.info(
            "classification_complete",
            file=path.name,
            loader=plan["loader_type"],
            document_type=plan["document_type"],
            requires_ocr=plan["requires_ocr"],
        )

        return plan

    def classify_many(self, file_paths: list[Path | str]) -> list[IngestionPlan]:
        """Clasifica múltiples archivos. Los errores producen planes de fallback."""
        plans = []
        for path in file_paths:
            try:
                plan = self.classify(path)
            except Exception as exc:
                log.error("classify_many_item_failed", file=str(path), error=str(exc))
                plan = self._fallback_plan(Path(path), reason=str(exc))
            plans.append(plan)
        return plans

    # ── Clasificación por reglas ───────────────────────────────────────────────

    def _classify_by_rules(
        self,
        path: Path,
        mime: MimeDetectionResult,
        quality: PDFQualityResult | None,
    ) -> tuple[IngestionPlan, float]:
        """
        Clasifica usando reglas deterministas.

        Returns:
            (IngestionPlan, confidence_score)
        """

        # ── Word ──────────────────────────────────────────────────────────────
        if mime.is_word:
            doc_type = "contrato" if self._looks_like_contract(path) else "decreto"
            cleaner = "contract" if doc_type == "contrato" else "legal_colombia"
            return self._make_plan(
                path, mime,
                loader_type="word",
                cleaner_profile=cleaner,
                requires_ocr=False,
                document_type=doc_type,
                confidence=0.95,
                reasoning="Archivo Word detectado por MIME type",
            ), 0.95

        # ── Excel ─────────────────────────────────────────────────────────────
        if mime.is_excel:
            return self._make_plan(
                path, mime,
                loader_type="excel",
                cleaner_profile="default",
                requires_ocr=False,
                document_type="excel",
                confidence=0.97,
                reasoning="Archivo Excel detectado por MIME type",
            ), 0.97

        # ── Imagen directa ────────────────────────────────────────────────────
        if mime.is_image:
            return self._make_plan(
                path, mime,
                loader_type="ocr",
                cleaner_profile="ocr_output",
                requires_ocr=True,
                document_type="imagen",
                confidence=0.95,
                reasoning="Imagen directa — requiere OCR",
            ), 0.95

        # ── PDF ───────────────────────────────────────────────────────────────
        if mime.is_pdf:
            return self._classify_pdf_by_rules(path, mime, quality)

        # ── Desconocido ───────────────────────────────────────────────────────
        return self._fallback_plan(path, reason="MIME type no reconocido"), 0.0

    def _classify_pdf_by_rules(
        self,
        path: Path,
        mime: MimeDetectionResult,
        quality: PDFQualityResult | None,
    ) -> tuple[IngestionPlan, float]:
        """Clasificación específica para PDFs según calidad y características."""

        # Sin datos de calidad → asumir OCR (más seguro)
        if quality is None:
            return self._make_plan(
                path, mime,
                loader_type="ocr",
                cleaner_profile="ocr_output",
                requires_ocr=True,
                document_type="decreto",
                confidence=0.50,
                reasoning="No se pudo analizar calidad PDF — asumiendo escaneado",
            ), 0.50

        # PDF escaneado con alta confianza
        if quality.is_scanned and quality.confidence >= 0.80:
            return self._make_plan(
                path, mime,
                loader_type="ocr",
                cleaner_profile="ocr_output",
                requires_ocr=True,
                document_type=self._infer_doc_type_from_name(path),
                confidence=quality.confidence,
                reasoning=f"PDF escaneado (confidence={quality.confidence:.2f})",
            ), quality.confidence

        # PDF nativo con alta confianza
        if quality.is_native and quality.confidence >= 0.80:
            loader, cleaner, doc_type = self._select_native_pdf_strategy(path, quality)
            return self._make_plan(
                path, mime,
                loader_type=loader,
                cleaner_profile=cleaner,
                requires_ocr=False,
                document_type=doc_type,
                confidence=quality.confidence,
                reasoning=f"PDF nativo (confidence={quality.confidence:.2f}, pages={quality.total_pages})",
            ), quality.confidence

        # Confianza baja — escalar al LLM
        return self._make_plan(
            path, mime,
            loader_type="pymupdf",
            cleaner_profile="legal_colombia",
            requires_ocr=False,
            document_type=self._infer_doc_type_from_name(path),
            confidence=0.55,
            reasoning="Calidad ambigua — confianza baja, puede necesitar revisión",
        ), 0.55

    def _select_native_pdf_strategy(
        self, path: Path, quality: PDFQualityResult
    ) -> tuple[str, str, str]:
        """
        Elige entre PyMuPDF y Docling para PDFs nativos.

        Heurística: PDFs largos (>50 páginas) probablemente tienen tablas
        y layouts complejos → Docling. PDFs cortos → PyMuPDF (más rápido).
        """
        doc_type = self._infer_doc_type_from_name(path)

        # Decreto 1072 específico
        if "1072" in path.name.lower() and "decreto" in path.name.lower():
            return "pymupdf", "decreto_1072", "decreto"

        # PDFs con muchas páginas → posiblemente con tablas
        if quality.total_pages > 50:
            return "docling", "legal_colombia", doc_type

        # Corto y nativo → PyMuPDF
        cleaner = "contract" if doc_type == "contrato" else "legal_colombia"
        return "pymupdf", cleaner, doc_type

    # ── Clasificación con LLM ─────────────────────────────────────────────────

    def _classify_with_llm(
        self,
        path: Path,
        mime: MimeDetectionResult,
        quality: PDFQualityResult | None,
    ) -> IngestionPlan:
        """
        Usa el LLM para clasificar documentos ambiguos.

        Lee una muestra del contenido para darle contexto al LLM.
        """
        from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415
        from src.agent.prompts.system import CLASSIFIER_PROMPT  # noqa: PLC0415

        content_sample = self._read_sample(path, mime)
        quality_label = quality.quality_label if quality else "unknown"
        page_count = quality.total_pages if quality else 0
        file_size = path.stat().st_size

        llm = get_llm()
        chain = CLASSIFIER_PROMPT | llm | StrOutputParser()

        raw = chain.invoke({
            "filename": path.name,
            "mime_type": mime.mime_type,
            "pdf_quality": quality_label,
            "file_size": str(file_size),
            "page_count": str(page_count),
            "content_sample": content_sample,
        })

        parsed = self._parse_llm_json(raw)

        return self._make_plan(
            path, mime,
            loader_type=parsed.get("loader_type", "pymupdf"),
            cleaner_profile=parsed.get("cleaner_profile", "legal_colombia"),
            requires_ocr=bool(parsed.get("requires_ocr", False)),
            document_type=parsed.get("document_type", "decreto"),
            confidence=float(parsed.get("confidence", 0.75)),
            reasoning=parsed.get("reasoning", "Clasificado por LLM"),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_plan(
        self,
        path: Path,
        mime: MimeDetectionResult,
        loader_type: str,
        cleaner_profile: str,
        requires_ocr: bool,
        document_type: str,
        confidence: float,
        reasoning: str,
    ) -> IngestionPlan:
        return IngestionPlan(
            loader_type=loader_type,
            cleaner_profile=cleaner_profile,
            requires_ocr=requires_ocr,
            document_type=document_type,
            source_path=str(path),
            mime_type=mime.mime_type,
            confidence=round(confidence, 3),
            reasoning=reasoning,
        )

    def _fallback_plan(self, path: Path, reason: str = "") -> IngestionPlan:
        """Plan de fallback seguro cuando la clasificación falla."""
        return IngestionPlan(
            loader_type="pymupdf",
            cleaner_profile="default",
            requires_ocr=False,
            document_type="desconocido",
            source_path=str(path),
            mime_type="application/octet-stream",
            confidence=0.0,
            reasoning=f"Fallback: {reason}",
        )

    def _infer_doc_type_from_name(self, path: Path) -> str:
        """Infiere el tipo de documento desde el nombre del archivo."""
        name = path.stem.lower()
        if any(k in name for k in ["decreto", "dec_", "d_"]):
            return "decreto"
        if any(k in name for k in ["resolucion", "resol", "res_"]):
            return "resolución"
        if any(k in name for k in ["circular", "circ"]):
            return "circular"
        if any(k in name for k in ["contrato", "cont_", "acuerdo"]):
            return "contrato"
        if any(k in name for k in ["ley", "ley_"]):
            return "ley"
        return "decreto"  # default para normativa colombiana

    def _looks_like_contract(self, path: Path) -> bool:
        """Heurística simple: el nombre sugiere que es un contrato."""
        name = path.stem.lower()
        return any(k in name for k in ["contrato", "acuerdo", "convenio", "otrosi"])

    def _read_sample(self, path: Path, mime: MimeDetectionResult) -> str:
        """Lee una muestra del contenido para el clasificador LLM."""
        try:
            if mime.is_pdf:
                import fitz  # noqa: PLC0415
                doc = fitz.open(str(path))
                text = doc[0].get_text("text") if len(doc) > 0 else ""
                doc.close()
                return text[:500]
        except Exception:
            pass
        return ""

    def _parse_llm_json(self, raw: str) -> dict[str, Any]:
        """Parsea JSON del output del LLM con limpieza de markdown."""
        clean = re.sub(r"```json\s?|```", "", raw).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError as exc:
            log.warning("llm_json_parse_failed", raw=raw[:200], error=str(exc))
            return {}


# ─── Instancia singleton ──────────────────────────────────────────────────────

_default_classifier: DocumentClassifierSkill | None = None


def get_document_classifier(**kwargs) -> DocumentClassifierSkill:
    """Retorna la instancia singleton del classifier."""
    global _default_classifier  # noqa: PLW0603
    if kwargs:
        return DocumentClassifierSkill(**kwargs)
    if _default_classifier is None:
        _default_classifier = DocumentClassifierSkill()
    return _default_classifier
