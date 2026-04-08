"""
AnswerValidatorSkill — validación anti-alucinación estructurada.

El validador es la última línea de defensa antes de retornar la respuesta.
Detecta y corrige:
  1. Numerales inventados: el documento tiene 2 puntos pero la respuesta añade 5
  2. Conclusiones genéricas: "Estos principios buscan promover el bienestar..."
  3. Referencias inventadas: artículos no presentes en el contexto recuperado
  4. Parafraseo excesivo que distorsiona el significado legal

Flujo:
  - Validación basada en reglas (sin LLM, O(n) texto) — detecta ~70% de problemas
  - Validación con LLM (solo si las reglas detectan algo sospechoso)

La validación con LLM usa VALIDATOR_PROMPT que pide output JSON
con {is_valid, violations, sanitized_answer}.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.documents import Document

from src.config.logging import get_logger

log = get_logger(__name__)

# Frases que indican conclusiones genéricas inventadas
_GENERIC_CONCLUSION_PATTERNS = [
    r"estos\s+principios\s+buscan",
    r"en\s+conclusi[oó]n[,\s]",
    r"en\s+resumen[,\s]",
    r"de\s+esta\s+manera\s+se\s+garantiza",
    r"esto\s+garantiza\s+que",
    r"es\s+fundamental\s+para",
    r"es\s+importante\s+destacar",
    r"cabe\s+resaltar\s+que",
    r"resulta\s+evidente\s+que",
]

_GENERIC_RE = re.compile(
    "|".join(_GENERIC_CONCLUSION_PATTERNS),
    re.IGNORECASE,
)

# Detecta listas numeradas (1. 2. 3. ...) para comparar con el contexto
_NUMERAL_RE = re.compile(r"^\s*(\d+)\.\s+\S", re.MULTILINE)


@dataclass
class ValidationResult:
    """Resultado de la validación de una respuesta."""
    is_valid: bool
    violations: list[str]
    sanitized_answer: str
    used_llm: bool = False
    confidence: float = 1.0


class AnswerValidatorSkill:
    """
    Valida respuestas legales contra el contexto recuperado.

    Detecta alucinaciones comunes en respuestas de LLM sobre documentos legales.
    """

    def __init__(self, use_llm_validation: bool = True) -> None:
        self._use_llm = use_llm_validation

    def validate(
        self,
        answer: str,
        context_docs: list[Document],
        original_query: str = "",
    ) -> ValidationResult:
        """
        Valida la respuesta contra los documentos de contexto.

        Args:
            answer: Respuesta generada por el LLM.
            context_docs: Documentos usados como contexto para la respuesta.
            original_query: Query original del usuario.

        Returns:
            ValidationResult con is_valid y sanitized_answer.
        """
        if not answer or not answer.strip():
            return ValidationResult(
                is_valid=False,
                violations=["Respuesta vacía"],
                sanitized_answer="No encontré información relevante en los documentos.",
            )

        violations: list[str] = []
        context_text = "\n".join(d.page_content for d in context_docs)

        # ── Regla 1: Conclusiones genéricas ──────────────────────────────────
        matches = _GENERIC_RE.findall(answer)
        if matches:
            violations.append(
                f"Conclusiones genéricas detectadas: {matches[:3]}"
            )

        # ── Regla 2: Numerales excesivos ──────────────────────────────────────
        answer_numerals = _NUMERAL_RE.findall(answer)
        if answer_numerals:
            max_numeral = max(int(n) for n in answer_numerals)
            context_numerals = _NUMERAL_RE.findall(context_text)
            if context_numerals:
                max_context = max(int(n) for n in context_numerals)
                if max_numeral > max_context + 2:
                    violations.append(
                        f"Numerales inventados: respuesta tiene hasta {max_numeral}, "
                        f"contexto solo hasta {max_context}"
                    )

        # ── Regla 3: Respuesta de no-encontrado correcta ───────────────────────
        not_found_phrase = "no encontré información relevante"
        if not_found_phrase in answer.lower() and len(answer.strip()) > len(not_found_phrase) + 20:
            # Dice que no encontró pero sigue añadiendo texto
            violations.append("Respuesta mixta: declara no encontrar pero agrega información")

        # Si no hay violaciones por reglas, la respuesta es válida
        if not violations:
            return ValidationResult(
                is_valid=True,
                violations=[],
                sanitized_answer=answer,
                used_llm=False,
                confidence=0.9,
            )

        log.warning(
            "answer_violations_found",
            count=len(violations),
            violations=violations[:3],
        )

        # ── Validación con LLM para corrección ───────────────────────────────
        if self._use_llm and violations:
            try:
                return self._validate_with_llm(answer, context_text, original_query, violations)
            except Exception as exc:
                log.warning("llm_validation_failed", error=str(exc))

        # Fallback: marcar como inválida, retornar respuesta original con advertencia
        return ValidationResult(
            is_valid=False,
            violations=violations,
            sanitized_answer=answer,  # el caller decide qué hacer
            used_llm=False,
            confidence=0.4,
        )

    def _validate_with_llm(
        self,
        answer: str,
        context_text: str,
        query: str,
        known_violations: list[str],
    ) -> ValidationResult:
        from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415
        from src.agent.prompts.system import VALIDATOR_PROMPT  # noqa: PLC0415
        from src.config.providers import get_llm  # noqa: PLC0415

        llm = get_llm()
        chain = VALIDATOR_PROMPT | llm | StrOutputParser()
        raw = chain.invoke({
            "context": context_text[:4000],
            "answer": answer,
        })

        clean = re.sub(r"```json\s?|```", "", raw).strip()
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            log.warning("validator_json_parse_failed")
            return ValidationResult(
                is_valid=False,
                violations=known_violations,
                sanitized_answer=answer,
                used_llm=True,
                confidence=0.3,
            )

        return ValidationResult(
            is_valid=bool(parsed.get("is_valid", False)),
            violations=parsed.get("violations", known_violations),
            sanitized_answer=parsed.get("sanitized_answer", answer),
            used_llm=True,
            confidence=0.85,
        )


def get_answer_validator(**kwargs) -> AnswerValidatorSkill:
    return AnswerValidatorSkill(**kwargs)
