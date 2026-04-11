"""
Reflection node — auto-evaluación de la respuesta generada.
"""

from __future__ import annotations

from src.agent.metrics import node_timer
from src.agent.state import AgentState, ReflectionOutput
from src.config.logging import get_logger

log = get_logger(__name__)


def reflection_node(state: AgentState) -> dict:
    """
    Auto-evaluación de la respuesta generada.

    Validación rule-based primero (sin costo de LLM).
    Si es válida → final_answer con route=END.
    Si no es válida y quedan iteraciones → reformular y re-retrieval.
    Si iteraciones agotadas → borrador con advertencia.
    """
    from src.agent.skills.answer_validator import AnswerValidatorSkill  # noqa: PLC0415
    from src.agent.skills.query_transformer import QueryTransformer  # noqa: PLC0415

    with node_timer(state, "reflection") as timer:
        draft = state.get("draft_answer", "")
        query = state.get("user_query", "")
        docs = state.get("retrieval_results", [])
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 2)

        # Validación rule-based primero (sin costo de LLM)
        validator = AnswerValidatorSkill()
        validation = validator.validate(draft, docs, query)

        if validation.is_valid:
            timer.update(extra={
                "validation_score": validation.confidence,
                "route": "END",
                "reason": "valid_response",
            })

            return {
                "final_answer": draft,
                "reflection": ReflectionOutput(
                    score=validation.confidence,
                    is_grounded=validation.is_valid,
                    has_hallucination=False,
                    cites_source=True,
                    feedback="Respuesta válida",
                    reformulated_query="",
                ),
                "route": "END",
                "iteration_count": iteration + 1,
                **timer.to_state(),
            }

        # Si no es válida y quedan iteraciones → reformular
        if iteration < max_iter:
            transformer = QueryTransformer()
            reformulated = transformer.rewrite(query)

            timer.update(extra={
                "validation_score": validation.confidence,
                "route": "retrieval",
                "reason": "invalid_response_retry",
                "violations": validation.violations[:3] if validation.violations else [],
            })

            return {
                "active_query": reformulated,
                "reflection": ReflectionOutput(
                    score=validation.confidence,
                    is_grounded=False,
                    has_hallucination=bool(validation.violations),
                    cites_source=False,
                    feedback=validation.violations[0] if validation.violations else "Respuesta inválida",
                    reformulated_query=reformulated,
                ),
                "route": "retrieval",
                "iteration_count": iteration + 1,
                **timer.to_state(),
            }

        # Iteraciones agotadas → usar borrador con advertencia
        timer.update(extra={
            "validation_score": validation.confidence,
            "violations": len(validation.violations),
            "route": "END",
            "reason": "iterations_exhausted",
        })

        return {
            "final_answer": draft + "\n\n⚠️ Nota: Esta respuesta puede estar incompleta.",
            "reflection": ReflectionOutput(
                score=validation.confidence,
                is_grounded=False,
                has_hallucination=False,
                cites_source=False,
                feedback="Iteraciones agotadas",
                reformulated_query="",
            ),
            "route": "END",
            "iteration_count": iteration + 1,
            **timer.to_state(),
        }
