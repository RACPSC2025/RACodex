"""
Generation node — Re2 condicional basado en CRAG grade_score.
"""

from __future__ import annotations

from src.agent.metrics import node_timer
from src.agent.state import AgentState
from src.config.logging import get_logger

log = get_logger(__name__)


def generation_node(state: AgentState) -> dict:
    """
    Genera respuesta con Re2 condicional basado en CRAG grade_score.

    - grade_score > 0.8: Generación directa (1 llamada LLM, bajo costo)
    - grade_score 0.5-0.8: Re2 dos pasadas (2 llamadas LLM, mayor precisión)
    - grade_score < 0.5: Re2 + advertencia (docs marginales, máxima precaución)
    """
    from src.agent.skills.rethinking import generate_direct, generate_with_rethinking  # noqa: PLC0415

    with node_timer(state, "generation") as timer:
        query = state.get("active_query") or state.get("user_query", "")
        docs = state.get("retrieval_results", [])
        grade_score = state.get("grade_score", 0.0)

        if not docs:
            timer.update(extra={"generation_mode": "no_docs", "grade_score": grade_score})
            return {
                "draft_answer": "No encontré documentos relevantes para responder tu consulta.",
                "sources": [],
                "generation_mode": "no_docs",
                **timer.to_state(),
            }

        # Decidir modo de generación basado en CRAG grade_score
        if grade_score > 0.8:
            # Documentos claramente relevantes → generación directa (1 llamada LLM)
            answer, sources = generate_direct(query, docs)
            generation_mode = "direct"
            llm_calls = 1

            timer.update(docs_count=len(sources), extra={
                "generation_mode": generation_mode,
                "grade_score": grade_score,
                "llm_calls": llm_calls,
            })

        elif grade_score >= 0.5:
            # Documentos parcialmente relevantes → Re2 dos pasadas (2 llamadas LLM)
            answer, sources = generate_with_rethinking(query, docs)
            generation_mode = "rethinking"
            llm_calls = 2

            timer.update(docs_count=len(sources), extra={
                "generation_mode": generation_mode,
                "grade_score": grade_score,
                "llm_calls": llm_calls,
            })

        else:
            # Documentos marginales → Re2 + advertencia
            answer, sources = generate_with_rethinking(query, docs)
            answer = (
                f"{answer}\n\n"
                f"⚠️ *Nota: Los documentos recuperados tienen relevancia limitada "
                f"(score: {grade_score:.2f}). Verifica la información con fuentes adicionales.*"
            )
            generation_mode = "rethinking_low_confidence"
            llm_calls = 2

            timer.update(docs_count=len(sources), extra={
                "generation_mode": generation_mode,
                "grade_score": grade_score,
                "llm_calls": llm_calls,
            })

        return {
            "draft_answer": answer,
            "sources": sources,
            "generation_mode": generation_mode,
            **timer.to_state(),
        }
