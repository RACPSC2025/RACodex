"""
CRAG — Corrective RAG.

Evalúa la calidad de los documentos recuperados y decide:
  - CORRECT (>0.7): documentos relevantes → generar respuesta directamente
  - AMBIGUOUS (0.3-0.7): parcialmente relevantes → re-retrieve con query reformulada
  - INCORRECT (<0.3): no relevantes → descartar y re-retrieve con step-back query

Sin dependencia de búsqueda web: cuando los docs son insuficientes,
se reescribe la query (rewriting o step-back) y se reintenta el retrieval.

Uso en el grafo:
    from src.agent.skills.crag import grade_documents, route_after_grading

    # En el nodo del grafo:
    def grade_node(state: AgentState) -> dict:
        return grade_documents(state)

    # Edge condicional:
    builder.add_conditional_edges(
        "grade",
        route_after_grading,
        {"correct": "generation", "ambiguous": "retrieval", "incorrect": "retrieval"},
    )
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from src.config.logging import get_logger
from src.config.providers import get_llm

log = get_logger(__name__)


# ─── Structured output para el grader ────────────────────────────────────────

class DocumentGrade(BaseModel):
    """Resultado del grading de documentos recuperados."""

    quality: str = Field(
        description="correct | ambiguous | incorrect"
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevancia general de los documentos (0-1).",
    )
    reasoning: str = Field(description="Breve justificación del score.")


# ─── Prompts ─────────────────────────────────────────────────────────────────

GRADER_SYSTEM = (
    "Eres un evaluador experto de documentos para sistemas RAG.\n"
    "Tu tarea es determinar si los documentos recuperados son suficientes "
    "para responder la consulta del usuario.\n\n"
    "Criterios:\n"
    "- **correct** (score > 0.7): documentos directamente relevantes y completos.\n"
    "- **ambiguous** (score 0.3-0.7): parcialmente relevantes, información incompleta.\n"
    "- **incorrect** (score < 0.3): no relevantes o fuera de contexto.\n\n"
    "Retorna SOLO el JSON con quality, score y reasoning."
)


# ─── Función principal ───────────────────────────────────────────────────────

def grade_documents(
    query: str,
    documents: list,
) -> DocumentGrade:
    """
    Evalúa la calidad de los documentos recuperados.

    Args:
        query: Consulta original del usuario.
        documents: Lista de Documents recuperados.

    Returns:
        DocumentGrade con quality (correct/ambiguous/incorrect), score y reasoning.
    """
    if not documents:
        return DocumentGrade(
            quality="incorrect",
            score=0.0,
            reasoning="No se recuperaron documentos.",
        )

    # Construir contexto compacto de los docs
    docs_summary = "\n---\n".join(
        f"[Doc {i + 1}] (source: {doc.metadata.get('source', '?')}, "
        f"chunk: {doc.metadata.get('chunk_index', '?')})\n"
        f"{doc.page_content[:500]}"
        for i, doc in enumerate(documents[:5])  # Max 5 docs para grading
    )

    llm = get_llm(temperature=0)
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", GRADER_SYSTEM),
        ("human", "{input}"),
    ])
    chain = grader_prompt | llm.with_structured_output(DocumentGrade)

    try:
        grade: DocumentGrade = chain.invoke({
            "input": f"Consulta: {query}\n\nDocumentos recuperados:\n{docs_summary}"
        })
    except Exception as exc:
        log.warning("crag_grading_failed", error=str(exc))
        # Fallback: ambiguous para reintentar
        return DocumentGrade(
            quality="ambiguous",
            score=0.5,
            reasoning=f"Error en grading: {exc}. Reintentando.",
        )

    log.info(
        "crag_grade_complete",
        quality=grade.quality,
        score=grade.score,
        docs_count=len(documents),
    )

    return grade


# ─── Query rewriting para re-retrieval ───────────────────────────────────────

def rewrite_query_for_reretrieval(
    query: str,
    grade: DocumentGrade,
    documents: list | None = None,
) -> str:
    """
    Reescribe la query para un nuevo intento de retrieval.

    - Si ambiguous: reformula con más detalle técnico.
    - Si incorrect: genera step-back query (más general).

    Args:
        query: Consulta original.
        grade: Resultado del grading.
        documents: Docs recuperados (para contexto del grader).

    Returns:
        Query reformulada para re-retrieval.
    """
    llm = get_llm(temperature=0)

    if grade.quality == "ambiguous":
        prompt = (
            "Los documentos recuperados son parcialmente relevantes. "
            "Reformula la siguiente consulta para ser más específica y técnica, "
            "de modo que el sistema de retrieval pueda encontrar documentos más precisos.\n\n"
            f"Consulta original: {query}\n\n"
            "Consulta reformulada (SOLO la consulta, sin explicaciones):"
        )
    else:
        # incorrect → step-back: query más general
        prompt = (
            "Los documentos recuperados no son relevantes. "
            "Genera una consulta más amplia y general sobre el tema de fondo "
            "para obtener contexto adicional.\n\n"
            f"Consulta específica: {query}\n\n"
            "Consulta general (SOLO la consulta, sin explicaciones):"
        )

    try:
        response = llm.invoke(prompt)
        result = response.content.strip()
        log.debug(
            "crag_query_rewritten",
            original=query[:60],
            rewritten=result[:60],
            quality=grade.quality,
        )
        return result
    except Exception as exc:
        log.warning("crag_rewrite_failed", error=str(exc))
        return query  # Fallback: query original


# ─── Helper para integración directa en nodos del grafo ──────────────────────

def grade_documents_node(state: dict) -> dict:
    """
    Nodo de grading para el grafo LangGraph.

    Args:
        state: AgentState con user_query, retrieval_results.

    Returns:
        Dict con doc_quality, grade_score, active_query (si rewrite), route.
    """
    query = state.get("active_query") or state.get("user_query", "")
    docs = state.get("retrieval_results", [])

    grade = grade_documents(query, docs)

    result: dict[str, object] = {
        "doc_quality": grade.quality,
        "grade_score": grade.score,
    }

    if grade.quality == "correct":
        result["route"] = "generation"
    else:
        # Ambiguous o incorrect → reescribir query y re-retrieval
        rewritten = rewrite_query_for_reretrieval(query, grade, docs)
        result["active_query"] = rewritten
        result["route"] = "retrieval"

    return result


def route_after_grading(state: dict) -> str:
    """
    Edge condicional para el grafo LangGraph.

    Args:
        state: AgentState con doc_quality y route.

    Returns:
        "generation" | "retrieval"
    """
    return state.get("route", "generation")
