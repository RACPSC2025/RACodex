"""
Estado central del agente LangGraph para Fénix RAG.

Principios de diseño:
  - `AgentState` es un TypedDict — LangGraph lo serializa/deserializa en cada
    paso del grafo. Todo campo debe ser JSON-serializable.
  - Los mensajes usan `Annotated[list, add_messages]` — el operador `add_messages`
    acumula sin reemplazar (patrón estándar de LangGraph para chat history).
  - Los campos no acumulativos usan el operador por defecto de reemplazo.
  - Separación clara entre estado de ingestion, retrieval, generación y reflexión.

Ciclo de vida del estado en el grafo:
  1. `__start__` → rellena `messages` con la query del usuario
  2. `document_router` → rellena `ingestion_plan`
  3. `ingestion_node` → rellena `ingested_documents`
  4. `retrieval_node` → rellena `retrieval_results`
  5. `generation_node` → rellena `draft_answer`
  6. `reflection_node` → rellena `reflection_score` y `reflection_feedback`
  7. Si score < umbral → vuelve a `retrieval_node` con `reformulated_query`
  8. Si score >= umbral → `__end__` con `final_answer`
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ─── Plan de ingestion ────────────────────────────────────────────────────────

class IngestionPlan(TypedDict):
    """
    Plan de procesamiento para un documento.

    Lo produce DocumentClassifierSkill antes de la ingestion.
    Define qué loader usar, qué cleaner aplicar y si se necesita OCR.
    """
    loader_type: str            # "pymupdf" | "ocr" | "docling" | "word" | "excel"
    cleaner_profile: str        # "legal_colombia" | "contract" | "ocr_output" | "default"
    requires_ocr: bool
    document_type: str          # "decreto" | "resolución" | "contrato" | "circular" | "excel"
    source_path: str            # path absoluto al archivo
    mime_type: str
    confidence: float           # confianza de la clasificación [0.0 - 1.0]
    reasoning: str              # explicación del classifier


# ─── Resultado de reflexión ───────────────────────────────────────────────────

class ReflectionOutput(TypedDict):
    """Resultado del nodo de reflexión sobre la respuesta generada."""
    score: float                # calidad [0.0 - 1.0]
    is_grounded: bool           # ¿la respuesta está fundamentada en los docs?
    has_hallucination: bool     # ¿hay información inventada?
    cites_source: bool          # ¿menciona el artículo/fuente?
    feedback: str               # qué mejorar en la próxima iteración
    reformulated_query: str     # query reformulada para re-retrieval (si aplica)


# ─── Estado principal del agente ──────────────────────────────────────────────

class AgentState(TypedDict):
    """
    Estado completo del agente Fénix RAG.

    Todos los campos son opcionales excepto `messages`.
    LangGraph reemplaza campos a menos que usen un operador acumulativo.

    Convención de nombres:
      `*_node`    — output de un nodo específico
      `*_plan`    — planes/decisiones tomadas por skills
      `*_results` — resultados de operaciones de retrieval
      `draft_*`   — respuesta en construcción (pre-reflexión)
      `final_*`   — respuesta aprobada post-reflexión
    """

    # ── Mensajes (acumulativos) ───────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Contexto de la sesión ─────────────────────────────────────────────────
    session_id: str
    user_query: str             # query original del usuario (sin modificar)
    active_query: str           # query actual (puede ser reformulada por reflection)

    # ── Documentos del usuario ────────────────────────────────────────────────
    uploaded_files: list[str]   # paths a archivos subidos por el usuario
    ingestion_plans: list[IngestionPlan]  # un plan por archivo
    ingested_documents: list[Document]   # chunks resultantes de la ingestion

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_results: list[Document]    # docs recuperados para responder
    retrieval_strategy: str              # estrategia usada ("hybrid", "full", etc.)

    # ── Generación ────────────────────────────────────────────────────────────
    draft_answer: str
    final_answer: str
    sources: list[dict[str, str]]        # [{source, article, page}]

    # ── Reflexión ─────────────────────────────────────────────────────────────
    reflection: ReflectionOutput | None
    iteration_count: int                 # número de intentos de generación
    max_iterations: int                  # límite de iteraciones (default: 2)

    # ── Metadata del grafo ────────────────────────────────────────────────────
    error: str | None                    # mensaje de error si algo falla
    route: str                           # próximo nodo ("END", "retrieval", etc.)


# ─── Estado inicial ───────────────────────────────────────────────────────────

def initial_state(
    user_query: str,
    session_id: str = "",
    uploaded_files: list[str] | None = None,
    max_iterations: int = 2,
) -> dict[str, Any]:
    """
    Construye el estado inicial para una nueva invocación del agente.

    Args:
        user_query: Pregunta del usuario.
        session_id: ID de sesión para trazabilidad.
        uploaded_files: Paths a archivos que el usuario quiere consultar.
        max_iterations: Máximo de ciclos reflection → re-retrieval.

    Returns:
        Dict compatible con AgentState para pasar a graph.invoke().
    """
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=user_query)],
        "session_id": session_id,
        "user_query": user_query,
        "active_query": user_query,
        "uploaded_files": uploaded_files or [],
        "ingestion_plans": [],
        "ingested_documents": [],
        "retrieval_results": [],
        "retrieval_strategy": "",
        "draft_answer": "",
        "final_answer": "",
        "sources": [],
        "reflection": None,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "error": None,
        "route": "",
    }
