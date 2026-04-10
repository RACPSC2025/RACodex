"""
Grafo LangGraph principal de Fénix RAG.

Arquitectura del grafo:
                    ┌──────────────────────┐
                    │      __start__       │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   document_router    │  ← clasifica archivos subidos
                    └──────┬──────┬────────┘
                   uploads │      │ sin uploads
                           │      │
              ┌────────────▼──┐   │
              │   ingestion   │   │
              └────────┬──────┘   │
                       │          │
              ┌────────▼──────────▼──┐
              │      retrieval       │  ← EnsembleRetriever (hybrid/full)
              └────────────┬─────────┘
                           │
              ┌────────────▼─────────┐
              │      generation      │  ← LLM + contexto
              └────────────┬─────────┘
                           │
              ┌────────────▼─────────┐
              │      reflection      │  ← Self-evaluation
              └──────┬───────┬───────┘
              score  │       │ score
              >= 0.8 │       │ < 0.8
                     │       │
           ┌─────────▼──┐  ┌─▼──────────────────┐
           │   __end__  │  │ retrieval (retry)   │
           └────────────┘  └────────────────────┘
                            (máx. max_iterations)

Herramientas disponibles para el agente ReAct (via ToolNode):
  - ingest_pdf, ingest_excel, ingest_word, ingest_image_pdf
  - semantic_search, hybrid_search, article_lookup
  - list_indexed_documents
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.nodes.all_nodes import (
    document_router_node,
    generation_node,
    ingestion_node,
    reflection_node,
    retrieval_node,
    supervisor_node,
)
from src.agent.skills.crag import grade_documents_node, route_after_grading
from src.agent.state import AgentState, initial_state
from src.agent.tools.ingest_tools import (
    ingest_excel,
    ingest_image_pdf,
    ingest_pdf,
    ingest_word,
    list_indexed_documents,
)
from src.agent.tools.memory_tools import (
    clear_context,
    list_context_keys,
    retrieve_context,
    save_context,
)
from src.agent.tools.search_tools import (
    article_lookup,
    hybrid_search,
    semantic_search,
)
from src.config.logging import get_logger

log = get_logger(__name__)

# ─── Todas las tools disponibles para el agente ReAct ─────────────────────────

ALL_TOOLS = [
    ingest_pdf,
    ingest_excel,
    ingest_word,
    ingest_image_pdf,
    list_indexed_documents,
    semantic_search,
    hybrid_search,
    article_lookup,
    save_context,
    retrieve_context,
    list_context_keys,
    clear_context,
]


# ─── Funciones de routing condicional ─────────────────────────────────────────

def route_after_router(
    state: AgentState,
) -> Literal["ingestion", "retrieval"]:
    """Después del document_router: ¿hay archivos para ingestar?"""
    route = state.get("route", "retrieval")
    return "ingestion" if route == "ingestion" else "retrieval"


def route_after_ingestion(
    state: AgentState,
) -> Literal["retrieval", "__end__"]:
    """Después de ingestion: ¿éxito o error fatal?"""
    if state.get("error") and not state.get("ingested_documents"):
        return END
    return "retrieval"


def route_after_reflection(
    state: AgentState,
) -> Literal["retrieval", "__end__"]:
    """
    Después de reflection: ¿respuesta aceptable o necesita re-retrieval?

    END   → respuesta aprobada (score >= 0.8 o iteraciones agotadas)
    retrieval → score bajo, reformulated_query disponible
    """
    route = state.get("route", "END")
    if route == "END":
        return END
    return "retrieval"


def route_after_generation(
    state: AgentState,
) -> Literal["tools", "reflection"]:
    """
    Después de generation en modo ReAct:
    Si el último mensaje del LLM contiene tool_calls → ToolNode.
    Si no → reflection.
    """
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
    return "reflection"


# ─── Constructor del grafo ────────────────────────────────────────────────────

def build_graph(
    with_tools: bool = True,
    checkpointer: Any | None = None,
) -> StateGraph:
    """
    Construye y compila el StateGraph principal de Fénix RAG.

    Args:
        with_tools: Si True, agrega ToolNode para el ciclo ReAct.
                    En producción siempre True. False para tests simples.
        checkpointer: Checkpointer de LangGraph para persistencia de estado
                      entre invocaciones (ej: SqliteSaver o PostgresSaver).
                      None = sin persistencia (stateless).

    Returns:
        StateGraph compilado listo para graph.invoke().
    """
    builder = StateGraph(AgentState)

    # ── Nodos ─────────────────────────────────────────────────────────────────
    builder.add_node("document_router", document_router_node)
    builder.add_node("ingestion", ingestion_node)
    builder.add_node(
        "retrieval",
        retrieval_node,
        retry=3,  # Reintentar hasta 3 veces ante fallos transitorios
    )
    builder.add_node("grade", grade_documents_node)
    builder.add_node(
        "generation",
        generation_node,
        retry=2,
    )
    builder.add_node("reflection", reflection_node)
    builder.add_node("supervisor", supervisor_node)

    if with_tools:
        tool_node = ToolNode(tools=ALL_TOOLS)
        builder.add_node("tools", tool_node)

    # ── Edges lineales ────────────────────────────────────────────────────────
    builder.add_edge(START, "document_router")
    builder.add_edge("retrieval", "grade")

    # ── Edges condicionales ───────────────────────────────────────────────────
    builder.add_conditional_edges(
        "document_router",
        route_after_router,
        {"ingestion": "ingestion", "retrieval": "retrieval"},
    )

    builder.add_conditional_edges(
        "ingestion",
        route_after_ingestion,
        {"retrieval": "retrieval", END: END},
    )

    if with_tools:
        builder.add_conditional_edges(
            "generation",
            route_after_generation,
            {"tools": "tools", "reflection": "reflection"},
        )
        builder.add_edge("tools", "generation")
    else:
        builder.add_edge("generation", "reflection")

    builder.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {"retrieval": "retrieval", END: END},
    )

    # CRAG grading: correct → generation, ambiguous/incorrect → retrieval (retry)
    builder.add_conditional_edges(
        "grade",
        route_after_grading,
        {"generation": "generation", "retrieval": "retrieval"},
    )

    # ── Compilar ──────────────────────────────────────────────────────────────
    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    # Node caching para evitar re-ejecución de nodos idénticos
    try:
        from langgraph.cache.memory import InMemoryCache  # noqa: PLC0415
        compile_kwargs["cache"] = InMemoryCache()
    except ImportError:
        pass  # Cache no disponible — continúa sin ella

    graph = builder.compile(**compile_kwargs)

    log.info(
        "graph_compiled",
        nodes=list(builder.nodes.keys()) if hasattr(builder, "nodes") else "unknown",
        with_tools=with_tools,
        with_checkpointer=checkpointer is not None,
    )

    return graph


# ─── Singleton del grafo ──────────────────────────────────────────────────────

_graph: StateGraph | None = None


def get_graph(
    with_tools: bool = True,
    checkpointer: Any | None = None,
    force_rebuild: bool = False,
) -> StateGraph:
    """
    Retorna la instancia singleton del grafo compilado.

    Args:
        with_tools: Incluir ToolNode en el grafo.
        checkpointer: Checkpointer para persistencia de estado.
        force_rebuild: Forzar reconstrucción (útil en tests).
    """
    global _graph  # noqa: PLW0603

    if _graph is None or force_rebuild:
        _graph = build_graph(with_tools=with_tools, checkpointer=checkpointer)

    return _graph


# ─── API de alto nivel ────────────────────────────────────────────────────────

def run_agent(
    user_query: str,
    uploaded_files: list[str] | None = None,
    session_id: str = "",
    max_iterations: int = 2,
    config: dict | None = None,
) -> dict[str, Any]:
    """
    Ejecuta el agente Fénix RAG para una query del usuario.

    Args:
        user_query: Pregunta del usuario.
        uploaded_files: Paths a archivos que el usuario quiere consultar.
        session_id: ID de sesión para trazabilidad y checkpointing.
        max_iterations: Máximo de ciclos reflection → re-retrieval.
        config: Config de LangGraph (ej: {"configurable": {"thread_id": session_id}}).

    Returns:
        Dict con final_answer, sources, reflection, iteration_count.
    """
    graph = get_graph()

    state = initial_state(
        user_query=user_query,
        session_id=session_id,
        uploaded_files=uploaded_files,
        max_iterations=max_iterations,
    )

    invoke_config = config or {}
    if session_id and "configurable" not in invoke_config:
        invoke_config["configurable"] = {"thread_id": session_id}

    log.info(
        "agent_run_start",
        query=user_query[:80],
        session=session_id,
        files=len(uploaded_files or []),
    )

    try:
        final_state = graph.invoke(state, config=invoke_config)
    except Exception as exc:
        log.error("agent_run_failed", error=str(exc))
        return {
            "final_answer": "Ocurrió un error al procesar tu consulta. Por favor intenta de nuevo.",
            "sources": [],
            "error": str(exc),
            "iteration_count": 0,
        }

    answer = (
        final_state.get("final_answer")
        or final_state.get("draft_answer")
        or "No encontré información relevante para responder tu pregunta."
    )

    log.info(
        "agent_run_complete",
        answer_len=len(answer),
        iterations=final_state.get("iteration_count", 0),
        sources=len(final_state.get("sources", [])),
    )

    return {
        "final_answer": answer,
        "sources": final_state.get("sources", []),
        "reflection": final_state.get("reflection"),
        "iteration_count": final_state.get("iteration_count", 0),
        "retrieval_strategy": final_state.get("retrieval_strategy", ""),
        "ingested_files": [
            p["source_path"] for p in final_state.get("ingestion_plans", [])
        ],
    }


async def arun_agent(
    user_query: str,
    uploaded_files: list[str] | None = None,
    session_id: str = "",
    max_iterations: int = 2,
) -> dict[str, Any]:
    """Versión async de run_agent para uso en FastAPI / endpoints async."""
    graph = get_graph()

    state = initial_state(
        user_query=user_query,
        session_id=session_id,
        uploaded_files=uploaded_files,
        max_iterations=max_iterations,
    )

    config = {"configurable": {"thread_id": session_id}} if session_id else {}

    try:
        final_state = await graph.ainvoke(state, config=config)
    except Exception as exc:
        log.error("agent_arun_failed", error=str(exc))
        return {
            "final_answer": "Error procesando la consulta.",
            "sources": [],
            "error": str(exc),
        }

    answer = (
        final_state.get("final_answer")
        or final_state.get("draft_answer")
        or "No encontré información relevante."
    )

    return {
        "final_answer": answer,
        "sources": final_state.get("sources", []),
        "reflection": final_state.get("reflection"),
        "iteration_count": final_state.get("iteration_count", 0),
    }
