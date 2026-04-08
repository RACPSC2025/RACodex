"""
MCP Tools — Funciones de backend para las herramientas MCP de Fénix RAG.

Cada función retorna un dict que el servidor MCP (server.py)
formatea para presentar al cliente.
"""

from __future__ import annotations

import uuid
from typing import Any

from src.config.logging import get_logger
from src.config.settings import get_settings

log = get_logger(__name__)


def query_legal_document(
    query: str,
    session_id: str = "",
    max_iterations: int = 2,
) -> dict[str, Any]:
    """
    Ejecuta la consulta al agente RAG.

    Returns:
        Dict con answer, sources, reflection_score, retrieval_strategy, iteration_count.
    """
    from src.agent.graph import run_agent  # noqa: PLC0415

    result = run_agent(
        user_query=query,
        session_id=session_id or str(uuid.uuid4()),
        max_iterations=max_iterations,
    )

    return {
        "answer": result.get("final_answer", "No encontré información relevante."),
        "sources": result.get("sources", []),
        "reflection_score": result.get("reflection", {}).get("score") if result.get("reflection") else None,
        "retrieval_strategy": result.get("retrieval_strategy", ""),
        "iteration_count": result.get("iteration_count", 0),
    }


def lookup_article(
    article_number: str,
    source_filter: str = "",
) -> dict[str, Any]:
    """
    Busca artículo por número en el corpus.

    Returns:
        Dict con found, content, source, page.
    """
    from src.retrieval.ensemble import EnsembleRetriever, RetrievalQuery  # noqa: PLC0415
    from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415

    vs = get_vector_store()
    if not vs.is_initialized:
        vs.open_or_create()

    ensemble = EnsembleRetriever(vector_store=vs)
    query_text = f"Artículo {article_number}"
    if source_filter:
        query_text += f" {source_filter}"

    result = ensemble.retrieve(RetrievalQuery(text=query_text, top_k=3))

    if not result.documents:
        return {"found": False}

    doc = result.documents[0]
    return {
        "found": True,
        "content": doc.page_content,
        "source": doc.metadata.get("source", "Desconocida"),
        "page": doc.metadata.get("page", ""),
    }


def search_documents(
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Búsqueda híbrida directa.

    Returns:
        Dict con results (lista de fragments).
    """
    from src.retrieval.ensemble import EnsembleRetriever, RetrievalQuery  # noqa: PLC0415
    from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415

    vs = get_vector_store()
    if not vs.is_initialized:
        vs.open_or_create()

    ensemble = EnsembleRetriever(vector_store=vs)
    result = ensemble.retrieve(RetrievalQuery(text=query, top_k=top_k))

    items = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", ""),
            "article": doc.metadata.get("article_number", ""),
            "page": doc.metadata.get("page", ""),
        }
        for doc in result.documents
    ]

    return {"results": items}


def ingest_document(
    file_path: str,
    session_id: str = "",
) -> dict[str, Any]:
    """
    Indexa un documento en el corpus.

    Returns:
        Dict con success, source, chunks_indexed, loader_used, already_indexed, error.
    """
    from src.ingestion.pipeline import IngestionPipeline  # noqa: PLC0415

    pipeline = IngestionPipeline()
    result = pipeline.ingest_file(file_path)

    return {
        "success": result.success,
        "source": str(result.source_path),
        "chunks_indexed": result.chunk_count,
        "loader_used": result.loader_used,
        "already_indexed": False,
        "error": result.errors[0] if result.errors else None,
    }


def extract_obligations_mcp(
    query: str,
    source_filter: str = "",
) -> dict[str, Any]:
    """
    Extrae obligaciones estructuradas.

    Returns:
        Dict con obligations (lista de dicts con articulo, sujeto, obligacion, plazo, sancion).
    """
    from src.config.providers import get_llm  # noqa: PLC0415

    llm = get_llm()

    prompt = (
        f"Extrae todas las obligaciones mencionadas en relación con: {query}\n\n"
        "Para cada obligación, retorna en formato JSON:\n"
        "- articulo: número del artículo\n"
        "- nivel_criticidad: 1-5\n"
        "- sujeto_obligado: quién debe cumplir\n"
        "- obligacion: qué debe hacer\n"
        "- plazo: cuándo\n"
        "- sancion: consecuencia del incumplimiento\n\n"
        "Si no hay información para un campo, usa 'no especificada'."
    )

    response = llm.invoke(prompt)

    # Parsear response en estructura de obligaciones
    obligations = []
    for line in response.content.split("\n"):
        line = line.strip()
        if line.startswith("-") or line.startswith("*"):
            parts = line.lstrip("-* ").split(":")
            if len(parts) >= 2:
                key = parts[0].strip().lower()
                value = ":".join(parts[1:]).strip()
                obligations.append({
                    "articulo": "N/A",
                    "nivel_criticidad": 3,
                    "sujeto_obligado": value if "sujeto" in key else "",
                    "obligacion": value if "obligacion" in key else "",
                    "plazo": value if "plazo" in key else "",
                    "sancion": value if "sancion" in key else "no especificada",
                })

    return {
        "obligations": obligations if obligations else [{"articulo": "N/A", "obligacion": response.content, "nivel_criticidad": 0, "sujeto_obligado": "", "plazo": "", "sancion": "no especificada"}],
    }


def get_corpus_stats_mcp() -> dict[str, Any]:
    """
    Estadísticas del corpus.

    Returns:
        Dict con total_chunks, sources (lista de nombres).
    """
    from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415

    vs = get_vector_store()

    if not vs.is_initialized:
        return {"total_chunks": 0, "sources": []}

    count = vs.count()

    return {
        "total_chunks": count,
        "sources": [],  # Se puede enriquecer con metadata del store
    }


__all__ = [
    "query_legal_document",
    "lookup_article",
    "search_documents",
    "ingest_document",
    "extract_obligations_mcp",
    "get_corpus_stats_mcp",
]
