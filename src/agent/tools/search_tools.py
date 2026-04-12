"""
Tools de búsqueda/retrieval para el agente LangGraph.

Tres niveles de granularidad:
  1. `semantic_search`  — búsqueda semántica directa (más rápida)
  2. `hybrid_search`    — vector + BM25 con RRF (mejor balance)
  3. `article_lookup`   — búsqueda exacta por número de artículo (más precisa)

El agente ReAct elige el tool según la naturaleza de la query:
  - Pregunta general → hybrid_search
  - Artículo específico ("2.2.4.6.1") → article_lookup
  - Exploración temática → semantic_search con más resultados
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.config.logging import get_logger

log = get_logger(__name__)


@tool
def semantic_search(query: str, top_k: int = 5, source_filter: str = "") -> dict:
    """
    Busca documentos semánticamente similares a la query.

    Usa búsqueda vectorial pura (embeddings). Óptima para preguntas
    conceptuales donde el vocabulario exacto puede variar.

    Args:
        query: Texto de la pregunta o tema a buscar.
        top_k: Número de documentos a retornar (1-20).
        source_filter: Filtrar por nombre de archivo (ej: "documentacion.pdf").

    Returns:
        Dict con {results: [{content, source, article, page, score}], total}.
    """
    try:
        from src.retrieval.base import RetrievalQuery  # noqa: PLC0415
        from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415

        store = get_vector_store()
        if not store.is_initialized:
            return {"results": [], "total": 0, "error": "Vector store no inicializado"}

        filters = {"source": source_filter} if source_filter else {}
        q = RetrievalQuery(text=query, top_k=min(max(1, top_k), 20), filters=filters)

        doc_score_pairs = store.similarity_search_with_score(q)

        results = [
            {
                "content": doc.page_content[:500],
                "source": doc.metadata.get("source", ""),
                "article": doc.metadata.get("article_number", ""),
                "page": doc.metadata.get("page", ""),
                "score": score,
            }
            for doc, score in doc_score_pairs
        ]

        return {"results": results, "total": len(results), "error": None}

    except Exception as exc:
        log.error("semantic_search_tool_failed", error=str(exc))
        return {"results": [], "total": 0, "error": str(exc)}


@tool
def hybrid_search(query: str, top_k: int = 8) -> dict:
    """
    Búsqueda híbrida combinando vectorial + BM25 con fusión RRF.

    Mejor opción para la mayoría de las consultas legales. Combina
    la semántica del embedding con la precisión léxica de BM25,
    ideal cuando la query menciona términos específicos (siglas, artículos).

    Args:
        query: Texto de la pregunta.
        top_k: Número de documentos a retornar (1-15).

    Returns:
        Dict con {results: [{content, source, article, page, rrf_score}], total}.
    """
    try:
        from src.retrieval.base import RetrievalQuery  # noqa: PLC0415
        from src.retrieval.ensemble import EnsembleRetriever, RetrievalStrategy  # noqa: PLC0415
        from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415

        store = get_vector_store()
        if not store.is_initialized:
            return {"results": [], "total": 0, "error": "Vector store no inicializado"}

        ensemble = EnsembleRetriever(
            vector_store=store,
            strategy=RetrievalStrategy.HYBRID,
            use_reranking=False,
            top_k=min(max(1, top_k), 15),
        )

        q = RetrievalQuery(text=query, top_k=min(max(1, top_k), 15))
        result = ensemble.retrieve(q)

        results = [
            {
                "content": doc.page_content[:500],
                "source": doc.metadata.get("source", ""),
                "article": doc.metadata.get("article_number", ""),
                "page": doc.metadata.get("page", ""),
                "rrf_score": doc.metadata.get("rrf_score", 0.0),
            }
            for doc in result.documents
        ]

        return {
            "results": results,
            "total": len(results),
            "strategy": "hybrid_rrf",
            "error": None,
        }

    except Exception as exc:
        log.error("hybrid_search_tool_failed", error=str(exc))
        return {"results": [], "total": 0, "error": str(exc)}


@tool
def article_lookup(article_number: str, source_filter: str = "") -> dict:
    """
    Busca el texto completo de un artículo específico por su número.

    Optimizado para consultas del tipo "¿qué dice el artículo 2.2.4.6.1?".
    Combina filtro exacto de metadata con búsqueda semántica para máxima precisión.

    Args:
        article_number: Número del artículo (ej: "2.2.4.6.1", "15", "ÚNICO").
        source_filter: Filtrar por nombre de archivo (ej: "documentacion.pdf").

    Returns:
        Dict con {found, article_number, content, source, page}.
    """
    try:
        from src.retrieval.base import RetrievalQuery  # noqa: PLC0415
        from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415

        store = get_vector_store()
        if not store.is_initialized:
            return {"found": False, "error": "Vector store no inicializado"}

        # Búsqueda filtrada por article_number en metadata
        filters: dict = {"article_number": article_number}
        if source_filter:
            filters["source"] = source_filter

        q = RetrievalQuery(
            text=f"artículo {article_number}",
            top_k=3,
            filters=filters,
        )

        docs = store.similarity_search(q)

        if not docs:
            # Fallback: búsqueda semántica sin filtro exacto
            q_fallback = RetrievalQuery(
                text=f"ARTÍCULO {article_number}",
                top_k=5,
                filters={"source": source_filter} if source_filter else {},
            )
            docs = store.similarity_search(q_fallback)

        if not docs:
            return {
                "found": False,
                "article_number": article_number,
                "content": "",
                "source": "",
                "page": "",
                "error": f"Artículo {article_number} no encontrado en los documentos indexados",
            }

        # Retornar el chunk más relevante
        best = docs[0]
        return {
            "found": True,
            "article_number": article_number,
            "content": best.page_content,
            "source": best.metadata.get("source", ""),
            "page": best.metadata.get("page", ""),
            "error": None,
        }

    except Exception as exc:
        log.error("article_lookup_tool_failed", article=article_number, error=str(exc))
        return {"found": False, "error": str(exc)}
