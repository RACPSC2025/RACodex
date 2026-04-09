"""Retrieval node — re-export from all_nodes."""
from src.agent.nodes.all_nodes import retrieval_node

__all__ = ["retrieval_node"]


def retrieval_node_with_query_transform(state: dict) -> dict:
    """
    Retrieval node con Query Transformation completa.

    Flujo:
      1. QueryTransformer → [original, rewritten, step-back]
      2. Retrieval con cada variante
      3. Deduplicación + RRF de resultados
      4. Retorna documentos únicos combinados
    """
    from src.agent.skills.query_transformer import QueryTransformer
    from src.retrieval.base import RetrievalQuery
    from src.retrieval.ensemble import get_ensemble_retriever
    from src.retrieval.vector_store import get_vector_store

    query_text = state.get("active_query") or state.get("user_query", "")
    if not query_text:
        return {"retrieval_results": [], "retrieval_strategy": "none"}

    vs = get_vector_store()
    if not vs.is_initialized:
        vs.open_or_create()

    ensemble = get_ensemble_retriever(
        vector_store=vs,
        use_context_enrichment=True,
        context_window_size=2,
    )

    # Query transformation
    transformer = QueryTransformer()
    query_variants = transformer.transform_all(query_text)

    # Retrieval con cada variante + deduplicación
    seen_ids: set[str] = set()
    all_docs = []

    for variant in query_variants:
        result = ensemble.retrieve(RetrievalQuery(text=variant))
        for doc in result.documents:
            doc_id = f"{doc.metadata.get('source', '')}::{doc.metadata.get('chunk_index', '')}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)

    return {
        "retrieval_results": all_docs,
        "retrieval_strategy": f"fusion_{len(query_variants)}variants",
    }
