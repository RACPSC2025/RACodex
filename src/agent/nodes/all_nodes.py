"""
Nodos individuales del grafo LangGraph — implementaciones reales.

Cada nodo es una función pura que recibe AgentState y retorna un dict
con las actualizaciones al estado.
"""

from __future__ import annotations

from typing import Any

from src.agent.state import AgentState
from src.config.logging import get_logger

log = get_logger(__name__)


def document_router_node(state: AgentState) -> dict:
    """Clasifica archivos subidos y decide ruta de ingestión."""
    from src.agent.skills.document_classifier import DocumentClassifierSkill  # noqa: PLC0415

    uploaded = state.get("uploaded_files", [])
    if not uploaded:
        return {"route": "retrieval", "ingestion_plans": []}

    classifier = DocumentClassifierSkill()
    plans = []

    for fpath in uploaded:
        plan = classifier.classify(fpath)
        plans.append(plan)

    log.info(
        "document_router_complete",
        files=len(uploaded),
        plans=[p.get("loader_type", "?") for p in plans],
    )

    return {
        "route": "ingestion",
        "ingestion_plans": plans,
    }


def ingestion_node(state: AgentState) -> dict:
    """Ejecuta IngestionPipeline e indexa chunks a Chroma."""
    from src.ingestion.pipeline import IngestionPipeline  # noqa: PLC0415

    plans = state.get("ingestion_plans", [])
    if not plans:
        return {"error": "No hay planes de ingestión", "ingested_documents": []}

    pipeline = IngestionPipeline()
    ingested = []

    for plan in plans:
        source_path = plan.get("source_path", "")
        if not source_path:
            continue

        result = pipeline.ingest_file(source_path)
        ingested.append({
            "source_path": source_path,
            "success": result.success,
            "chunk_count": result.chunk_count,
            "page_count": result.page_count,
            "loader_used": result.loader_used,
            "errors": result.errors,
        })

        if result.success:
            log.info(
                "ingestion_success",
                file=source_path,
                chunks=result.chunk_count,
                pages=result.page_count,
            )
        else:
            log.error(
                "ingestion_failed",
                file=source_path,
                errors=result.errors,
            )

    return {
        "ingested_documents": ingested,
        "error": None if any(i["success"] for i in ingested) else "Todos los archivos fallaron",
    }


def retrieval_node(state: AgentState) -> dict:
    """
    Ejecuta EnsembleRetriever con Multi-Query Fusion + Context Enrichment.

    1. QueryTransformer genera variantes: [original, rewritten, step-back]
    2. Retrieval con cada variante del query
    3. Deduplicación por source::chunk_index
    4. Context Enrichment aplicado por el ensemble
    """
    from src.agent.skills.query_transformer import QueryTransformer  # noqa: PLC0415
    from src.retrieval.base import RetrievalQuery  # noqa: PLC0415
    from src.retrieval.ensemble import get_ensemble_retriever  # noqa: PLC0415
    from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415

    query_text = state.get("active_query") or state.get("user_query", "")
    if not query_text:
        return {"retrieval_results": [], "retrieval_strategy": "none"}

    vs = get_vector_store()
    if not vs.is_initialized:
        vs.open_or_create()

    # Ensemble con context enrichment habilitado
    ensemble = get_ensemble_retriever(
        vector_store=vs,
        use_context_enrichment=True,
        context_window_size=2,
    )

    # Paso 1: QueryTransformer genera variantes
    transformer = QueryTransformer()
    variants = transformer.transform_all(query_text)

    log.debug(
        "retrieval_query_variants",
        original=query_text[:60],
        variants=len(variants),
        queries=[v[:60] for v in variants],
    )

    # Paso 2: Retrieval con cada variante + deduplicación
    seen_ids: set[str] = set()
    all_docs = []
    total_before_dedup = 0

    for variant in variants:
        result = ensemble.retrieve(RetrievalQuery(text=variant))
        total_before_dedup += len(result.documents)
        for doc in result.documents:
            doc_id = f"{doc.metadata.get('source', '')}::{doc.metadata.get('chunk_index', '')}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)

    log.info(
        "retrieval_complete",
        original_query=query_text[:60],
        variants=len(variants),
        total_docs_before_dedup=total_before_dedup,
        total_docs_after_dedup=len(all_docs),
        strategy="multi_query_fusion",
    )

    return {
        "retrieval_results": all_docs,
        "retrieval_strategy": "multi_query_fusion",
    }


def generation_node(state: AgentState) -> dict:
    """Genera respuesta con Rethinking (Re2) — dos pasadas de lectura."""
    from src.agent.skills.rethinking import generate_with_rethinking  # noqa: PLC0415

    query = state.get("active_query") or state.get("user_query", "")
    docs = state.get("retrieval_results", [])

    if not docs:
        return {
            "draft_answer": "No encontré documentos relevantes para responder tu consulta.",
            "sources": [],
        }

    answer, sources = generate_with_rethinking(query, docs)

    return {
        "draft_answer": answer,
        "sources": sources,
    }


def reflection_node(state: AgentState) -> dict:
    """Auto-evaluación de la respuesta generada."""
    from src.agent.skills.answer_validator import AnswerValidatorSkill  # noqa: PLC0415
    from src.agent.state import ReflectionOutput  # noqa: PLC0415

    draft = state.get("draft_answer", "")
    query = state.get("user_query", "")
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 2)

    # Validación rule-based primero (sin costo de LLM)
    validator = AnswerValidatorSkill()
    validation = validator.validate(draft, query)

    if validation.is_valid:
        return {
            "final_answer": draft,
            "reflection": ReflectionOutput(
                score=validation.score,
                is_grounded=True,
                has_hallucination=False,
                cites_source=validation.cites_source,
                feedback="Respuesta válida",
                reformulated_query="",
            ),
            "route": "END",
            "iteration_count": iteration + 1,
        }

    # Si no es válida y quedan iteraciones → reformular
    if iteration < max_iter:
        reformulated = validator.suggest_reformulation(query, draft)
        return {
            "active_query": reformulated,
            "reflection": ReflectionOutput(
                score=validation.score,
                is_grounded=False,
                has_hallucination=validation.has_hallucination,
                cites_source=validation.cites_source,
                feedback=validation.feedback,
                reformulated_query=reformulated,
            ),
            "route": "retrieval",
            "iteration_count": iteration + 1,
        }

    # Iteraciones agotadas → usar borrador con advertencia
    return {
        "final_answer": draft + "\n\n⚠️ Nota: Esta respuesta puede estar incompleta.",
        "reflection": ReflectionOutput(
            score=validation.score,
            is_grounded=False,
            has_hallucination=False,
            cites_source=validation.cites_source,
            feedback="Iteraciones agotadas",
            reformulated_query="",
        ),
        "route": "END",
        "iteration_count": iteration + 1,
    }


def supervisor_node(state: AgentState) -> dict:
    """Supervisor pattern: coordina subagentes especializados."""
    # Por ahora, solo valida la ruta — implementación completa en Fase futura
    return {"route": state.get("route", "retrieval")}


__all__ = [
    "document_router_node",
    "ingestion_node",
    "retrieval_node",
    "generation_node",
    "reflection_node",
    "supervisor_node",
]
