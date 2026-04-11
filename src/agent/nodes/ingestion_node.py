"""
Ingestion node — ejecuta IngestionPipeline e indexa chunks a Chroma.
"""

from __future__ import annotations

from src.agent.metrics import node_timer
from src.agent.state import AgentState
from src.config.logging import get_logger

log = get_logger(__name__)


def ingestion_node(state: AgentState) -> dict:
    """
    Ejecuta IngestionPipeline e indexa chunks a Chroma.

    Procesa cada plan de ingestión generado por el document_router.
    Retorna resultados de ingestión por archivo.
    """
    from src.ingestion.pipeline import IngestionPipeline  # noqa: PLC0415

    with node_timer(state, "ingestion") as timer:
        plans = state.get("ingestion_plans", [])
        if not plans:
            return {"error": "No hay planes de ingestión", "ingested_documents": [], **timer.to_state()}

        pipeline = IngestionPipeline()
        ingested = []
        total_chunks = 0

        for plan in plans:
            source_path = plan.get("source_path", "")
            if not source_path:
                continue

            result = pipeline.ingest_file(source_path)
            chunk_count = result.chunk_count if result.success else 0
            total_chunks += chunk_count

            ingested.append({
                "source_path": source_path,
                "success": result.success,
                "chunk_count": chunk_count,
                "page_count": result.page_count,
                "loader_used": result.loader_used,
                "errors": result.errors,
            })

            if result.success:
                log.info(
                    "ingestion_success",
                    file=source_path,
                    chunks=chunk_count,
                    pages=result.page_count,
                )
            else:
                log.error(
                    "ingestion_failed",
                    file=source_path,
                    errors=result.errors,
                )

        timer.update(docs_count=len(ingested), extra={
            "total_chunks": total_chunks,
            "success_count": sum(1 for i in ingested if i["success"]),
        })

        return {
            "ingested_documents": ingested,
            "error": None if any(i["success"] for i in ingested) else "Todos los archivos fallaron",
            **timer.to_state(),
        }
