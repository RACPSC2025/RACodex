"""
Tools de ingestion para el agente LangGraph.

Estos @tool son las herramientas que el agente ReAct puede invocar
para procesar documentos subidos por el usuario.

Diferencia tool vs nodo:
  - Los nodos (nodes/) son pasos del grafo ejecutados siempre en secuencia
  - Los tools son funciones que el agente invoca discrecionalmente con
    ToolNode cuando el LLM decide que necesita una herramienta concreta

Para ingestion: el DocumentRouterNode decide el plan,
el IngestionNode ejecuta el pipeline. Los tools aquí son para
casos donde el agente necesita re-procesar o inspeccionar documentos
de forma interactiva en el ciclo ReAct.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.tools import tool

from src.config.logging import get_logger

log = get_logger(__name__)


@tool
def ingest_pdf(file_path: str, loader_type: str = "auto") -> dict:
    """
    Procesa un PDF y lo indexa en el vector store.

    Usa este tool cuando el usuario sube un PDF y quiere hacerle preguntas.
    El loader_type puede ser 'auto' (detección automática), 'pymupdf'
    (PDF nativo), 'ocr' (PDF escaneado), o 'docling' (PDF complejo con tablas).

    Args:
        file_path: Ruta absoluta al archivo PDF.
        loader_type: Tipo de loader a usar. Default: 'auto' (clasificación automática).

    Returns:
        Dict con {success, chunks_indexed, source, loader_used, error}.
    """
    path = Path(file_path)

    if not path.exists():
        return {"success": False, "error": f"Archivo no encontrado: {file_path}", "chunks_indexed": 0}

    try:
        from src.agent.skills.document_classifier import get_document_classifier  # noqa: PLC0415
        from src.ingestion.registry import get_registry  # noqa: PLC0415
        from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415

        # Clasificar si es auto
        if loader_type == "auto":
            classifier = get_document_classifier()
            plan = classifier.classify(path)
            effective_loader = plan["loader_type"]
        else:
            effective_loader = loader_type

        # Cargar con el loader apropiado
        registry = get_registry()
        loader = registry.select(path)
        documents = loader.load(path)

        # Indexar en vector store
        store = get_vector_store()
        store.open_or_create()
        indexed = store.add_documents(documents)

        log.info(
            "ingest_pdf_tool_complete",
            file=path.name,
            loader=effective_loader,
            chunks=indexed,
        )

        return {
            "success": True,
            "chunks_indexed": indexed,
            "source": path.name,
            "loader_used": effective_loader,
            "error": None,
        }

    except Exception as exc:
        log.error("ingest_pdf_tool_failed", file=str(path), error=str(exc))
        return {
            "success": False,
            "chunks_indexed": 0,
            "source": path.name,
            "loader_used": loader_type,
            "error": str(exc),
        }


@tool
def ingest_excel(file_path: str, rows_per_chunk: int = 20) -> dict:
    """
    Procesa un archivo Excel y lo indexa en el vector store.

    Ideal para tablas normativas, matrices de sanciones, listados de EPS/ARL.
    Cada grupo de filas se convierte en un chunk con nombre de columna como contexto.

    Args:
        file_path: Ruta al archivo .xlsx o .xls.
        rows_per_chunk: Filas por chunk. Menor = chunks más específicos.

    Returns:
        Dict con {success, chunks_indexed, sheets_processed, error}.
    """
    path = Path(file_path)

    if not path.exists():
        return {"success": False, "error": f"No encontrado: {file_path}", "chunks_indexed": 0}

    try:
        from src.ingestion.loaders.excel_loader import ExcelLoader  # noqa: PLC0415
        from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415

        loader = ExcelLoader(rows_per_chunk=rows_per_chunk)
        documents = loader.load(path)

        store = get_vector_store()
        store.open_or_create()
        indexed = store.add_documents(documents)

        sheets = list({d.metadata.get("sheet_name", "") for d in documents})

        return {
            "success": True,
            "chunks_indexed": indexed,
            "sheets_processed": sheets,
            "source": path.name,
            "error": None,
        }

    except Exception as exc:
        log.error("ingest_excel_tool_failed", file=str(path), error=str(exc))
        return {"success": False, "chunks_indexed": 0, "error": str(exc)}


@tool
def ingest_word(file_path: str) -> dict:
    """
    Procesa un documento Word (.docx o .doc) y lo indexa.

    Args:
        file_path: Ruta al archivo Word.

    Returns:
        Dict con {success, chunks_indexed, error}.
    """
    path = Path(file_path)

    if not path.exists():
        return {"success": False, "error": f"No encontrado: {file_path}", "chunks_indexed": 0}

    try:
        from src.ingestion.loaders.word_loader import WordLoader  # noqa: PLC0415
        from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415

        loader = WordLoader()
        documents = loader.load(path)

        store = get_vector_store()
        store.open_or_create()
        indexed = store.add_documents(documents)

        return {
            "success": True,
            "chunks_indexed": indexed,
            "source": path.name,
            "error": None,
        }

    except Exception as exc:
        log.error("ingest_word_tool_failed", file=str(path), error=str(exc))
        return {"success": False, "chunks_indexed": 0, "error": str(exc)}


@tool
def ingest_image_pdf(file_path: str, render_dpi: int = 300) -> dict:
    """
    Procesa un PDF escaneado o imagen usando OCR.

    Usa este tool cuando el PDF es una fotografía o escaneo sin texto seleccionable.
    El OCR puede tardar 30-60 segundos por página.

    Args:
        file_path: Ruta al PDF escaneado o imagen (JPEG/PNG).
        render_dpi: DPI de renderizado. 300 = alta calidad, 150 = rápido.

    Returns:
        Dict con {success, chunks_indexed, pages_processed, error}.
    """
    path = Path(file_path)

    if not path.exists():
        return {"success": False, "error": f"No encontrado: {file_path}", "chunks_indexed": 0}

    try:
        from src.ingestion.loaders.pdf_ocr import OCRLoader  # noqa: PLC0415
        from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415

        loader = OCRLoader(render_dpi=render_dpi)
        documents = loader.load(path)

        store = get_vector_store()
        store.open_or_create()
        indexed = store.add_documents(documents)

        pages = len({d.metadata.get("page", "") for d in documents if d.metadata.get("page")})

        return {
            "success": True,
            "chunks_indexed": indexed,
            "pages_processed": pages,
            "source": path.name,
            "error": None,
        }

    except Exception as exc:
        log.error("ingest_ocr_tool_failed", file=str(path), error=str(exc))
        return {"success": False, "chunks_indexed": 0, "error": str(exc)}


@tool
def list_indexed_documents() -> dict:
    """
    Lista todos los documentos indexados en el vector store.

    Returns:
        Dict con {total_chunks, sources} donde sources es la lista de documentos únicos.
    """
    try:
        from src.retrieval.vector_store import get_vector_store  # noqa: PLC0415

        store = get_vector_store()
        if not store.is_initialized:
            return {"total_chunks": 0, "sources": [], "error": None}

        collection = store.get_raw_collection()
        results = collection.get(include=["metadatas"])

        sources = list({
            m.get("source", "unknown")
            for m in (results.get("metadatas") or [])
        })

        return {
            "total_chunks": store.count(),
            "sources": sorted(sources),
            "error": None,
        }

    except Exception as exc:
        return {"total_chunks": 0, "sources": [], "error": str(exc)}
