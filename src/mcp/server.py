"""
MCP Server para Fénix RAG.

Expone las capacidades del agente como herramientas MCP para
Claude Desktop, Cursor y otros clientes compatibles.

Transporte:
  stdio — Claude Desktop/Cursor (python -m src.mcp.server)
  http  — integración remota   (python -m src.mcp.server --transport http --port 8001)

Claude Desktop config (~/.claude/claude_desktop_config.json):
  {
    "mcpServers": {
      "fenix-rag": {
        "command": "python",
        "args": ["-m", "src.mcp.server"],
        "cwd": "/ruta/al/proyecto"
      }
    }
  }
"""
from __future__ import annotations

import asyncio
import sys

from src.config.logging import configure_logging, get_logger
from src.config.settings import get_settings
from src.mcp.tools import (
    extract_obligations_mcp,
    get_corpus_stats_mcp,
    ingest_document,
    lookup_article,
    query_legal_document,
    search_documents,
)

log = get_logger(__name__)


def build_server():
    """Construye el servidor MCP con todas las tools registradas."""
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("mcp no instalado. Ejecuta: pip install mcp") from exc

    server = FastMCP(
        name="fenix-rag",
        version="1.0.0",
        description=(
            "Sistema Profesional de Retrieval-Augmented Generation. "
            "Consulta documentos con retrieval híbrido y auto-reflexión."
        ),
    )

    # ── query_legal ───────────────────────────────────────────────────────────
    @server.tool(
        name="query_legal",
        description=(
            "Consulta completa al agente RAG legal. Ejecuta retrieval híbrido, "
            "genera respuesta y aplica auto-reflexión. "
            "Usar para cualquier pregunta sobre normativa colombiana."
        ),
    )
    async def query_legal(query: str, session_id: str = "", max_iterations: int = 2) -> str:
        result = await query_legal_document(query, session_id, max_iterations)
        answer = result.get("answer", "No encontré información relevante.")
        sources = result.get("sources", [])
        score = result.get("reflection_score")

        lines = [answer, ""]
        if sources:
            lines.append("**Fuentes:**")
            for s in sources[:5]:
                ref = f"- {s.get('source', '')}"
                if s.get("article"):
                    ref += f" | Art. {s['article']}"
                if s.get("page"):
                    ref += f" | Pág. {s['page']}"
                lines.append(ref)
        if score is not None:
            strategy = result.get("retrieval_strategy", "")
            iters = result.get("iteration_count", 0)
            lines.append(f"\n*Calidad: {score:.2f} | {strategy} | iter: {iters}*")

        return "\n".join(lines)

    # ── lookup_article ────────────────────────────────────────────────────────
    @server.tool(
        name="lookup_article",
        description=(
            "Busca el texto completo de un artículo por número. "
            "Usar para '¿qué dice el artículo 2.2.4.6.1?'. "
            "Más rápido que query_legal cuando se conoce el número exacto."
        ),
    )
    async def find_article(article_number: str, source_filter: str = "") -> str:
        result = await lookup_article(article_number, source_filter)
        if not result.get("found"):
            return f"No se encontró el artículo {article_number} en los documentos indexados."
        content = result.get("content", "")
        source = result.get("source", "")
        page = result.get("page", "")
        footer = f"\n\n*Fuente: {source}"
        if page:
            footer += f" | Pág. {page}"
        return content + footer + "*"

    # ── search_corpus ─────────────────────────────────────────────────────────
    @server.tool(
        name="search_corpus",
        description=(
            "Búsqueda híbrida directa — retorna fragmentos sin interpretación LLM. "
            "Usar cuando se necesitan los textos originales tal cual."
        ),
    )
    async def search_corpus(query: str, top_k: int = 5) -> str:
        result = await search_documents(query, top_k)
        items = result.get("results", [])
        if not items:
            return "No se encontraron documentos relevantes."
        lines = [f"**{len(items)} fragmentos encontrados:**\n"]
        for i, item in enumerate(items, 1):
            header = f"**[{i}] {item.get('source', '')}"
            if item.get("article"):
                header += f" | Art. {item['article']}"
            if item.get("page"):
                header += f" | Pág. {item['page']}"
            lines.append(header + "**")
            lines.append(item.get("content", "")[:400])
            lines.append("")
        return "\n".join(lines)

    # ── ingest_document ───────────────────────────────────────────────────────
    @server.tool(
        name="ingest_document",
        description=(
            "Indexa un documento en el corpus. "
            "Soporta PDF (nativo y escaneado), Word, Excel e imágenes. "
            "El tipo se detecta automáticamente."
        ),
    )
    async def index_document(file_path: str, session_id: str = "") -> str:
        result = await ingest_document(file_path, session_id)
        if not result.get("success"):
            return f"Error al indexar '{file_path}': {result.get('error', 'desconocido')}"
        if result.get("already_indexed"):
            return f"'{result.get('source', file_path)}' ya estaba indexado."
        return (
            f"'{result.get('source', file_path)}' indexado.\n"
            f"- Chunks: {result.get('chunks_indexed', 0)}\n"
            f"- Loader: {result.get('loader_used', 'auto')}"
        )

    # ── extract_obligations ───────────────────────────────────────────────────
    @server.tool(
        name="extract_obligations",
        description=(
            "Extrae obligaciones legales estructuradas: sujeto, acción, plazo, sanción. "
            "Útil para auditorías de cumplimiento SST."
        ),
    )
    async def obligations(query: str, source_filter: str = "") -> str:
        result = await extract_obligations_mcp(query, source_filter)
        items = result.get("obligations", [])
        if not items:
            return f"No se encontraron obligaciones para: {query}"
        lines = [f"**{len(items)} obligaciones:**\n"]
        for i, ob in enumerate(items, 1):
            lines.append(f"**{i}. Art. {ob.get('articulo', '')}** (criticidad {ob.get('nivel_criticidad', 0)}/5)")
            if ob.get("sujeto_obligado"):
                lines.append(f"   Sujeto: {ob['sujeto_obligado']}")
            lines.append(f"   Obligación: {ob.get('obligacion', '')}")
            if ob.get("plazo"):
                lines.append(f"   Plazo: {ob['plazo']}")
            if ob.get("sancion") and ob["sancion"] != "no especificada":
                lines.append(f"   Sanción: {ob['sancion']}")
            lines.append("")
        return "\n".join(lines)

    # ── corpus_status ─────────────────────────────────────────────────────────
    @server.tool(
        name="corpus_status",
        description="Muestra documentos indexados y estadísticas del corpus.",
    )
    async def corpus_status() -> str:
        result = await get_corpus_stats_mcp()
        total = result.get("total_chunks", 0)
        sources = result.get("sources", [])
        if total == 0:
            return "El corpus está vacío. Usa 'ingest_document' para indexar documentos."
        lines = [
            "**Corpus Fénix RAG**",
            f"- Chunks indexados: {total}",
            f"- Documentos únicos: {len(sources)}",
        ]
        if sources:
            lines.append("\n**Documentos:**")
            for src in sorted(sources)[:20]:
                lines.append(f"  - {src}")
            if len(sources) > 20:
                lines.append(f"  ... y {len(sources) - 20} más")
        return "\n".join(lines)

    log.info("mcp_server_built", name=server.name)
    return server


async def run_stdio() -> None:
    configure_logging()
    get_settings().ensure_directories()
    log.info("mcp_server_starting", transport="stdio")
    server = build_server()
    await server.run_stdio_async()


async def run_http(host: str = "0.0.0.0", port: int = 8001) -> None:
    configure_logging()
    get_settings().ensure_directories()
    log.info("mcp_server_starting", transport="http", host=host, port=port)
    server = build_server()
    await server.run_http_async(host=host, port=port)


if __name__ == "__main__":
    import argparse  # noqa: PLC0415
    parser = argparse.ArgumentParser(description="Fénix RAG MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.transport == "http":
        asyncio.run(run_http(args.host, args.port))
    else:
        asyncio.run(run_stdio())
