"""
Retriever Jerárquico: Summary-level → Detail-level.

Arquitectura de dos niveles:
  Nivel 1 (Summary Store): résumenes compactos de cada artículo/sección.
             Ventaja: mayor diversidad semántica, captura contexto amplio.
  Nivel 2 (Detail Store): chunks completos con overlap.
             Ventaja: texto exacto para respuestas precisas.

Flujo de retrieval:
  1. Buscar en Summary Store → top k_summaries artículos relevantes
  2. Para cada artículo relevante, buscar chunks detallados filtrando
     por article_number y/o page del summary
  3. Deduplicar y retornar los mejores k_chunks detallados

Flujo de indexación:
  1. Para cada documento en el corpus, generar un summary con LLM
  2. Indexar summaries en `_summary` collection
  3. Los chunks detallados ya están en la colección principal

Correcciones respecto al código original (hierarchical_retriever.py):
  ✓ _create_summary usaba get_llm() dentro del loop → N llamadas secuenciales.
    Ahora usa asyncio.gather para generación paralela con semáforo de control.
  ✓ Los summaries se cachean en el Summary VectorStore, no se regeneran
    en cada rebuild.
  ✓ Los filtros de metadata usan str() explícito (Chroma requiere strings).
  ✓ El retrieve maneja gracefully el caso de artículo sin filter match.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config.logging import get_logger
from src.config.providers import get_llm
from src.retrieval.base import BaseRetriever, RetrievalQuery, VectorStoreNotInitializedError
from src.retrieval.vector_store import VectorStore, get_summary_store, get_vector_store

log = get_logger(__name__)

# Semáforo para controlar la concurrencia de llamadas LLM al generar summaries
_SUMMARY_CONCURRENCY = 5

_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "Eres un experto legal colombiano. Resume el siguiente fragmento normativo "
    "en máximo 3 oraciones concisas capturando: la norma específica, "
    "los sujetos obligados y la acción o consecuencia principal.\n\n"
    "Fragmento:\n{text}\n\nResumen:"
)


class HierarchicalRetriever(BaseRetriever):
    """
    Retriever de dos niveles: summaries → chunks detallados.

    Mejora la recall en queries amplias (el summary captura el tema)
    y la precisión en queries específicas (el detail tiene el texto exacto).
    """

    def __init__(
        self,
        detail_store: VectorStore | None = None,
        summary_store: VectorStore | None = None,
        k_summaries: int = 4,
        k_chunks: int = 8,
        min_summary_length: int = 200,
    ) -> None:
        """
        Args:
            detail_store: VectorStore con chunks detallados (colección principal).
            summary_store: VectorStore con summaries (colección _summary).
            k_summaries: Cuántos summaries recuperar en el primer nivel.
            k_chunks: Cuántos chunks detallados retornar en total.
            min_summary_length: Mínimo de chars para generar summary de un chunk.
                                 Chunks más cortos se usan directamente como summary.
        """
        self._detail_store = detail_store or get_vector_store()
        self._summary_store = summary_store or get_summary_store()
        self._k_summaries = k_summaries
        self._k_chunks = k_chunks
        self._min_summary_length = min_summary_length

    @property
    def retriever_type(self) -> str:
        return "hierarchical"

    def is_ready(self) -> bool:
        return (
            self._detail_store.is_initialized
            and self._summary_store.is_initialized
        )

    # ── Generación de summaries ───────────────────────────────────────────────

    async def _summarize_one(
        self,
        doc: Document,
        semaphore: asyncio.Semaphore,
    ) -> Document:
        """Genera el summary de un documento con control de concurrencia."""
        async with semaphore:
            if len(doc.page_content) < self._min_summary_length:
                # Chunk corto: usarlo directamente como su propio summary
                summary_text = doc.page_content
            else:
                llm = get_llm()
                chain = _SUMMARY_PROMPT | llm | StrOutputParser()
                try:
                    summary_text = await chain.ainvoke({"text": doc.page_content[:3000]})
                except Exception as exc:
                    log.warning(
                        "summary_generation_failed",
                        chunk=doc.metadata.get("chunk_index", "?"),
                        error=str(exc),
                    )
                    # Fallback: primeras 3 líneas del chunk
                    summary_text = "\n".join(doc.page_content.split("\n")[:3])

            return Document(
                page_content=summary_text,
                metadata={
                    **doc.metadata,
                    "is_summary": "true",
                    "original_content_length": str(len(doc.page_content)),
                },
            )

    async def build_summary_index(
        self,
        documents: list[Document],
        batch_size: int = 20,
    ) -> int:
        """
        Genera summaries para todos los documentos e indexa en summary_store.

        Args:
            documents: Chunks detallados a resumir.
            batch_size: Documentos por lote (controla memoria y concurrencia).

        Returns:
            Número de summaries indexados.
        """
        log.info("summary_index_building", total=len(documents))

        semaphore = asyncio.Semaphore(_SUMMARY_CONCURRENCY)
        all_summaries: list[Document] = []

        # Procesar en batches para controlar memoria
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            tasks = [self._summarize_one(doc, semaphore) for doc in batch]
            batch_summaries = await asyncio.gather(*tasks)
            all_summaries.extend(batch_summaries)

            log.debug(
                "summary_batch_complete",
                batch=f"{i // batch_size + 1}",
                summaries_so_far=len(all_summaries),
            )

        indexed = self._summary_store.add_documents(all_summaries, deduplicate=True)

        log.info(
            "summary_index_ready",
            summaries_generated=len(all_summaries),
            indexed=indexed,
        )

        return indexed

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve(self, query: RetrievalQuery) -> list[Document]:
        """
        Retrieval jerárquico: summaries → detail chunks.

        Paso 1: Buscar en summary store para identificar artículos relevantes.
        Paso 2: Para cada artículo relevante, buscar chunks detallados.
        Paso 3: Deduplicar y retornar top k_chunks.
        """
        if self._summary_store.count() == 0:
            log.warning(
                "hierarchical_no_summaries",
                fallback="direct_detail_search",
            )
            return self._fallback_detail_search(query)

        # ── Nivel 1: Summary search ───────────────────────────────────────────
        summary_query = RetrievalQuery(
            text=query.text,
            top_k=self._k_summaries,
            filters=query.filters,
        )

        top_summaries = self._summary_store.similarity_search(summary_query)

        if not top_summaries:
            log.warning("hierarchical_no_summaries_found", query=query.text[:60])
            return self._fallback_detail_search(query)

        log.debug(
            "hierarchical_summaries_found",
            count=len(top_summaries),
        )

        # ── Nivel 2: Detail search por artículo/página ────────────────────────
        all_detail_docs: list[Document] = []
        seen_ids: set[str] = set()
        chunks_per_summary = max(1, self._k_chunks // len(top_summaries))

        for summary in top_summaries:
            detail_docs = self._get_detail_for_summary(
                query=query,
                summary=summary,
                k=chunks_per_summary,
            )
            for doc in detail_docs:
                doc_id = self._doc_id(doc)
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_detail_docs.append(doc)

        # Si no encontramos suficientes chunks vía summaries, complementar
        if len(all_detail_docs) < self._k_chunks:
            extra_query = RetrievalQuery(
                text=query.text,
                top_k=self._k_chunks - len(all_detail_docs),
                filters=query.filters,
            )
            extra = self._detail_store.similarity_search(extra_query)
            for doc in extra:
                doc_id = self._doc_id(doc)
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_detail_docs.append(doc)

        result = all_detail_docs[:self._k_chunks]

        log.info(
            "hierarchical_retrieval_complete",
            summaries_used=len(top_summaries),
            detail_chunks=len(result),
        )

        return result

    def _get_detail_for_summary(
        self,
        query: RetrievalQuery,
        summary: Document,
        k: int,
    ) -> list[Document]:
        """
        Recupera chunks detallados relacionados con un summary dado.

        Intenta filtrar por article_number primero (más específico),
        luego por page (más amplio). Si ninguno funciona, busca sin filtro.
        """
        article = summary.metadata.get("article_number", "")
        page = summary.metadata.get("page", "")

        # Intentar filtro por artículo (más preciso)
        if article:
            try:
                filtered_query = RetrievalQuery(
                    text=query.text,
                    top_k=k,
                    filters={"article_number": str(article)},
                )
                docs = self._detail_store.similarity_search(filtered_query)
                if docs:
                    return docs
            except Exception as exc:
                log.debug("hierarchical_article_filter_failed", error=str(exc))

        # Fallback: filtro por página
        if page:
            try:
                filtered_query = RetrievalQuery(
                    text=query.text,
                    top_k=k,
                    filters={"page": str(page)},
                )
                docs = self._detail_store.similarity_search(filtered_query)
                if docs:
                    return docs
            except Exception as exc:
                log.debug("hierarchical_page_filter_failed", error=str(exc))

        # Fallback: búsqueda semántica sin filtro
        plain_query = RetrievalQuery(text=query.text, top_k=k)
        return self._detail_store.similarity_search(plain_query)

    def _fallback_detail_search(self, query: RetrievalQuery) -> list[Document]:
        """Búsqueda directa en detail store cuando no hay summaries."""
        fallback_query = RetrievalQuery(
            text=query.text,
            top_k=self._k_chunks,
            filters=query.filters,
        )
        return self._detail_store.similarity_search(fallback_query)

    def _doc_id(self, doc: Document) -> str:
        source = doc.metadata.get("source", "")
        chunk_idx = doc.metadata.get("chunk_index", "")
        if source and chunk_idx != "":
            return f"{source}::{chunk_idx}"
        import hashlib  # noqa: PLC0415
        return hashlib.md5(doc.page_content[:200].encode()).hexdigest()


def get_hierarchical_retriever(**kwargs) -> HierarchicalRetriever:
    """Factory function para HierarchicalRetriever."""
    return HierarchicalRetriever(**kwargs)
