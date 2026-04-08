"""
Context Enrichment Window — Retrieves neighboring chunks for each retrieved document.

When a chunk is retrieved, this wrapper also fetches the N neighboring chunks
(before and after) from the same source document, concatenating them with
overlap handling to avoid duplicate text.

This solves the most frequent problem in legal RAG: an article is cut in half
because the chunk_size split it right in the middle of paragraph 2.

Usage:
    from src.retrieval.context_enricher import ContextEnrichmentWindow

    enricher = ContextEnrichmentWindow(vector_store, window_size=2)
    enriched_docs = enricher.enrich(retrieved_docs)

Or as a wrapper around any retriever:
    from src.retrieval.context_enricher import EnrichedRetriever

    base_retriever = ensemble_retriever
    enriched = EnrichedRetriever(base_retriever, vector_store, window_size=2)
    results = enriched.retrieve(query)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from langchain_core.documents import Document

from src.config.logging import get_logger
from src.config.settings import get_settings

log = get_logger(__name__)


class ContextEnrichmentWindow:
    """
    Retrieves neighboring chunks for context enrichment.

    For each retrieved chunk, fetches `window_size` neighbors before and after
    from the same source document, concatenating with overlap handling.
    """

    def __init__(
        self,
        vector_store: Any,
        window_size: int = 2,
    ) -> None:
        """
        Args:
            vector_store: Chroma vector store instance (must support get() by ID).
            window_size: Number of neighbors to fetch on each side.
                         1 = prev + current + next (3 chunks total)
                         2 = prev2 + prev1 + current + next1 + next2 (5 chunks total)
        """
        self._vector_store = vector_store
        self._window_size = window_size

        # Get chunk overlap from settings for proper deduplication
        settings = get_settings()
        self._chunk_overlap = settings.chunk_overlap

    def enrich(self, documents: list[Document]) -> list[Document]:
        """
        Enriches each retrieved document with its neighboring context.

        Args:
            documents: List of retrieved chunks.

        Returns:
            List of enriched documents with expanded page_content.
        """
        if not documents:
            return []

        # Group documents by source to fetch neighbors from the same document
        by_source: dict[str, list[Document]] = defaultdict(list)
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            by_source[source].append(doc)

        enriched: list[Document] = []
        seen_ids: set[str] = set()

        for source, docs in by_source.items():
            for doc in docs:
                doc_id = self._doc_unique_id(doc)
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                # Get neighbors for this document
                neighbor_docs = self._get_neighbors(doc, source)

                if not neighbor_docs or len(neighbor_docs) == 1:
                    # No neighbors found or only the original — return as-is
                    enriched.append(doc)
                    continue

                # Concatenate with overlap handling
                combined_text = self._concatenate_with_dedup(neighbor_docs)

                # Create enriched document using the original as base
                enriched_doc = Document(
                    page_content=combined_text,
                    metadata={
                        **doc.metadata,
                        "context_enriched": True,
                        "context_window_size": self._window_size,
                        "context_chunks_count": len(neighbor_docs),
                    },
                )
                enriched.append(enriched_doc)

        log.debug(
            "context_enrichment_complete",
            input_docs=len(documents),
            output_docs=len(enriched),
            window_size=self._window_size,
        )

        return enriched

    def _get_neighbors(
        self,
        target_doc: Document,
        source: str,
    ) -> list[Document]:
        """
        Fetches the target chunk and its neighbors from the same source.

        Uses chunk_index metadata to find adjacent chunks.
        """
        chunk_idx = target_doc.metadata.get("chunk_index")
        if chunk_idx is None:
            return [target_doc]

        # Calculate the range of chunk indices to fetch
        start_idx = max(0, chunk_idx - self._window_size)
        end_idx = chunk_idx + self._window_size + 1

        # Build filter for the same source and index range
        # Chroma filter syntax
        where_filter = {
            "$and": [
                {"source": source},
                {"chunk_index": {"$gte": start_idx, "$lte": end_idx - 1}},
            ]
        }

        try:
            # Fetch all chunks in the range from the same source
            results = self._vector_store.get(
                where=where_filter,
                include=["metadatas", "documents"],
            )

            if not results or not results.get("documents"):
                return [target_doc]

            # Reconstruct Document objects with metadata
            neighbor_docs = []
            metadatas = results.get("metadatas", [])
            contents = results.get("documents", [])

            for content, meta in zip(contents, metadatas):
                neighbor_docs.append(Document(
                    page_content=content,
                    metadata=meta,
                ))

            # Sort by chunk_index to ensure correct order
            neighbor_docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))

            return neighbor_docs

        except Exception as exc:
            log.warning(
                "context_enrichment_neighbor_fetch_failed",
                source=source,
                chunk_idx=chunk_idx,
                error=str(exc),
            )
            return [target_doc]

    def _concatenate_with_dedup(self, docs: list[Document]) -> str:
        """
        Concatenates chunks handling overlap to avoid duplicate text.

        When chunks overlap, the end of chunk N is the same as the start
        of chunk N+1. We detect and remove this overlap.
        """
        if not docs:
            return ""

        if len(docs) == 1:
            return docs[0].page_content

        # Start with the first chunk
        combined = docs[0].page_content

        for i in range(1, len(docs)):
            current_chunk = docs[i].page_content

            # Try to detect overlap at the boundary
            overlap_removed = self._detect_and_remove_overlap(combined, current_chunk)

            if overlap_removed is not None:
                combined = overlap_removed
            else:
                # No overlap detected — just concatenate with separator
                combined = combined + "\n\n" + current_chunk

        return combined

    def _detect_and_remove_overlap(self, text_a: str, text_b: str) -> str | None:
        """
        Detects overlap between the end of text_a and the start of text_b.

        Uses the configured chunk_overlap as the maximum overlap size.
        Returns the concatenated text with overlap removed, or None if no overlap found.
        """
        max_overlap = min(self._chunk_overlap, len(text_a), len(text_b))
        if max_overlap < 10:
            return None  # Too small to be meaningful

        # Check for overlap from max_overlap down to a minimum threshold
        for overlap_size in range(max_overlap, 20, -1):
            end_of_a = text_a[-overlap_size:]
            start_of_b = text_b[:overlap_size]

            if end_of_a == start_of_b:
                return text_a + text_b[overlap_size:]

        return None

    def _doc_unique_id(self, doc: Document) -> str:
        """Creates a unique ID for a document based on source + chunk_index."""
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        return f"{source}::{chunk_idx}"


class EnrichedRetriever:
    """
    Wrapper that adds context enrichment to any existing retriever.

    Usage:
        base_retriever = get_ensemble_retriever(vector_store)
        enriched = EnrichedRetriever(base_retriever, vector_store, window_size=2)
        results = enriched.retrieve(query)
    """

    def __init__(
        self,
        base_retriever: Any,
        vector_store: Any,
        window_size: int = 2,
    ) -> None:
        self._base_retriever = base_retriever
        self._enricher = ContextEnrichmentWindow(vector_store, window_size=window_size)

    def retrieve(self, query: Any) -> Any:
        """
        Retrieves with context enrichment.

        Args:
            query: RetrievalQuery or string.

        Returns:
        RetrievalResult with enriched documents.
        """
        # Get base retrieval results
        base_result = self._base_retriever.retrieve(query)

        # Enrich with context window
        enriched_docs = self._enricher.enrich(base_result.documents)

        # Return same result type with enriched docs
        base_result.documents = enriched_docs
        return base_result


def get_context_enricher(
    vector_store: Any,
    window_size: int = 2,
) -> ContextEnrichmentWindow:
    """
    Factory function for ContextEnrichmentWindow.

    Args:
        vector_store: Chroma vector store instance.
        window_size: Number of neighbors on each side.

    Returns:
        Configured ContextEnrichmentWindow instance.
    """
    return ContextEnrichmentWindow(vector_store, window_size=window_size)
