"""
Tests unitarios para el módulo de retrieval.

Todos los tests mockean Chroma, BM25 y FlashRank para:
  - No requerir GPU ni colecciones reales
  - Ejecución rápida en CI
  - Tests deterministas y repetibles

Qué se testea:
  - RetrievalQuery: validación y construcción
  - RetrievalResult: propiedades y métodos
  - BaseRetriever: timing, logging, manejo de errores
  - VectorStore: deduplicación de IDs, filtros de metadata
  - BM25Retriever: tokenización, filtros de metadata
  - HybridRetriever: fusión RRF correcta
  - Reranker: degradación graceful sin FlashRank
  - EnsembleRetriever: selección automática de estrategia
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.retrieval.base import (
    BaseRetriever,
    RetrievalError,
    RetrievalQuery,
    RetrievalResult,
    RetrieverProtocol,
    VectorStoreNotInitializedError,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_doc(
    content: str = "texto legal",
    source: str = "decreto.pdf",
    chunk_index: int = 0,
    article: str = "",
    page: str = "1",
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "chunk_index": str(chunk_index),
            "article_number": article,
            "page": page,
            "loader": "pymupdf",
        },
    )


def make_docs(n: int, base_source: str = "test.pdf") -> list[Document]:
    return [
        make_doc(f"chunk {i} con texto legal relevante", base_source, i)
        for i in range(n)
    ]


class StubRetriever(BaseRetriever):
    """Retriever stub para tests de BaseRetriever."""

    def __init__(self, docs: list[Document], ready: bool = True) -> None:
        self._docs = docs
        self._ready = ready

    @property
    def retriever_type(self) -> str:
        return "stub"

    def is_ready(self) -> bool:
        return self._ready

    def _retrieve(self, query: RetrievalQuery) -> list[Document]:
        return self._docs[:query.top_k]


class FailingRetriever(BaseRetriever):
    @property
    def retriever_type(self) -> str:
        return "failing"

    def is_ready(self) -> bool:
        return True

    def _retrieve(self, query: RetrievalQuery) -> list[Document]:
        raise RuntimeError("error simulado")


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: RetrievalQuery
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrievalQuery:
    def test_basic_construction(self) -> None:
        q = RetrievalQuery(text="¿Qué es el COPASST?")
        assert q.text == "¿Qué es el COPASST?"
        assert q.top_k == 10

    def test_text_is_stripped(self) -> None:
        q = RetrievalQuery(text="  texto con espacios  ")
        assert q.text == "texto con espacios"

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValueError):
            RetrievalQuery(text="")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError):
            RetrievalQuery(text="   ")

    def test_filters_default_empty(self) -> None:
        q = RetrievalQuery(text="query")
        assert q.filters == {}

    def test_custom_top_k(self) -> None:
        q = RetrievalQuery(text="query", top_k=5)
        assert q.top_k == 5

    def test_filters_set(self) -> None:
        q = RetrievalQuery(text="query", filters={"source": "decreto.pdf"})
        assert q.filters["source"] == "decreto.pdf"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: RetrievalResult
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrievalResult:
    def test_count_property(self) -> None:
        r = RetrievalResult(query="q", documents=make_docs(5))
        assert r.count == 5

    def test_is_empty_true(self) -> None:
        r = RetrievalResult(query="q")
        assert r.is_empty is True

    def test_is_empty_false(self) -> None:
        r = RetrievalResult(query="q", documents=make_docs(1))
        assert r.is_empty is False

    def test_top_n_returns_slice(self) -> None:
        r = RetrievalResult(query="q", documents=make_docs(10))
        assert len(r.top(3)) == 3

    def test_repr_contains_retriever(self) -> None:
        r = RetrievalResult(query="q", retriever_used="hybrid_rrf", documents=make_docs(3))
        assert "hybrid_rrf" in repr(r)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: BaseRetriever
# ═══════════════════════════════════════════════════════════════════════════════

class TestBaseRetriever:
    def test_retrieve_returns_result_object(self) -> None:
        retriever = StubRetriever(make_docs(5))
        query = RetrievalQuery(text="SST empleadores")
        result = retriever.retrieve(query)
        assert isinstance(result, RetrievalResult)

    def test_retrieve_respects_top_k(self) -> None:
        retriever = StubRetriever(make_docs(10))
        query = RetrievalQuery(text="query", top_k=3)
        result = retriever.retrieve(query)
        assert result.count == 3

    def test_retrieve_measures_elapsed_time(self) -> None:
        retriever = StubRetriever(make_docs(3))
        result = retriever.retrieve(RetrievalQuery(text="query"))
        assert result.elapsed_ms >= 0

    def test_retrieve_sets_retriever_type(self) -> None:
        retriever = StubRetriever(make_docs(3))
        result = retriever.retrieve(RetrievalQuery(text="query"))
        assert result.retriever_used == "stub"

    def test_not_ready_raises_error(self) -> None:
        retriever = StubRetriever(make_docs(3), ready=False)
        with pytest.raises(VectorStoreNotInitializedError):
            retriever.retrieve(RetrievalQuery(text="query"))

    def test_retriever_exception_wrapped(self) -> None:
        retriever = FailingRetriever()
        with pytest.raises(RetrievalError) as exc_info:
            retriever.retrieve(RetrievalQuery(text="query"))
        assert "failing" in str(exc_info.value)

    def test_protocol_satisfied(self) -> None:
        retriever = StubRetriever(make_docs(1))
        assert isinstance(retriever, RetrieverProtocol)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: VectorStore
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorStore:
    @pytest.fixture
    def mock_chroma(self):
        """Mock completo de Chroma para tests sin DB real."""
        with patch("src.retrieval.vector_store.Chroma") as mock_cls:
            mock_store = MagicMock()
            mock_store._collection.count.return_value = 0
            mock_cls.return_value = mock_store
            yield mock_cls, mock_store

    @pytest.fixture
    def mock_embeddings(self):
        with patch("src.retrieval.vector_store.get_embeddings") as mock:
            mock.return_value = MagicMock()
            yield mock

    def test_open_or_create_initializes_store(
        self, mock_chroma, mock_embeddings, tmp_path
    ) -> None:
        from src.retrieval.vector_store import VectorStore

        store = VectorStore(
            collection_name="test_col",
            persist_directory=tmp_path / "chroma",
        )
        store.open_or_create()

        assert store.is_initialized is True
        mock_chroma[0].assert_called_once()

    def test_open_or_create_idempotent(
        self, mock_chroma, mock_embeddings, tmp_path
    ) -> None:
        from src.retrieval.vector_store import VectorStore

        store = VectorStore(
            collection_name="test_col",
            persist_directory=tmp_path / "chroma",
        )
        store.open_or_create()
        store.open_or_create()  # segunda llamada no debe re-crear

        mock_chroma[0].assert_called_once()

    def test_make_ids_deterministic(
        self, mock_chroma, mock_embeddings, tmp_path
    ) -> None:
        from src.retrieval.vector_store import VectorStore

        store = VectorStore(persist_directory=tmp_path / "chroma")
        docs = [make_doc("content", "decreto.pdf", i) for i in range(3)]
        ids1 = store._make_ids(docs)
        ids2 = store._make_ids(docs)

        assert ids1 == ids2
        assert len(set(ids1)) == 3  # todos únicos

    def test_make_ids_format(
        self, mock_chroma, mock_embeddings, tmp_path
    ) -> None:
        from src.retrieval.vector_store import VectorStore

        store = VectorStore(persist_directory=tmp_path / "chroma")
        doc = make_doc("content", "decreto_1072.pdf", 5)
        ids = store._make_ids([doc])

        assert "decreto_1072.pdf" in ids[0]
        assert "5" in ids[0]

    def test_build_filter_single(
        self, mock_chroma, mock_embeddings, tmp_path
    ) -> None:
        from src.retrieval.vector_store import VectorStore

        store = VectorStore(persist_directory=tmp_path / "chroma")
        f = store._build_filter({"source": "decreto.pdf"})
        assert f == {"source": {"$eq": "decreto.pdf"}}

    def test_build_filter_multiple(
        self, mock_chroma, mock_embeddings, tmp_path
    ) -> None:
        from src.retrieval.vector_store import VectorStore

        store = VectorStore(persist_directory=tmp_path / "chroma")
        f = store._build_filter({"source": "d.pdf", "page": "5"})
        assert "$and" in f
        assert len(f["$and"]) == 2

    def test_build_filter_empty_returns_none(
        self, mock_chroma, mock_embeddings, tmp_path
    ) -> None:
        from src.retrieval.vector_store import VectorStore

        store = VectorStore(persist_directory=tmp_path / "chroma")
        assert store._build_filter({}) is None

    def test_require_store_raises_if_not_init(self, tmp_path) -> None:
        from src.retrieval.vector_store import VectorStore

        store = VectorStore(persist_directory=tmp_path / "chroma")
        with pytest.raises(VectorStoreNotInitializedError):
            store._require_store()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: BM25Retriever
# ═══════════════════════════════════════════════════════════════════════════════

class TestBM25Retriever:
    def test_retriever_type(self) -> None:
        from src.retrieval.bm25_retriever import BM25Retriever
        r = BM25Retriever(use_spacy=False)
        assert r.retriever_type == "bm25"

    def test_is_ready_with_vector_store(self) -> None:
        from src.retrieval.bm25_retriever import BM25Retriever
        mock_vs = MagicMock()
        r = BM25Retriever(vector_store=mock_vs, use_spacy=False)
        assert r.is_ready() is True

    def test_tokenize_without_spacy(self) -> None:
        from src.retrieval.bm25_retriever import BM25Retriever
        r = BM25Retriever(use_spacy=False)
        tokens = r.tokenize("El empleador debe cumplir las normas SST")
        # Stopwords eliminadas, palabras relevantes preservadas
        assert "empleador" in tokens
        assert "sst" in tokens
        assert "el" not in tokens
        assert "las" not in tokens

    def test_tokenize_filters_short_tokens(self) -> None:
        from src.retrieval.bm25_retriever import BM25Retriever
        r = BM25Retriever(use_spacy=False)
        tokens = r.tokenize("a b c de trabajo")
        # Tokens de 1 char eliminados
        assert "a" not in tokens
        assert "b" not in tokens

    def test_tokenize_empty_returns_empty(self) -> None:
        from src.retrieval.bm25_retriever import BM25Retriever
        r = BM25Retriever(use_spacy=False)
        assert r.tokenize("") == []

    def test_build_index_from_documents(self) -> None:
        from src.retrieval.bm25_retriever import BM25Retriever

        r = BM25Retriever(use_spacy=False, cache_index=False)
        docs = make_docs(10)
        r.build_index(documents=docs)

        assert r._bm25 is not None
        assert len(r._corpus_docs) == 10

    def test_retrieve_returns_relevant_docs(self) -> None:
        from src.retrieval.bm25_retriever import BM25Retriever

        docs = [
            make_doc("COPASST comité paritario SST empleadores", "d.pdf", 0),
            make_doc("Decreto obligaciones contrato laboral vigencia", "d.pdf", 1),
            make_doc("ARL accidente trabajo pensión invalidez", "d.pdf", 2),
        ]
        r = BM25Retriever(use_spacy=False, cache_index=False)
        r.build_index(documents=docs)

        query = RetrievalQuery(text="COPASST SST", top_k=2)
        result = r.retrieve(query)

        assert result.count >= 1
        # El primer resultado debe ser el más relevante (COPASST)
        assert "COPASST" in result.documents[0].page_content

    def test_retrieve_with_metadata_filter(self) -> None:
        from src.retrieval.bm25_retriever import BM25Retriever

        docs = [
            make_doc("texto decreto 1072", "decreto.pdf", 0),
            make_doc("texto resolución 2400", "resolucion.pdf", 1),
        ]
        r = BM25Retriever(use_spacy=False, cache_index=False)
        r.build_index(documents=docs)

        query = RetrievalQuery(
            text="texto",
            top_k=5,
            filters={"source": "decreto.pdf"},
        )
        result = r.retrieve(query)

        # Solo debe retornar docs de decreto.pdf
        assert all(d.metadata["source"] == "decreto.pdf" for d in result.documents)

    def test_bm25_score_injected_in_metadata(self) -> None:
        from src.retrieval.bm25_retriever import BM25Retriever

        r = BM25Retriever(use_spacy=False, cache_index=False)
        r.build_index(documents=make_docs(5))

        query = RetrievalQuery(text="chunk 0 texto legal", top_k=3)
        result = r.retrieve(query)

        for doc in result.documents:
            assert "bm25_score" in doc.metadata

    def test_invalidate_clears_index(self) -> None:
        from src.retrieval.bm25_retriever import BM25Retriever

        r = BM25Retriever(use_spacy=False, cache_index=False)
        r.build_index(documents=make_docs(3))
        assert r._bm25 is not None

        r.invalidate_index()
        assert r._bm25 is None
        assert r._corpus_docs == []


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: HybridRetriever (RRF)
# ═══════════════════════════════════════════════════════════════════════════════

class TestHybridRetriever:
    def _make_hybrid(self, vector_docs: list[Document], bm25_docs: list[Document]):
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.bm25_retriever import BM25Retriever

        mock_vs = MagicMock()
        mock_vs.is_initialized = True
        mock_vs.similarity_search.return_value = vector_docs

        mock_bm25 = MagicMock(spec=BM25Retriever)
        mock_bm25.retriever_type = "bm25"
        mock_bm25.is_ready.return_value = True
        mock_bm25_result = MagicMock()
        mock_bm25_result.documents = bm25_docs
        mock_bm25.retrieve.return_value = mock_bm25_result

        return HybridRetriever(
            vector_store=mock_vs,
            bm25_retriever=mock_bm25,
        )

    def test_retriever_type(self) -> None:
        from src.retrieval.hybrid_retriever import HybridRetriever
        h = HybridRetriever(vector_store=MagicMock(is_initialized=True))
        assert h.retriever_type == "hybrid_rrf"

    def test_rrf_score_decreases_with_rank(self) -> None:
        from src.retrieval.hybrid_retriever import HybridRetriever
        h = HybridRetriever(vector_store=MagicMock(is_initialized=True), rrf_k=60)
        assert h._rrf_score(1) > h._rrf_score(5) > h._rrf_score(10)

    def test_fuse_deduplicates_same_doc(self) -> None:
        """Un doc que aparece en ambas listas debe acumular score, no duplicarse."""
        from src.retrieval.hybrid_retriever import HybridRetriever

        doc_shared = make_doc("contenido compartido", "d.pdf", 0)
        doc_unique_v = make_doc("solo vector", "d.pdf", 1)
        doc_unique_b = make_doc("solo bm25", "d.pdf", 2)

        h = HybridRetriever(vector_store=MagicMock(is_initialized=True))
        fused = h._fuse_with_rrf(
            vector_docs=[doc_shared, doc_unique_v],
            bm25_docs=[doc_shared, doc_unique_b],
            top_k=5,
        )

        contents = [d.page_content for d in fused]
        assert contents.count("contenido compartido") == 1

    def test_fuse_shared_doc_gets_higher_score(self) -> None:
        """Doc en ambas listas debe rankear más alto que docs exclusivos."""
        from src.retrieval.hybrid_retriever import HybridRetriever

        doc_shared = make_doc("compartido", "d.pdf", 0)
        doc_only_vector = make_doc("solo vector", "d.pdf", 1)

        h = HybridRetriever(vector_store=MagicMock(is_initialized=True))
        fused = h._fuse_with_rrf(
            vector_docs=[doc_shared, doc_only_vector],
            bm25_docs=[doc_shared],
            top_k=3,
        )

        # El compartido debe ser el primero
        assert fused[0].page_content == "compartido"

    def test_rrf_score_in_metadata(self) -> None:
        from src.retrieval.hybrid_retriever import HybridRetriever

        docs = make_docs(3)
        h = HybridRetriever(vector_store=MagicMock(is_initialized=True))
        fused = h._fuse_with_rrf(vector_docs=docs, bm25_docs=[], top_k=3)

        for doc in fused:
            assert "rrf_score" in doc.metadata

    def test_retrieve_with_mocked_stores(self) -> None:
        vector_docs = make_docs(5, "v.pdf")
        bm25_docs = make_docs(5, "b.pdf")
        h = self._make_hybrid(vector_docs, bm25_docs)

        query = RetrievalQuery(text="SST decreto empleadores", top_k=5)
        result = h.retrieve(query)

        assert result.count > 0
        assert result.retriever_used == "hybrid_rrf"

    def test_retrieve_fallback_when_bm25_fails(self) -> None:
        from src.retrieval.hybrid_retriever import HybridRetriever

        mock_vs = MagicMock()
        mock_vs.is_initialized = True
        mock_vs.similarity_search.return_value = make_docs(3)

        mock_bm25 = MagicMock()
        mock_bm25.retriever_type = "bm25"
        mock_bm25.is_ready.return_value = True
        mock_bm25.retrieve.side_effect = Exception("BM25 falló")

        h = HybridRetriever(vector_store=mock_vs, bm25_retriever=mock_bm25)
        result = h.retrieve(RetrievalQuery(text="query", top_k=3))

        # Debe retornar resultados del vector aunque BM25 falle
        assert result.count > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Reranker
# ═══════════════════════════════════════════════════════════════════════════════

class TestReranker:
    def test_reranker_available_false_without_flashrank(self) -> None:
        from src.retrieval.reranker import Reranker

        r = Reranker()
        with patch("builtins.__import__", side_effect=lambda name, *a, **k: (
            (_ for _ in ()).throw(ImportError("flashrank not found"))
            if name == "flashrank"
            else __import__(name, *a, **k)
        )):
            r._available = None
            r._ranker = None
            _ = r._get_ranker()

        assert r.is_available is False

    def test_rerank_returns_subset_when_no_ranker(self) -> None:
        from src.retrieval.reranker import Reranker

        r = Reranker()
        r._available = False  # simular no disponible

        docs = make_docs(10)
        result = r.rerank("query SST", docs, top_k=3)

        assert len(result) == 3
        assert result == docs[:3]  # retorna sin cambios

    def test_rerank_empty_docs_returns_empty(self) -> None:
        from src.retrieval.reranker import Reranker

        r = Reranker()
        result = r.rerank("query", [], top_k=5)
        assert result == []

    def test_rerank_with_mocked_flashrank(self) -> None:
        from src.retrieval.reranker import Reranker

        r = Reranker()
        docs = make_docs(5)

        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": 2, "score": 0.95},
            {"id": 0, "score": 0.80},
            {"id": 4, "score": 0.72},
        ]
        r._ranker = mock_ranker
        r._available = True

        with patch("src.retrieval.reranker.RerankRequest", MagicMock()):
            result = r.rerank("SST COPASST", docs, top_k=3)

        assert len(result) == 3
        # El doc con índice 2 debe ser primero (score más alto)
        assert result[0].page_content == docs[2].page_content
        assert "rerank_score" in result[0].metadata
        assert result[0].metadata["rerank_score"] == 0.95

    @pytest.mark.asyncio
    async def test_arerank_runs_in_thread(self) -> None:
        from src.retrieval.reranker import Reranker

        r = Reranker()
        r._available = False  # degradación graceful

        docs = make_docs(5)
        result = await r.arerank("query", docs, top_k=3)

        assert len(result) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: EnsembleRetriever - selección de estrategia
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnsembleRetriever:
    def _make_ensemble(self, strategy: str = "hybrid"):
        from src.retrieval.ensemble import EnsembleRetriever

        mock_vs = MagicMock()
        mock_vs.is_initialized = True
        mock_vs.similarity_search.return_value = make_docs(5)

        ensemble = EnsembleRetriever(
            vector_store=mock_vs,
            strategy=strategy,
            use_reranking=False,
            top_k=5,
        )
        return ensemble, mock_vs

    def test_auto_strategy_article_number(self) -> None:
        from src.retrieval.ensemble import EnsembleRetriever, RetrievalStrategy

        e, _ = self._make_ensemble("auto")
        q = RetrievalQuery(text="¿Qué dice el artículo 2.2.4.6.1?", top_k=5)
        strategy = e._select_strategy(q)
        assert strategy == RetrievalStrategy.HYBRID

    def test_auto_strategy_short_query(self) -> None:
        from src.retrieval.ensemble import EnsembleRetriever, RetrievalStrategy

        e, _ = self._make_ensemble("auto")
        q = RetrievalQuery(text="COPASST SST", top_k=5)
        strategy = e._select_strategy(q)
        assert strategy == RetrievalStrategy.HYBRID

    def test_auto_strategy_long_query(self) -> None:
        from src.retrieval.ensemble import EnsembleRetriever, RetrievalStrategy

        e, _ = self._make_ensemble("auto")
        long_query = "cuáles son las obligaciones del empleador en materia " \
                     "de seguridad salud trabajo frente al sistema gestión SST " \
                     "según decreto 1072 de 2015 Colombia"
        q = RetrievalQuery(text=long_query, top_k=5)
        strategy = e._select_strategy(q)
        assert strategy == RetrievalStrategy.FULL

    def test_retriever_type_includes_strategy(self) -> None:
        from src.retrieval.ensemble import EnsembleRetriever

        e, _ = self._make_ensemble("hybrid")
        assert "hybrid" in e.retriever_type

    def test_retrieve_returns_result(self) -> None:
        e, mock_vs = self._make_ensemble("vector")

        # Simular el método _execute_strategy vía vector
        result = e.retrieve(RetrievalQuery(text="SST", top_k=3))

        assert isinstance(result, RetrievalResult)
        assert result.count > 0

    def test_evaluate_returns_metrics(self) -> None:
        e, mock_vs = self._make_ensemble("vector")

        eval_data = [
            {"query": "COPASST", "relevant_chunks": ["0", "1"]},
            {"query": "empleador obligaciones", "relevant_chunks": ["2"]},
        ]

        metrics = e.evaluate(eval_data)

        assert "hit_rate" in metrics
        assert "mrr" in metrics
        assert "total_queries" in metrics
        assert metrics["total_queries"] == 2
