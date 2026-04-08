"""
BM25 Retriever con tokenizador NLP en español.

Por qué BM25 importa en RAG legal:
  Las búsquedas léxicas capturan coincidencias exactas que el embedding
  puede perder: números de artículos ("2.2.4.6.1"), siglas ("SST", "ARL",
  "COPASST"), términos legales poco frecuentes en el corpus de entrenamiento.

  Ejemplo: una query sobre "COPASST" puede tener embedding similar al de
  "comité" pero BM25 va directo al texto que contiene "COPASST".

Por qué spaCy en lugar de .lower().split():
  El código original usaba tokenización naïve. En español:
  - "empleadores" y "empleador" son tokens distintos con split()
  - Con lematización de spaCy → ambos → "empleador"
  - "obligación" y "obligaciones" → "obligación"
  - Stopwords eliminadas: "el", "la", "de", "que" no aportan a BM25

Modelo spaCy recomendado: es_core_news_sm (8 MB, solo tokenizador + lemma)
  python -m spacy download es_core_news_sm

Fallback: si spaCy no está disponible, usa tokenizador simple mejorado
  con lista de stopwords en español hardcodeada.

Indexación lazy:
  El índice BM25 se construye la primera vez que se llama retrieve().
  La construcción carga TODOS los documentos de Chroma en memoria —
  esto es inevitable con BM25 Okapi pero lo hacemos una sola vez.
  Para colecciones > 100K docs, considerar BM25S con memoria mapeada.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.retrieval.base import (
    BaseRetriever,
    RetrievalQuery,
    RetrieverUnavailableError,
    VectorStoreNotInitializedError,
)

log = get_logger(__name__)

# Stopwords en español (mínima lista robusta, sin depender de NLTK)
_ES_STOPWORDS: frozenset[str] = frozenset({
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con",
    "contra", "cual", "cuando", "de", "del", "desde", "donde", "durante",
    "e", "el", "ella", "ellas", "ellos", "en", "entre", "era", "es", "esa",
    "esas", "ese", "eso", "esos", "esta", "estas", "este", "esto", "estos",
    "fue", "hay", "la", "las", "le", "les", "lo", "los", "más", "me", "mi",
    "mismo", "muy", "ni", "no", "nos", "o", "otra", "otras", "otro", "otros",
    "para", "pero", "por", "que", "sea", "ser", "si", "sin", "sobre", "son",
    "su", "sus", "también", "tanto", "te", "tiene", "tienen", "todo", "todos",
    "una", "unas", "uno", "unos", "ya", "y", "yo",
})


class BM25Retriever(BaseRetriever):
    """
    Retriever BM25 con tokenización NLP española.

    Carga todos los documentos del VectorStore al construir el índice.
    El índice puede persistirse en disco para evitar reconstrucción
    en cada reinicio (útil cuando la colección no cambia frecuentemente).
    """

    def __init__(
        self,
        vector_store: Any | None = None,      # VectorStore instance (avoid circular import)
        cache_dir: Path | None = None,
        use_spacy: bool = True,
        cache_index: bool = True,
    ) -> None:
        """
        Args:
            vector_store: Instancia de VectorStore de donde cargar documentos.
                          Si None, se carga al primer retrieve().
            cache_dir: Directorio donde persistir el índice BM25 serializado.
            use_spacy: Si True, intenta usar spaCy es_core_news_sm para
                       lematización. Si False o si spaCy no está disponible,
                       usa tokenizador simple con stopwords.
            cache_index: Si True, serializa el índice en cache_dir para
                         reutilizar en reinicios sin recargar Chroma.
        """
        settings = get_settings()
        self._vector_store = vector_store
        self._cache_dir = cache_dir or settings.bm25_cache_dir
        self._use_spacy = use_spacy
        self._cache_index = cache_index

        # Estado interno — lazy initialized
        self._bm25: Any | None = None
        self._corpus_docs: list[Document] = []
        self._nlp: Any | None = None        # spaCy pipeline
        self._spacy_available: bool | None = None  # None = no comprobado aún

    @property
    def retriever_type(self) -> str:
        return "bm25"

    def is_ready(self) -> bool:
        return self._bm25 is not None or self._vector_store is not None

    # ── Tokenización ──────────────────────────────────────────────────────────

    def _get_nlp(self) -> Any | None:
        """
        Lazy init del pipeline spaCy.

        Intenta cargar es_core_news_sm. Si no está disponible,
        registra el fallback y retorna None.
        """
        if self._spacy_available is not None:
            return self._nlp if self._spacy_available else None

        if not self._use_spacy:
            self._spacy_available = False
            return None

        try:
            import spacy  # noqa: PLC0415
            self._nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
            self._spacy_available = True
            log.info("spacy_loaded", model="es_core_news_sm")
            return self._nlp
        except OSError:
            log.warning(
                "spacy_model_not_found",
                model="es_core_news_sm",
                hint="python -m spacy download es_core_news_sm",
            )
            self._spacy_available = False
            return None
        except ImportError:
            log.info("spacy_not_installed", fallback="simple_tokenizer")
            self._spacy_available = False
            return None

    def tokenize(self, text: str) -> list[str]:
        """
        Tokeniza texto para BM25.

        Con spaCy: lematización + eliminación de stopwords + solo palabras/números.
        Sin spaCy: lowercasing + split + eliminación de stopwords simples.
        """
        if not text:
            return []

        nlp = self._get_nlp()

        if nlp is not None:
            # Procesamiento completo con spaCy
            doc = nlp(text.lower())
            tokens = [
                token.lemma_
                for token in doc
                if (
                    not token.is_stop
                    and not token.is_punct
                    and not token.is_space
                    and len(token.lemma_) > 1
                    and token.lemma_ not in _ES_STOPWORDS
                )
            ]
        else:
            # Tokenizador simple mejorado (fallback)
            import re  # noqa: PLC0415
            words = re.findall(r"\b[a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9\.]+\b", text.lower())
            tokens = [
                w for w in words
                if len(w) > 1 and w not in _ES_STOPWORDS
            ]

        return tokens

    # ── Construcción del índice ────────────────────────────────────────────────

    def _cache_path(self, collection_name: str) -> Path:
        return self._cache_dir / f"bm25_{collection_name}.pkl"

    def _try_load_from_cache(self, collection_name: str) -> bool:
        """Intenta cargar el índice BM25 desde el caché en disco."""
        if not self._cache_index:
            return False

        cache_path = self._cache_path(collection_name)
        if not cache_path.exists():
            return False

        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)  # noqa: S301

            self._bm25 = cached["bm25"]
            self._corpus_docs = cached["docs"]
            log.info(
                "bm25_loaded_from_cache",
                path=str(cache_path),
                docs=len(self._corpus_docs),
            )
            return True
        except Exception as exc:
            log.warning("bm25_cache_load_failed", error=str(exc))
            return False

    def _save_to_cache(self, collection_name: str) -> None:
        """Persiste el índice BM25 en disco."""
        if not self._cache_index:
            return

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path(collection_name)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {"bm25": self._bm25, "docs": self._corpus_docs},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            log.info("bm25_saved_to_cache", path=str(cache_path))
        except Exception as exc:
            log.warning("bm25_cache_save_failed", error=str(exc))

    def build_index(self, documents: list[Document] | None = None) -> None:
        """
        Construye el índice BM25 desde documentos o desde el VectorStore.

        Args:
            documents: Lista de Documents para indexar directamente.
                       Si None, carga desde el VectorStore.
        """
        try:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415
        except ImportError as exc:
            raise RetrieverUnavailableError(
                "rank-bm25 no está instalado. Ejecuta: pip install rank-bm25"
            ) from exc

        if documents is None:
            if self._vector_store is None:
                raise VectorStoreNotInitializedError(
                    "BM25Retriever necesita un VectorStore o una lista de documentos."
                )
            documents = self._load_all_from_store()

        if not documents:
            log.warning("bm25_index_empty_corpus")
            return

        log.info("bm25_building_index", total_docs=len(documents))

        self._corpus_docs = documents
        tokenized_corpus = [
            self.tokenize(doc.page_content)
            for doc in documents
        ]

        self._bm25 = BM25Okapi(tokenized_corpus)

        log.info(
            "bm25_index_ready",
            docs=len(documents),
            using_spacy=self._spacy_available,
        )

    def _load_all_from_store(self) -> list[Document]:
        """Carga todos los documentos del VectorStore para el índice BM25."""
        log.info("bm25_loading_corpus_from_chroma")

        try:
            collection = self._vector_store.get_raw_collection()
            results = collection.get()
        except Exception as exc:
            raise VectorStoreNotInitializedError(
                "No se pudo cargar el corpus desde Chroma para BM25.",
                cause=exc,
            ) from exc

        if not results.get("documents"):
            log.warning("bm25_corpus_empty_in_chroma")
            return []

        docs = [
            Document(
                page_content=text,
                metadata=meta or {},
            )
            for text, meta in zip(
                results["documents"],
                results.get("metadatas") or [{}] * len(results["documents"]),
            )
        ]

        log.info("bm25_corpus_loaded", total=len(docs))
        return docs

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        """Construye el índice si no existe (lazy build en primer retrieve)."""
        if self._bm25 is None:
            # Intentar cargar desde caché primero
            collection_name = "default"
            if self._vector_store:
                collection_name = getattr(
                    self._vector_store, "collection_name", "default"
                )

            if not self._try_load_from_cache(collection_name):
                self.build_index()
                self._save_to_cache(collection_name)

    def _retrieve(self, query: RetrievalQuery) -> list[Document]:
        """
        Búsqueda BM25 sobre el corpus en memoria.

        Retorna los top_k documentos con mayor score BM25.
        """
        import numpy as np  # noqa: PLC0415

        self._ensure_index()

        if self._bm25 is None or not self._corpus_docs:
            return []

        tokenized_query = self.tokenize(query.text)
        if not tokenized_query:
            log.warning("bm25_empty_tokenized_query", original=query.text)
            return []

        scores = self._bm25.get_scores(tokenized_query)

        # Seleccionar los top_k con score > 0
        top_indices = np.argsort(scores)[::-1]
        results = []

        for idx in top_indices:
            if len(results) >= query.top_k:
                break
            score = float(scores[idx])
            if score <= 0:
                break  # BM25 scores restantes serán 0 o negativos

            doc = self._corpus_docs[idx]

            # Aplicar filtros de metadata si existen
            if query.filters and not self._matches_filters(doc, query.filters):
                continue

            # Inyectar score en metadata para ensemble/reranking
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "bm25_score": round(score, 4)},
            )
            results.append(doc_copy)

        return results

    def _matches_filters(self, doc: Document, filters: dict[str, str]) -> bool:
        """Verifica que el documento cumple todos los filtros de metadata."""
        for key, value in filters.items():
            if str(doc.metadata.get(key, "")) != str(value):
                return False
        return True

    def invalidate_index(self) -> None:
        """
        Invalida el índice BM25 en memoria y en disco.

        Llamar cuando se indexan nuevos documentos en Chroma.
        """
        self._bm25 = None
        self._corpus_docs = []

        if self._vector_store and self._cache_index:
            collection_name = getattr(
                self._vector_store, "collection_name", "default"
            )
            cache_path = self._cache_path(collection_name)
            try:
                cache_path.unlink(missing_ok=True)
                log.info("bm25_cache_invalidated", path=str(cache_path))
            except Exception:
                pass


def get_bm25_retriever(
    vector_store: Any | None = None,
    **kwargs,
) -> BM25Retriever:
    """Factory function para BM25Retriever."""
    return BM25Retriever(vector_store=vector_store, **kwargs)
