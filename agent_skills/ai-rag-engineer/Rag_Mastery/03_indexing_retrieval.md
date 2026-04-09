# RAG 03 — Indexing & Retrieval: BM25, Semantic, Hybrid

## Regla de Oro
> Hybrid Search (Dense + Sparse) supera a cualquiera por separado.
> Para documentos estructurados: BM25 encuentra la sección por número exacto,
> Semantic Search encuentra el concepto aunque uses palabras distintas.

---

## 1. Dense Retrieval (Semantic Search)

```python
# retrieval/dense_retriever.py
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


def build_dense_retriever(
    documents: list[Document],
    collection_name: str = "legal_docs",
    persist_dir: str = "./chroma_db",
    k: int = 5,
) -> VectorStoreRetriever:
    """
    Retriever semántico con ChromaDB.
    Encuentra documentos por significado, no por palabras exactas.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},  # Cosine para texto
    )

    return vectorstore.as_retriever(
        search_type="mmr",  # MMR: diversidad + relevancia
        search_kwargs={
            "k": k,
            "fetch_k": k * 3,   # Busca 3x más, filtra por diversidad
            "lambda_mult": 0.7, # 0=máx diversidad, 1=máx relevancia
        },
    )


def load_existing_chroma(
    collection_name: str = "legal_docs",
    persist_dir: str = "./chroma_db",
    k: int = 5,
) -> VectorStoreRetriever:
    """Carga un ChromaDB existente sin re-indexar."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
```

---

## 2. Sparse Retrieval (BM25)

```python
# retrieval/bm25_retriever.py
"""
BM25: algoritmo de recuperación basado en frecuencia de términos.
Excelente para:
- Búsqueda exacta de números de artículo ("Artículo 45")
- Términos legales técnicos específicos
- Nombres propios, fechas, montos
"""
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def build_bm25_retriever(
    documents: list[Document],
    k: int = 5,
) -> BM25Retriever:
    """
    Construye un retriever BM25 en memoria.
    Rápido, sin dependencias externas, ideal para corpus < 100k docs.
    """
    retriever = BM25Retriever.from_documents(
        documents,
        k=k,
        # Preprocesamiento para español legal
        preprocess_func=_preprocess_legal_text,
    )
    return retriever


def _preprocess_legal_text(text: str) -> list[str]:
    """
    Tokenización y normalización para texto legal en español.
    Preserva números de artículo y términos técnicos.
    """
    import re
    # Normalizar: minúsculas, preservar números
    text = text.lower()
    # Tokenizar por espacios y puntuación (preservar compuestos como "art.5")
    tokens = re.findall(r"\b[\w]+\b", text)
    # Eliminar stopwords básicas (no eliminar términos legales)
    stopwords = {"de", "la", "el", "en", "y", "a", "los", "las", "del", "al"}
    return [t for t in tokens if t not in stopwords or len(t) > 4]
```

---

## 3. Hybrid Search con RRF (Reciprocal Rank Fusion)

```python
# retrieval/hybrid_retriever.py
"""
Combina BM25 + Dense con Reciprocal Rank Fusion.
RRF es superior a simple score averaging porque:
- No necesita normalizar scores de distintas escalas
- Robusto ante outliers
- Fórmula: score(d) = Σ 1/(k + rank(d)) donde k=60 por defecto
"""
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field
from collections import defaultdict


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever: BM25 + Dense con Reciprocal Rank Fusion.
    El mejor punto de partida para cualquier RAG profesional.
    """
    dense_retriever: BaseRetriever
    sparse_retriever: BaseRetriever
    k: int = 5
    rrf_k: int = 60          # Constante RRF — 60 es el valor estándar
    dense_weight: float = 0.7   # Peso del retriever semántico
    sparse_weight: float = 0.3  # Peso del BM25

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        # Recuperar de ambas fuentes
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.sparse_retriever.invoke(query)

        return self._reciprocal_rank_fusion(dense_docs, sparse_docs)

    def _reciprocal_rank_fusion(
        self,
        dense_docs: list[Document],
        sparse_docs: list[Document],
    ) -> list[Document]:
        """Aplica RRF para combinar y re-rankear los resultados."""
        scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}

        # Calcular score RRF para dense
        for rank, doc in enumerate(dense_docs):
            doc_id = self._get_doc_id(doc)
            scores[doc_id] += self.dense_weight * (1 / (self.rrf_k + rank + 1))
            doc_map[doc_id] = doc

        # Calcular score RRF para sparse
        for rank, doc in enumerate(sparse_docs):
            doc_id = self._get_doc_id(doc)
            scores[doc_id] += self.sparse_weight * (1 / (self.rrf_k + rank + 1))
            doc_map[doc_id] = doc

        # Ordenar por score final y retornar top-k
        ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in ranked_ids[:self.k]]

    def _get_doc_id(self, doc: Document) -> str:
        """Identificador único por documento."""
        return doc.metadata.get("node_id") or doc.page_content[:100]


def build_hybrid_retriever(
    documents: list[Document],
    collection_name: str = "legal_docs",
    k: int = 5,
) -> HybridRetriever:
    """Factory para crear el hybrid retriever completo."""
    from retrieval.dense_retriever import build_dense_retriever
    from retrieval.bm25_retriever import build_bm25_retriever

    dense = build_dense_retriever(documents, collection_name, k=k * 2)
    sparse = build_bm25_retriever(documents, k=k * 2)

    return HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        k=k,
    )
```

---

## 4. Hierarchical Indices (Multi-nivel)

```python
# retrieval/hierarchical_index.py
"""
Para corpus grandes (múltiples leyes, reglamentos):
1. Índice de resúmenes → encuentra el documento correcto
2. Índice de chunks → encuentra el pasaje específico dentro del doc
"""
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document


class HierarchicalIndexRetriever:
    """
    Retrieval en dos pasos:
    1. Buscar en índice de resúmenes → identificar documento relevante
    2. Buscar chunks dentro de ese documento → precisión máxima
    """

    def __init__(
        self,
        summary_docs: list[Document],       # Un doc por ley/reglamento (resumen)
        chunk_docs: list[Document],          # Chunks detallados
        persist_dir: str = "./chroma_db",
    ) -> None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        self._summary_store = Chroma(
            collection_name="legal_summaries",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        self._chunk_store = Chroma(
            collection_name="legal_chunks",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

        self._summary_store.add_documents(summary_docs)
        self._chunk_store.add_documents(chunk_docs)

    def retrieve(self, query: str, top_docs: int = 2, chunks_per_doc: int = 4) -> list[Document]:
        """
        Retrieval jerárquico:
        1. Encuentra los top_docs documentos más relevantes por resumen
        2. Busca chunks solo dentro de esos documentos
        """
        # Paso 1: Identificar documentos relevantes
        relevant_summaries = self._summary_store.similarity_search(query, k=top_docs)
        relevant_sources = {
            doc.metadata.get("source", "") for doc in relevant_summaries
        }

        # Paso 2: Buscar chunks filtrados por fuente
        all_chunks = self._chunk_store.similarity_search(
            query,
            k=chunks_per_doc * top_docs,
            filter={"source": {"$in": list(relevant_sources)}},
        )

        return all_chunks[:chunks_per_doc * top_docs]


def generate_document_summary(
    documents: list[Document],
    source_name: str,
    model: str = "gpt-4o-mini",
) -> Document:
    """Genera un resumen del documento completo para el índice de resúmenes."""
    llm = ChatOpenAI(model=model, temperature=0)
    full_text = "\n".join(d.page_content for d in documents[:20])  # Primeras 20 páginas

    summary = llm.invoke(
        f"Genera un resumen técnico de este documento legal (máx 500 palabras):\n\n{full_text}"
    ).content

    return Document(
        page_content=summary,
        metadata={
            "source": source_name,
            "doc_type": "summary",
            "total_chunks": len(documents),
        },
    )
```

---

## Cuándo Usar Cada Estrategia

| Estrategia | Caso de uso | Ejemplo |
|-----------|-------------|---------|
| BM25 solo | Buscar por término exacto | "Artículo 45", "plazo de 30 días" |
| Dense solo | Buscar por concepto | "obligaciones del empleador" |
| **Hybrid** | **Caso general — siempre** | Cualquier query en RAG legal |
| Hierarchical | Corpus con múltiples leyes | Código Civil + Laboral + Penal |
