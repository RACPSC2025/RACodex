# RAG 05 — Reranking & Filtering: Cross-Encoder, MMR, Compression

## Regla de Oro
> El retriever tiene alto recall pero bajo precision.
> El reranker tiene alta precision. Úsalos juntos: retrieve muchos, rerank pocos.
> Pipeline: retrieve 20 → rerank → top 5 al LLM

---

## 1. Cross-Encoder Reranking (Local, Sin Costo de API)

```python
# reranking/cross_encoder_reranker.py
"""
Cross-Encoder evalúa pares (query, doc) conjuntamente.
Más preciso que bi-encoders pero más lento → úsalo post-retrieval.
Modelos recomendados:
- "cross-encoder/ms-marco-MiniLM-L-6-v2" → inglés, rápido
- "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1" → multilingüe
- "BAAI/bge-reranker-v2-m3" → multilingüe, state-of-art
"""
from langchain_core.documents import Document
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever


def build_reranking_retriever(
    base_retriever,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    top_n: int = 5,
):
    """
    Envuelve cualquier retriever con Cross-Encoder reranking.
    Retrieve muchos → rerank → retorna top_n más relevantes.
    """
    cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )
```

---

## 2. Cohere Rerank (API — Máxima Calidad)

```python
# reranking/cohere_reranker.py
"""
Cohere Rerank es el mejor reranker disponible como API.
Soporta español, multilingüe, documentos largos.
Costo: ~$0.002 por 1000 docs rerankeados.
"""
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document


def build_cohere_reranker(
    base_retriever,
    top_n: int = 5,
    model: str = "rerank-multilingual-v3.0",  # Soporta español
):
    """Reranker con API de Cohere."""
    reranker = CohereRerank(
        model=model,
        top_n=top_n,
        cohere_api_key="...",  # Desde env: COHERE_API_KEY
    )
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )
```

---

## 3. Question Relevance Filter (LLM-based)

```python
# reranking/relevance_filter.py
"""
Usa el LLM para filtrar documentos que NO son relevantes para la query.
Más costoso pero más inteligente que filtros por score.
"""
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document


class RelevanceScore(BaseModel):
    relevant: bool = Field(description="True si el documento es relevante para la query.")
    score: float = Field(ge=0.0, le=1.0, description="Score de relevancia 0-1.")
    reasoning: str = Field(description="Por qué es o no es relevante.")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def filter_relevant_documents(
    query: str,
    documents: list[Document],
    threshold: float = 0.5,
) -> list[Document]:
    """
    Filtra documentos irrelevantes usando el LLM como juez.
    Retorna solo documentos con score >= threshold.
    """
    relevant_docs: list[Document] = []

    for doc in documents:
        score: RelevanceScore = llm.with_structured_output(RelevanceScore).invoke(
            f"""¿Es este fragmento legal relevante para responder la consulta?

            Consulta: {query}

            Fragmento:
            {doc.page_content[:500]}"""
        )

        if score.relevant and score.score >= threshold:
            # Añadir score de relevancia a metadata para debugging
            doc.metadata["relevance_score"] = score.score
            doc.metadata["relevance_reasoning"] = score.reasoning
            relevant_docs.append(doc)

    relevant_docs.sort(key=lambda d: d.metadata.get("relevance_score", 0), reverse=True)
    print(f"   🔍 Relevance filter: {len(relevant_docs)}/{len(documents)} docs pasaron")
    return relevant_docs
```

---

## 4. Contextual Compression — Extrae Solo lo Relevante

```python
# reranking/contextual_compression.py
"""
En lugar de retornar el chunk completo, extrae SOLO las oraciones
relevantes para la query. Reduce ruido en el contexto del LLM.
Muy útil cuando los chunks son artículos legales largos.
"""
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document


def build_compression_retriever(base_retriever):
    """
    Retriever que comprime cada documento recuperado a solo las
    partes relevantes para la query.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


# ─── Pipeline Completo Recomendado ────────────────────────────────────────────

def build_production_retriever(
    documents: list[Document],
    collection_name: str = "legal_docs",
    k_retrieve: int = 15,   # Recuperar muchos
    k_final: int = 5,       # Retornar pocos al LLM
):
    """
    Pipeline de retrieval completo para producción:
    HybridRetriever(k=15) → CrossEncoder Rerank → top 5

    retrieve 15 → rerank con cross-encoder → top 5 al LLM
    """
    from retrieval.hybrid_retriever import build_hybrid_retriever
    from reranking.cross_encoder_reranker import build_reranking_retriever

    # Base: Hybrid retrieval
    hybrid = build_hybrid_retriever(documents, collection_name, k=k_retrieve)

    # Reranking encima del hybrid
    return build_reranking_retriever(hybrid, top_n=k_final)
```

---

## 5. MMR — Diversidad en los Resultados

```python
# reranking/mmr.py
"""
Maximal Marginal Relevance evita retornar 5 chunks casi idénticos.
Balancea relevancia con diversidad.
Muy útil cuando hay artículos con texto similar o redundante.
"""
from langchain_chroma import Chroma
from langchain_core.documents import Document


def mmr_retrieve(
    vectorstore: Chroma,
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,  # 0=máx diversidad, 1=máx relevancia
) -> list[Document]:
    """
    lambda_mult=0.5: balance entre relevancia y diversidad.
    Para documentos legales con mucho contenido similar, usa 0.3-0.4.
    """
    return vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )
```

---

## Stack de Reranking Recomendado

```
Producción (sin costo extra):
  HybridRetriever(k=20) → CrossEncoder bge-reranker-v2-m3 → top 5

Producción (máxima calidad):
  HybridRetriever(k=20) → Cohere Rerank multilingual → top 5

Para corpus legales con artículos repetitivos:
  HybridRetriever(k=20) → MMR(lambda=0.4) → CrossEncoder → top 5
```
