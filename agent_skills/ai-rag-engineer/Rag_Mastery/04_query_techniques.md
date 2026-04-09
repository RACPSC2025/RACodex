# RAG 04 — Query Enhancement: HyDE, Self-Query, Transformations, Fusion, Adaptive

## Regla de Oro
> La query del usuario NUNCA es la query óptima para el retriever.
> Siempre transforma, enriquece o expande antes de buscar.

---

## 1. HyDE — Hypothetical Document Embeddings

```python
# query/hyde.py
"""
HyDE genera un documento hipotético que RESPONDERÍA a la query,
luego busca documentos similares a ESE documento (no a la query original).

¿Por qué funciona? El espacio de embeddings de una respuesta es más
cercano al espacio de documentos que el de una pregunta.

Ideal para: queries abstractas, conceptuales o vagas.
"""
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma


HYDE_PROMPT = """Eres un experto en derecho. Escribe un fragmento de artículo legal
(2-3 párrafos) que respondería directamente a esta consulta.
Usa lenguaje técnico-legal formal. NO respondas la pregunta, escribe
COMO SI FUERA un artículo de ley que la respondería.

Consulta: {query}

Fragmento hipotético de ley:"""


class HyDERetriever:
    """Retriever con Hypothetical Document Embeddings."""

    def __init__(
        self,
        vectorstore: Chroma,
        k: int = 5,
        model: str = "gpt-4o-mini",
    ) -> None:
        self._vectorstore = vectorstore
        self._k = k
        self._llm = ChatOpenAI(model=model, temperature=0.3)
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    def retrieve(self, query: str) -> list[Document]:
        # Paso 1: Generar documento hipotético
        hypothetical_doc = self._llm.invoke(
            HYDE_PROMPT.format(query=query)
        ).content
        print(f"   📝 HyDE doc: {hypothetical_doc[:80]}...")

        # Paso 2: Embeddear el documento hipotético
        hyp_embedding = self._embeddings.embed_query(hypothetical_doc)

        # Paso 3: Buscar por similitud con ese embedding
        results = self._vectorstore.similarity_search_by_vector(
            hyp_embedding, k=self._k
        )
        return results
```

---

## 2. Self-Query — Filtrado por Metadata

```python
# query/self_query.py
"""
Self-Query permite al LLM extraer FILTROS de la query natural.
"Artículos del Capítulo III sobre obligaciones" →
  filter: {chapter: "Capítulo III"} + query: "obligaciones"

Fundamental para corpus legales con metadata rica.
"""
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma


# Define la metadata disponible para filtrado
LEGAL_METADATA_FIELDS = [
    AttributeInfo(
        name="article_number",
        description="Número del artículo legal (ej: '45', '102')",
        type="string",
    ),
    AttributeInfo(
        name="chapter",
        description="Capítulo del documento (ej: 'Capítulo I', 'Capítulo III')",
        type="string",
    ),
    AttributeInfo(
        name="level",
        description="Nivel jerárquico: 'articulo', 'inciso', 'capitulo'",
        type="string",
    ),
    AttributeInfo(
        name="doc_type",
        description="Tipo de documento: 'legal', 'reglamento', 'decreto'",
        type="string",
    ),
    AttributeInfo(
        name="source",
        description="Nombre del archivo o ley fuente",
        type="string",
    ),
]

DOCUMENT_DESCRIPTION = """Artículos y disposiciones de documentos legales.
Contiene capítulos, artículos e incisos de leyes y reglamentos."""


def build_self_query_retriever(
    vectorstore: Chroma,
    k: int = 5,
) -> SelfQueryRetriever:
    """
    Construye un retriever que extrae filtros automáticamente de la query.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    return SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=DOCUMENT_DESCRIPTION,
        metadata_field_info=LEGAL_METADATA_FIELDS,
        search_kwargs={"k": k},
        verbose=True,
        # Permite filtros complejos (AND, OR, comparaciones)
        structured_query_translator=None,  # Auto-detecta el translator para Chroma
    )

# Ejemplo de queries que genera filtros automáticos:
# "¿Qué dice el Artículo 45 sobre plazos?"
#   → filter: {article_number: "45"} + semantic query: "plazos"
# "Dame todos los artículos del Capítulo III"
#   → filter: {chapter: "Capítulo III"} + sin query semántica
```

---

## 3. Query Transformations

```python
# query/transformations.py
"""
Transforma la query original en versiones optimizadas para retrieval.
Técnicas:
- Query Rewriting: reformula para mayor precisión
- Query Decomposition: divide queries complejas en sub-queries
- Step-Back: genera una query más general para contexto amplio
- Query Expansion: añade sinónimos y términos relacionados
"""
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document


# ─── Schemas ──────────────────────────────────────────────────────────────────

class RewrittenQuery(BaseModel):
    rewritten: str = Field(description="Query reformulada para búsqueda en documentos legales.")
    reasoning: str = Field(description="Por qué esta reformulación mejora el retrieval.")


class DecomposedQueries(BaseModel):
    sub_queries: list[str] = Field(
        description="Lista de sub-queries atómicas que juntas responden la query original.",
        min_length=1,
        max_length=5,
    )


class ExpandedQuery(BaseModel):
    original: str
    synonyms: list[str] = Field(description="Sinónimos y términos equivalentes en lenguaje legal.")
    related_terms: list[str] = Field(description="Términos relacionados del dominio legal.")
    expanded_query: str = Field(description="Query expandida con términos adicionales.")


llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ─── Query Rewriting ──────────────────────────────────────────────────────────

def rewrite_query(query: str) -> str:
    """Reformula la query para mejor retrieval en corpus legal."""
    result: RewrittenQuery = llm.with_structured_output(RewrittenQuery).invoke(
        f"""Reformula esta consulta para búsqueda óptima en un corpus legal.
        Usa terminología técnico-legal formal. Incluye sinónimos legales relevantes.
        
        Query original: {query}"""
    )
    return result.rewritten


# ─── Query Decomposition ──────────────────────────────────────────────────────

def decompose_query(query: str) -> list[str]:
    """Divide una query compleja en sub-queries atómicas."""
    result: DecomposedQueries = llm.with_structured_output(DecomposedQueries).invoke(
        f"""Descompón esta consulta legal compleja en sub-preguntas simples y atómicas.
        Cada sub-pregunta debe poder responderse con un único artículo o inciso.
        
        Query: {query}"""
    )
    return result.sub_queries


# ─── Step-Back Prompting ──────────────────────────────────────────────────────

def step_back_query(query: str) -> str:
    """
    Genera una query más general para recuperar contexto de fondo.
    Luego ambas queries (original + step-back) se usan para retrieval.
    """
    step_back = llm.invoke(
        f"""Genera una pregunta más general que proporcione contexto para responder:
        "{query}"
        
        La pregunta general debe cubrir el marco legal o principio que aplica.
        Solo escribe la pregunta general, sin explicaciones."""
    ).content
    return step_back


# ─── Query Expansion ──────────────────────────────────────────────────────────

def expand_query(query: str) -> ExpandedQuery:
    """Expande la query con sinónimos y términos del dominio legal."""
    return llm.with_structured_output(ExpandedQuery).invoke(
        f"""Expande esta consulta legal con términos equivalentes y relacionados.
        Considera terminología legal formal, informal y latinismos comunes.
        
        Query: {query}"""
    )


# ─── Pipeline Combinado ───────────────────────────────────────────────────────

def multi_transform_query(query: str) -> list[str]:
    """
    Pipeline completo: genera múltiples variantes de la query.
    Retorna lista de queries para Fusion Retrieval.
    """
    queries = [query]  # Query original siempre incluida

    # Reescritura
    try:
        queries.append(rewrite_query(query))
    except Exception:
        pass

    # Step-back para contexto
    try:
        queries.append(step_back_query(query))
    except Exception:
        pass

    # Deduplicar manteniendo orden
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)

    return unique
```

---

## 4. Fusion Retrieval con RRF Multi-Query

```python
# query/fusion_retrieval.py
"""
Genera N variantes de la query, hace retrieval con cada una,
y combina con RRF. Dramáticamente mejor recall que una sola query.
"""
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from collections import defaultdict


class FusionRetriever:
    """
    Multi-Query Fusion con RRF.
    1. Genera N variantes de la query
    2. Hace retrieval con cada variante
    3. Combina resultados con RRF
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        n_queries: int = 3,
        k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        self._retriever = retriever
        self._n_queries = n_queries
        self._k = k
        self._rrf_k = rrf_k

    def retrieve(self, query: str) -> list[Document]:
        from query.transformations import multi_transform_query
        queries = multi_transform_query(query)[:self._n_queries]
        print(f"   🔀 Fusion: {len(queries)} queries generadas")

        all_results: list[list[Document]] = []
        for q in queries:
            results = self._retriever.invoke(q)
            all_results.append(results)

        return self._rrf_fusion(all_results)

    def _rrf_fusion(self, results_lists: list[list[Document]]) -> list[Document]:
        scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}

        for results in results_lists:
            for rank, doc in enumerate(results):
                doc_id = doc.metadata.get("node_id") or doc.page_content[:80]
                scores[doc_id] += 1 / (self._rrf_k + rank + 1)
                doc_map[doc_id] = doc

        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [doc_map[did] for did in ranked[:self._k]]
```

---

## 5. Adaptive Retrieval

```python
# query/adaptive_retrieval.py
"""
Selecciona dinámicamente la estrategia de retrieval según la complejidad
y tipo de la query. No todas las queries necesitan el mismo pipeline.
"""
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from enum import Enum


class QueryComplexity(str, Enum):
    SIMPLE = "simple"       # Una fact, un artículo específico
    MODERATE = "moderate"   # Comparación, contexto
    COMPLEX = "complex"     # Multi-documento, análisis


class QueryAnalysis(BaseModel):
    complexity: QueryComplexity
    requires_hyde: bool = Field(description="True si la query es abstracta o conceptual.")
    requires_decomposition: bool = Field(description="True si hay múltiples sub-preguntas.")
    has_article_reference: bool = Field(description="True si menciona un artículo específico.")
    suggested_k: int = Field(ge=3, le=15, description="Número óptimo de chunks a recuperar.")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def adaptive_retrieve(
    query: str,
    retriever,  # HybridRetriever
    hyde_retriever=None,
    fusion_retriever=None,
) -> list[Document]:
    """
    Analiza la query y aplica la estrategia óptima automáticamente.
    """
    analysis: QueryAnalysis = llm.with_structured_output(QueryAnalysis).invoke(
        f"""Analiza esta consulta legal y determina la estrategia de retrieval óptima.
        Query: {query}"""
    )

    print(f"   🎯 Adaptive: complejidad={analysis.complexity}, k={analysis.suggested_k}")

    if analysis.has_article_reference:
        # Query con artículo específico → BM25 directo es suficiente
        print("      → Estrategia: BM25 (artículo específico)")
        return retriever.sparse_retriever.invoke(query)[:analysis.suggested_k]

    if analysis.requires_hyde and hyde_retriever:
        print("      → Estrategia: HyDE (query abstracta)")
        return hyde_retriever.retrieve(query)

    if analysis.requires_decomposition and fusion_retriever:
        print("      → Estrategia: Fusion Multi-Query (query compleja)")
        return fusion_retriever.retrieve(query)

    # Default: Hybrid Search
    print("      → Estrategia: Hybrid (default)")
    return retriever.retrieve(query)
```
