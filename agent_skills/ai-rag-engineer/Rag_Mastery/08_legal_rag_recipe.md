# RAG 08 — Receta para Documentos Estructurados: Pipeline Completo de Producción

## Tu Problema Exacto + Solución Exacta

```
❌ Problema 1: PDFs con texto nativo + escaneados mezclados
❌ Problema 2: Secciones que ocupan 2+ páginas → chunking fijo las parte
❌ Problema 3: Sub-ítems (a, b, c) separados de la sección padre
❌ Problema 4: El LLM no sabe a qué sección pertenece un inciso

✅ Solución: SmartPDFLoader → HierarchicalChunker → ParentDocumentRetriever
             → HybridSearch → CrossEncoder Rerank → Rethinking Generate
```

---

## Pipeline Completo de Producción

```python
# legal_rag/pipeline.py
"""
Pipeline RAG completo para documentos legales en español.
Resuelve: artículos multi-página, sub-ítems, OCR mixto.
"""
from pathlib import Path
from typing import TypedDict, Annotated
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.storage import InMemoryByteStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Importar módulos de esta skill
import sys
sys.path.insert(0, ".")
from loaders.pdf_native_loader import SmartPDFLoader
from chunking.hierarchical_chunker import LegalHierarchicalChunker
from reranking.cross_encoder_reranker import build_reranking_retriever


# ─── Estado del Agente ────────────────────────────────────────────────────────

class LegalRAGState(TypedDict):
    messages: Annotated[list, add_messages]
    retrieved_docs: list[dict]
    sources_cited: list[str]
    session_id: str


# ─── ETAPA 1: Ingesta de Documentos ─────────────────────────────────────────

class LegalDocumentIngester:
    """
    Ingesta completa de documentos legales:
    Load → Hierarchical Chunk → Index (Parent-Child + BM25)
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "legal_docs",
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self._parent_store = InMemoryByteStore()
        self._bm25_docs: list[Document] = []

        self._vectorstore = Chroma(
            collection_name=f"{collection_name}_children",
            embedding_function=self._embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def ingest_pdf(self, pdf_path: str | Path) -> dict:
        """
        Procesa un PDF legal completo:
        1. Carga con SmartPDFLoader (nativo + OCR)
        2. Parsea jerarquía legal (capítulos, artículos, incisos)
        3. Indexa con Parent-Child: busca por inciso, retorna artículo
        """
        pdf_path = Path(pdf_path)
        print(f"\n📄 Ingesting: {pdf_path.name}")

        # Paso 1: Cargar PDF
        loader = SmartPDFLoader(pdf_path)
        raw_pages = loader.load()
        print(f"   ✅ Loaded: {len(raw_pages)} páginas")

        # Paso 2: Parsear jerarquía legal
        chunker = LegalHierarchicalChunker(source_name=pdf_path.stem)
        hierarchical_docs = chunker.chunk(raw_pages)
        print(f"   ✅ Hierarchical chunks: {len(hierarchical_docs)}")

        # Paso 3: Separar artículos completos (padres) de incisos (hijos)
        parent_docs = [d for d in hierarchical_docs if d.metadata.get("level") == "articulo"]
        child_docs = [d for d in hierarchical_docs if d.metadata.get("level") == "inciso"]

        # Si no hay incisos separados, usar los artículos directamente
        if not child_docs:
            child_docs = parent_docs

        # Paso 4: Indexar en Chroma (chunks pequeños para búsqueda precisa)
        self._vectorstore.add_documents(child_docs)

        # Paso 5: BM25 (todos los docs para búsqueda exacta)
        self._bm25_docs.extend(hierarchical_docs)

        stats = {
            "file": pdf_path.name,
            "pages": len(raw_pages),
            "articles": len(parent_docs),
            "chunks_indexed": len(child_docs),
        }
        print(f"   📊 Stats: {stats}")
        return stats

    def build_retriever(self, k: int = 5):
        """Construye el retriever híbrido post-ingesta."""
        dense = self._vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k * 3, "fetch_k": k * 5, "lambda_mult": 0.6},
        )
        sparse = BM25Retriever.from_documents(self._bm25_docs, k=k * 2)

        from retrieval.hybrid_retriever import HybridRetriever
        hybrid = HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            k=k * 2,
        )
        # CrossEncoder reranking encima del hybrid
        return build_reranking_retriever(hybrid, top_n=k)


# ─── ETAPA 2: Query Enhancement ──────────────────────────────────────────────

LEGAL_QUERY_ENHANCER = """Eres un asistente que transforma consultas para búsqueda óptima.
Dado el historial de conversación, reformula la pregunta del usuario para:
1. Usar terminología técnica formal del dominio
2. Incluir el número de sección o identificador si se puede inferir del contexto
3. Ser auto-contenida (no dependa del historial)

Historial: {history}
Pregunta actual: {question}

Pregunta reformulada (solo la pregunta, sin explicaciones):"""


def enhance_query(query: str, history: list, llm) -> str:
    """Mejora la query considerando el historial de conversación."""
    if not history:
        return query
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:100]}"
        for m in history[-4:]
    )
    enhanced = llm.invoke(
        LEGAL_QUERY_ENHANCER.format(history=history_text, question=query)
    ).content
    return enhanced.strip()


# ─── ETAPA 3: Generation con Rethinking ──────────────────────────────────────

LEGAL_SYSTEM_PROMPT = """Eres un asistente experto y preciso.

REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE basándote en los documentos proporcionados en el contexto.
2. SIEMPRE cita la fuente exacta: "Según la Sección X..." o "El Capítulo X, inciso b) establece..."
3. Si la información NO está en el contexto, responde: "Esta información no se encuentra en los documentos disponibles."
4. NUNCA inventes o extrapoles más allá del texto.
5. Si hay ambigüedad, indica las distintas interpretaciones posibles.

Formato de respuesta:
- Respuesta directa a la pregunta
- Cita textual o parafraseada del documento
- Número de sección/fuente y, si aplica, inciso
- Advertencia de limitaciones si aplica"""


def generate_with_rethinking(
    query: str,
    documents: list[Document],
    llm,
) -> tuple[str, list[str]]:
    """
    Genera respuesta con Re-Reading:
    1ra lectura: identifica pasajes clave
    2da lectura: genera respuesta citando fuentes
    """
    context = "\n\n".join(
        f"[{doc.metadata.get('breadcrumb', '')} — {doc.metadata.get('title', '')}]\n{doc.page_content}"
        for doc in documents
    )

    # Primera lectura: identificar pasajes relevantes
    key_passages = llm.invoke([
        SystemMessage(content="Eres un experto leyendo documentos para encontrar información relevante."),
        HumanMessage(content=(
            f"Consulta: {query}\n\n"
            f"Lee estos documentos e identifica los pasajes EXACTOS y números de sección que responden la consulta:\n\n{context}"
        )),
    ]).content

    # Segunda lectura: generar respuesta final
    final_response = llm.invoke([
        SystemMessage(content=LEGAL_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Consulta: {query}\n\n"
            f"Pasajes relevantes identificados:\n{key_passages}\n\n"
            f"Contexto completo de referencia:\n{context}"
        )),
    ]).content

    # Extraer secciones citadas para metadata
    import re
    cited = re.findall(r"[Ss]ecci[óo]n\s+(\d+[\w]*)", final_response)
    sources = list(set(cited))

    return final_response, sources


# ─── ETAPA 4: Agente RAG Completo con LangGraph ──────────────────────────────

def build_legal_rag_agent(retriever, persist_dir: str = "./chroma_db"):
    """
    Agente RAG completo con:
    - Query enhancement con historial
    - Hybrid retrieval + CrossEncoder reranking
    - Rethinking generation
    - Memoria conversacional persistente
    - Source tracking
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    memory = MemorySaver()

    def agent_node(state: LegalRAGState) -> dict:
        messages = state["messages"]
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )
        if not last_human:
            return {}

        query = last_human.content
        history = [m for m in messages[:-1] if not isinstance(m, SystemMessage)]

        print(f"\n🔍 Query: {query[:60]}")

        # Paso 1: Mejorar query con contexto del historial
        enhanced_query = enhance_query(query, history, llm)
        if enhanced_query != query:
            print(f"   ✏️  Enhanced: {enhanced_query[:60]}")

        # Paso 2: Retrieve
        docs = retriever.invoke(enhanced_query)
        print(f"   📚 Retrieved: {len(docs)} documentos")

        if not docs:
            return {
                "messages": [AIMessage(content="No encontré artículos relevantes en los documentos disponibles para esta consulta.")],
                "retrieved_docs": [],
                "sources_cited": [],
            }

        # Paso 3: Generar con Rethinking
        response, sources = generate_with_rethinking(query, docs, llm)
        print(f"   ✅ Generated | Artículos citados: {sources}")

        return {
            "messages": [AIMessage(content=response)],
            "retrieved_docs": [
                {
                    "content": d.page_content[:200],
                    "article": d.metadata.get("article_number"),
                    "source": d.metadata.get("source"),
                }
                for d in docs
            ],
            "sources_cited": sources,
        }

    graph = StateGraph(LegalRAGState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    return graph.compile(checkpointer=memory)


# ─── MAIN: Ejemplo de uso completo ───────────────────────────────────────────

def main():
    """
    Ejemplo de uso completo del pipeline legal.
    """
    # 1. Ingestar documentos
    ingester = LegalDocumentIngester(persist_dir="./legal_chroma_db")
    ingester.ingest_pdf("./docs/codigo_laboral.pdf")
    ingester.ingest_pdf("./docs/reglamento_interno.pdf")

    # 2. Construir retriever
    retriever = ingester.build_retriever(k=5)

    # 3. Construir agente
    agent = build_legal_rag_agent(retriever)

    # 4. Conversar
    config = {"configurable": {"thread_id": "session-001"}}

    queries = [
        "¿Qué obligaciones tiene el empleador respecto a vacaciones?",
        "¿Y cuál es el plazo para pagarlas?",  # Query dependiente del historial
        "¿Qué dice el Artículo 45?",            # Query con número exacto
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"👤 {query}")
        result = agent.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "retrieved_docs": [],
                "sources_cited": [],
                "session_id": "session-001",
            },
            config=config,
        )
        last_ai = next(
            m for m in reversed(result["messages"]) if isinstance(m, AIMessage)
        )
        print(f"🤖 {last_ai.content[:300]}...")
        print(f"📎 Fuentes: {result['sources_cited']}")


if __name__ == "__main__":
    main()
```

---

## Checklist de Calidad para RAG Profesional

```
INGESTA
  □ SmartPDFLoader detecta páginas OCR automáticamente
  □ HierarchicalChunker preserva Capítulo → Sección → Sub-sección
  □ Cada sub-sección tiene en su metadata: section_number, chapter, breadcrumb
  □ Secciones multi-página no se parten (el chunker usa estructura, no tokens)

RETRIEVAL
  □ BM25 para búsqueda exacta de números de sección
  □ Dense para búsqueda semántica de conceptos
  □ Hybrid RRF combina ambos
  □ CrossEncoder reranking filtra top-5

GENERACIÓN
  □ System prompt instruye a citar secciones específicas
  □ Rethinking realiza 2 lecturas del contexto
  □ Respuesta incluye número de sección/fuente
  □ "No encontrado" explícito cuando no hay información

MEMORIA
  □ MemorySaver persiste historial por thread_id
  □ Query enhancement considera turnos previos
  □ LongTermMemory guarda conclusiones importantes

EVALUACIÓN (RAGAS)
  □ Faithfulness: respuesta fundamentada en docs
  □ Answer Relevancy: respuesta responde la pregunta
  □ Context Precision: docs recuperados son relevantes
  □ Context Recall: se recuperó todo lo necesario
```

---

## Diagnóstico de Ingesta

```python
# Verificar que las secciones se parsearon correctamente
def debug_ingestion(ingester: LegalDocumentIngester, pdf_path: str):
    """Herramienta de debugging para verificar la ingesta."""
    loader = SmartPDFLoader(pdf_path)
    pages = loader.load()
    chunker = LegalHierarchicalChunker()
    chunks = chunker.chunk(pages)

    print(f"\n{'='*50}")
    print(f"DIAGNÓSTICO DE INGESTA: {pdf_path}")
    print(f"{'='*50}")
    print(f"Páginas cargadas: {len(pages)}")
    print(f"Chunks generados: {len(chunks)}")

    secciones = [c for c in chunks if c.metadata.get("level") == "seccion"]
    subsecciones = [c for c in chunks if c.metadata.get("level") == "subseccion"]

    print(f"Secciones detectadas: {len(secciones)}")
    print(f"Sub-secciones detectadas: {len(subsecciones)}")

    print("\nPrimeras 3 secciones:")
    for sec in secciones[:3]:
        print(f"  Sec. {sec.metadata.get('section_number')} — {sec.metadata.get('section_title', '')[:50]}")
        print(f"  Breadcrumb: {sec.metadata.get('breadcrumb', 'N/A')}")
        print(f"  Sub-secciones: {len([s for s in subsecciones if s.metadata.get('parent_section') == sec.metadata.get('section_number')])}")
        print()
```
