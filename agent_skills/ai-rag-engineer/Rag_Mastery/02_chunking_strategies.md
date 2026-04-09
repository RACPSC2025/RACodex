# RAG 02 — Chunking Strategies: El Paso Más Crítico

## Regla de Oro
> Si el chunking falla, todo lo demás falla.
> Para documentos estructurados: **NUNCA uses fixed-size chunking.**
> Una sección es la unidad mínima de contexto.

---

## 1. Hierarchical Chunking — Tu Solución Principal para Documentos Estructurados

```
Documento
  └── Capítulo I: Disposiciones Generales       ← Nivel 1 (parent)
        └── Artículo 5. De las obligaciones     ← Nivel 2 (parent)
              ├── 5.a) El empleador deberá...   ← Nivel 3 (leaf — lo que se indexa)
              ├── 5.b) Los trabajadores...      ← Nivel 3 (leaf)
              └── 5.c) En caso de incumplimiento... ← Nivel 3 (leaf)
```

```python
# chunking/hierarchical_chunker.py
import re
from dataclasses import dataclass, field
from langchain_core.documents import Document


@dataclass
class LegalNode:
    """Nodo en la jerarquía de un documento."""
    node_id: str
    level: int          # 0=Documento, 1=Capítulo, 2=Artículo, 3=Inciso
    title: str
    content: str
    parent_id: str | None
    children_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def full_context(self) -> str:
        """Texto completo con título para contexto."""
        return f"{self.title}\n{self.content}".strip()


class LegalHierarchicalChunker:
    """
    Chunker especializado para documentos estructurados en español.
    Preserva la jerarquía: Capítulo → Sección → Sub-sección.

    Problema que resuelve:
    - Secciones partidas a mitad entre chunks.
    - Sub-ítems (a, b, c) separados de su sección padre.
    - Pérdida de referencia al número de sección.
    """

    # Patrones regex para estructura legal española/latinoamericana
    PATTERNS = {
        "capitulo":  re.compile(r"^(CAP[ÍI]TULO|Capítulo)\s+([\w\s]+)", re.MULTILINE | re.IGNORECASE),
        "seccion":   re.compile(r"^(SECCIÓN|Sección)\s+([\w\s]+)", re.MULTILINE | re.IGNORECASE),
        "articulo":  re.compile(r"^(Art[íi]culo|ARTÍCULO|Art\.)\s+(\d+[\w]*)[.\-–]?\s*(.*)", re.MULTILINE),
        "inciso":    re.compile(r"^\s*([a-zA-Z]\)|\d+\)|\d+\.)\s+(.+)", re.MULTILINE),
        "parrafo":   re.compile(r"^\s*§\s*(\d+)", re.MULTILINE),
    }

    def __init__(self, source_name: str = "") -> None:
        self.source_name = source_name
        self._nodes: dict[str, LegalNode] = {}
        self._counter = 0

    def chunk(self, documents: list[Document]) -> list[Document]:
        """
        Procesa documentos y retorna chunks con jerarquía preservada.
        Cada chunk leaf lleva en su metadata el contexto de sus padres.
        """
        full_text = "\n".join(doc.page_content for doc in documents)
        source = documents[0].metadata.get("source", self.source_name) if documents else ""

        nodes = self._parse_legal_structure(full_text, source)
        return self._nodes_to_documents(nodes)

    def _parse_legal_structure(self, text: str, source: str) -> list[LegalNode]:
        """Parsea el texto y construye el árbol jerárquico."""
        lines = text.split("\n")
        nodes: list[LegalNode] = []

        doc_node = LegalNode(
            node_id="doc_0",
            level=0,
            title=source or "Documento",
            content="",
            parent_id=None,
        )
        self._nodes["doc_0"] = doc_node
        nodes.append(doc_node)

        current_chapter: LegalNode | None = None
        current_article: LegalNode | None = None
        article_lines: list[str] = []

        def flush_article() -> None:
            """Cierra y registra el artículo actual."""
            if current_article and article_lines:
                raw = "\n".join(article_lines).strip()
                current_article.content = raw
                self._extract_incisos(current_article, raw, source)
            article_lines.clear()

        for line in lines:
            # ── Detectar Capítulo ──────────────────────────────────────────
            cap_match = self.PATTERNS["capitulo"].match(line)
            if cap_match:
                flush_article()
                current_article = None
                node_id = self._next_id("cap")
                current_chapter = LegalNode(
                    node_id=node_id,
                    level=1,
                    title=line.strip(),
                    content="",
                    parent_id="doc_0",
                    metadata={"source": source, "level": "capitulo"},
                )
                self._nodes[node_id] = current_chapter
                doc_node.children_ids.append(node_id)
                nodes.append(current_chapter)
                continue

            # ── Detectar Artículo ──────────────────────────────────────────
            art_match = self.PATTERNS["articulo"].match(line)
            if art_match:
                flush_article()
                art_num = art_match.group(2)
                art_title = art_match.group(3).strip()
                node_id = self._next_id("art")
                parent_id = current_chapter.node_id if current_chapter else "doc_0"

                current_article = LegalNode(
                    node_id=node_id,
                    level=2,
                    title=f"Artículo {art_num}. {art_title}",
                    content="",
                    parent_id=parent_id,
                    metadata={
                        "source": source,
                        "article_number": art_num,
                        "article_title": art_title,
                        "level": "articulo",
                        "chapter": current_chapter.title if current_chapter else "",
                    },
                )
                self._nodes[node_id] = current_article
                if current_chapter:
                    current_chapter.children_ids.append(node_id)
                nodes.append(current_article)
                article_lines = [line]
                continue

            # ── Acumular líneas del artículo actual ───────────────────────
            if current_article is not None:
                article_lines.append(line)

        flush_article()  # Cerrar último artículo
        return nodes

    def _extract_incisos(
        self, article_node: LegalNode, text: str, source: str
    ) -> None:
        """
        Extrae incisos (a, b, c...) como nodos hijos del artículo.
        Cada inciso hereda la metadata del artículo padre.
        """
        inciso_pattern = re.compile(
            r"(?:^|\n)\s*([a-zA-Z]\)|\d+\))\s+(.+?)(?=\n\s*[a-zA-Z]\)|\n\s*\d+\)|\Z)",
            re.DOTALL,
        )

        incisos = inciso_pattern.findall(text)
        if not incisos:
            return  # Artículo sin incisos — se usa completo

        for marker, content in incisos:
            node_id = self._next_id("inc")
            inciso_node = LegalNode(
                node_id=node_id,
                level=3,
                title=f"{article_node.title} — inciso {marker}",
                content=content.strip(),
                parent_id=article_node.node_id,
                metadata={
                    **article_node.metadata,
                    "inciso": marker,
                    "level": "inciso",
                    "parent_article": article_node.metadata.get("article_number", ""),
                    "parent_title": article_node.title,
                },
            )
            self._nodes[node_id] = inciso_node
            article_node.children_ids.append(node_id)

    def _nodes_to_documents(self, nodes: list[LegalNode]) -> list[Document]:
        """
        Convierte nodos a Documents para indexar.
        ESTRATEGIA: Indexa leaf nodes (incisos o artículos sin incisos)
        pero incluye contexto del padre en el page_content.
        """
        documents: list[Document] = []

        for node in nodes:
            # Solo indexar hojas (incisos) o artículos sin hijos
            if node.level == 0 or node.level == 1:
                continue  # No indexar documento ni capítulo directamente

            is_leaf = len(node.children_ids) == 0
            if not is_leaf and node.level == 2:
                continue  # Artículo con incisos: los incisos se indexan por separado

            # Construir chunk con contexto jerárquico completo
            breadcrumb = self._build_breadcrumb(node)
            chunk_content = f"{breadcrumb}\n\n{node.full_context}"

            documents.append(Document(
                page_content=chunk_content,
                metadata={
                    **node.metadata,
                    "node_id": node.node_id,
                    "node_level": node.level,
                    "parent_id": node.parent_id or "",
                    "breadcrumb": breadcrumb,
                    "chunk_type": "inciso" if node.level == 3 else "articulo",
                },
            ))

        return documents

    def _build_breadcrumb(self, node: LegalNode) -> str:
        """Construye ruta jerárquica: Capítulo I > Artículo 5 > inciso a)"""
        parts: list[str] = []
        current_id = node.parent_id

        while current_id and current_id in self._nodes:
            parent = self._nodes[current_id]
            if parent.level > 0:  # Omitir nivel documento
                parts.insert(0, parent.title)
            current_id = parent.parent_id

        return " > ".join(parts) if parts else ""

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"
```

---

## 2. Parent-Child Retrieval — Indexa Pequeño, Retorna Grande

```python
# chunking/parent_child_retrieval.py
"""
ESTRATEGIA CLAVE para documentos legales:
- Index:   chunks pequeños (incisos, oraciones) para búsqueda precisa
- Retrieve: chunk PADRE completo (artículo) para contexto al LLM

Esto resuelve: "recupero el inciso b) pero el LLM necesita ver el artículo completo"
"""
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def build_parent_child_retriever(
    parent_docs: list[Document],
    collection_name: str = "legal_docs",
) -> ParentDocumentRetriever:
    """
    Construye un retriever Parent-Child para documentos legales.

    Flujo:
    1. parent_docs = artículos completos (contexto completo)
    2. child_splitter divide cada artículo en incisos pequeños
    3. Se indexan los incisos → búsqueda precisa
    4. Al recuperar, se retorna el artículo padre completo → contexto al LLM
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Vector store para chunks pequeños (búsqueda)
    vectorstore = Chroma(
        collection_name=f"{collection_name}_children",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    # Store para documentos padre completos
    parent_store = InMemoryByteStore()

    # Splitter para chunks hijos (pequeños, precisos)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=parent_store,
        child_splitter=child_splitter,
        # parent_splitter=None → usa los docs tal como vienen (artículos completos)
    )

    retriever.add_documents(parent_docs)
    return retriever
```

---

## 3. Semantic Chunking — Corta por Cambio de Tema

```python
# chunking/semantic_chunker.py
"""
Semantic Chunking detecta cambios de significado en el texto
y corta ahí, en lugar de cortar por número fijo de tokens.
"""
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


def semantic_chunk_documents(
    documents: list[Document],
    breakpoint_type: str = "percentile",  # o "standard_deviation", "interquartile"
    breakpoint_threshold: float = 95.0,
) -> list[Document]:
    """
    Divide documentos por cambio semántico.
    breakpoint_threshold: percentil de similitud para cortar (95 = solo corta en cambios grandes)
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_type,
        breakpoint_threshold_amount=breakpoint_threshold,
    )

    all_chunks: list[Document] = []
    for doc in documents:
        chunks = chunker.create_documents(
            [doc.page_content],
            metadatas=[doc.metadata],
        )
        all_chunks.extend(chunks)

    return all_chunks
```

---

## 4. Proposition Chunking — Máxima Precisión

```python
# chunking/proposition_chunker.py
"""
Descompone cada chunk en proposiciones atómicas (una idea = un chunk).
Máxima precisión para Q&A factual. Más costoso (requiere LLM).
Ideal para: citas de artículos, fechas, montos, obligaciones específicas.
"""
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class PropositionList(BaseModel):
    propositions: list[str] = Field(
        description="Lista de proposiciones atómicas extraídas del texto."
    )


PROPOSITION_PROMPT = """Descompón el siguiente texto legal en proposiciones atómicas.
Cada proposición debe:
1. Ser una sola idea completa e independiente
2. Ser comprensible sin el texto original
3. Preservar números de artículo, fechas, partes y obligaciones exactas
4. Estar en español claro

Texto:
{text}"""


def proposition_chunk(
    documents: list[Document],
    model: str = "gpt-4o-mini",
) -> list[Document]:
    """Divide documentos en proposiciones atómicas usando LLM."""
    llm = ChatOpenAI(model=model, temperature=0).with_structured_output(PropositionList)
    proposition_docs: list[Document] = []

    for doc in documents:
        result: PropositionList = llm.invoke(
            PROPOSITION_PROMPT.format(text=doc.page_content)
        )
        for i, prop in enumerate(result.propositions):
            proposition_docs.append(Document(
                page_content=prop,
                metadata={
                    **doc.metadata,
                    "proposition_index": i,
                    "source_chunk": doc.page_content[:100],
                    "chunk_type": "proposition",
                },
            ))

    return proposition_docs
```

---

## Comparativa de Estrategias

| Estrategia | Precisión | Costo | Velocidad | Mejor para |
|-----------|-----------|-------|-----------|------------|
| Fixed-size | ❌ Baja | 🟢 Cero | 🟢 Rápido | Nunca en documentos estructurados |
| Recursive | 🟡 Media | 🟢 Cero | 🟢 Rápido | Texto simple sin estructura |
| Semantic | 🟡 Alta | 🟡 Embeddings | 🟡 Medio | Texto narrativo |
| **Hierarchical** | ✅ **Muy Alta** | 🟢 Cero | 🟢 Rápido | **Documentos estructurados** ✅ |
| **Parent-Child** | ✅ **Muy Alta** | 🟢 Cero | 🟢 Rápido | **Secciones con sub-ítems** ✅ |
| Proposition | ✅ Máxima | 🔴 Alto (LLM) | 🔴 Lento | Citas exactas de artículos |

## Recomendación para tu caso

```
PDFs Legales (tu situación):
  1. SmartPDFLoader → texto por página
  2. LegalHierarchicalChunker → árbol Capítulo/Artículo/Inciso
  3. ParentDocumentRetriever → indexa incisos, retorna artículos
  4. Opcional: PropositionChunker en artículos clave para máxima precisión
```
