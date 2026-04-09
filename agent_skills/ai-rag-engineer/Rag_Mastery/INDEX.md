# Skill: RAG Mastery — Nivel Dios
**Versión:** 1.0.0 | **Stack:** LangChain · LangGraph · ChromaDB · Python 3.12+

---

## Propósito

Esta skill es la referencia técnica completa para construir sistemas RAG de nivel enterprise.
Cubre desde carga de documentos hasta evaluación continua, con foco en documentos
estructurados complejos (técnicos, multi-página con sub-ítems, tablas, formularios).

**Ante cualquier problema con RAG, consulta esta skill primero.**

---

## Mapa de Técnicas

### 📥 Etapa 1 — Carga de Documentos
| Técnica | Archivo | Cuándo |
|---------|---------|--------|
| PDF nativo (texto seleccionable) | `01_document_loading.md` | Documentos digitales |
| PDF escaneado (OCR) | `01_document_loading.md` | PDFs imagen, documentos viejos |
| Multimodal (tablas, imágenes) | `01_document_loading.md` | PDFs con figuras y tablas |
| Web / HTML scraping | `01_document_loading.md` | Normativa online |

### ✂️ Etapa 2 — Chunking (El más crítico)
| Técnica | Archivo | Cuándo |
|---------|---------|--------|
| Semantic Chunking | `02_chunking_strategies.md` | Texto con cambios de tema |
| Proposition Chunking | `02_chunking_strategies.md` | Alta precisión, Q&A factual |
| Hierarchical Chunking | `02_chunking_strategies.md` | **Documentos estructurados, técnicos** |
| Parent-Child Chunking | `02_chunking_strategies.md` | Sub-ítems con contexto padre |
| Late Chunking | `02_chunking_strategies.md` | Chunks con embeddings contextuales |

### 🔍 Etapa 3 — Indexing & Retrieval Base
| Técnica | Archivo | Cuándo |
|---------|---------|--------|
| Dense (Semantic Search) | `03_indexing_retrieval.md` | Búsqueda por significado |
| Sparse (BM25) | `03_indexing_retrieval.md` | Búsqueda por términos exactos |
| Hybrid (Dense + Sparse) | `03_indexing_retrieval.md` | **Siempre — mejor combinado** |
| Hierarchical Indices | `03_indexing_retrieval.md` | Corpus grandes, multi-doc |
| Multi-vector Retrieval | `03_indexing_retrieval.md` | Documentos con resúmenes |

### 🔄 Etapa 4 — Query Enhancement
| Técnica | Archivo | Cuándo |
|---------|---------|--------|
| HyDE | `04_query_techniques.md` | Queries abstractas / vagas |
| Self-Query | `04_query_techniques.md` | Filtrado por metadata |
| Query Transformations | `04_query_techniques.md` | Rewriting, decomposition |
| Fusion Retrieval (RRF) | `04_query_techniques.md` | Combinar múltiples búsquedas |
| Adaptive Retrieval | `04_query_techniques.md` | Decide estrategia dinámicamente |
| Step-Back Prompting | `04_query_techniques.md` | Preguntas que necesitan contexto amplio |

### 🎯 Etapa 5 — Reranking & Filtering
| Técnica | Archivo | Cuándo |
|---------|---------|--------|
| Cross-Encoder Reranking | `05_reranking_filtering.md` | Siempre post-retrieval |
| Question Relevance Filter | `05_reranking_filtering.md` | Eliminar docs irrelevantes |
| MMR (Diversidad) | `05_reranking_filtering.md` | Evitar redundancia |
| Contextual Compression | `05_reranking_filtering.md` | Chunks muy largos |
| Cohere Rerank | `05_reranking_filtering.md` | API reranking state-of-art |

### 🧠 Etapa 6 — Patrones RAG Avanzados
| Técnica | Archivo | Cuándo |
|---------|---------|--------|
| CRAG (Corrective RAG) | `06_advanced_patterns.md` | Docs de baja calidad / inciertos |
| Self-Retrieval | `06_advanced_patterns.md` | El modelo decide qué buscar |
| Rethinking / Re-reading | `06_advanced_patterns.md` | Respuestas complejas multi-hop |
| Self-Discover | `06_advanced_patterns.md` | Razonamiento sobre estructura |
| Reflexion en RAG | `06_advanced_patterns.md` | Auto-corrección de respuestas |
| Plan-and-Execute RAG | `06_advanced_patterns.md` | Queries multi-documento |
| Human-on-the-Loop | `06_advanced_patterns.md` | Validación humana en puntos críticos |

### 💾 Etapa 7 — Memoria & Persistencia
| Técnica | Archivo | Cuándo |
|---------|---------|--------|
| Short-term Memory | `07_agent_memory_persistence.md` | Contexto conversacional |
| Long-term Memory (Vector) | `07_agent_memory_persistence.md` | Recordar entre sesiones |
| Entity Memory | `07_agent_memory_persistence.md` | Personas, conceptos, entidades |
| ChromaDB Persistence | `07_agent_memory_persistence.md` | **Tu stack actual** |
| LangGraph Checkpointer | `07_agent_memory_persistence.md` | Estado de agentes persistente |

### ⚖️ Receta para Documentos Estructurados (Caso Especial)
| Problema | Solución | Archivo |
|----------|----------|---------|
| Sección partida en chunks | Hierarchical + Parent-Child | `08_legal_rag_recipe.md` |
| Sub-ítems separados del padre | Parent-Child con metadata | `08_legal_rag_recipe.md` |
| OCR + texto nativo mixto | Pipeline unificado | `08_legal_rag_recipe.md` |
| Citas de secciones perdidas | Proposition + metadata enrichment | `08_legal_rag_recipe.md` |

---

## Diagnóstico Rápido de Problemas

```
❌ "El LLM no encuentra la sección correcta"
   → Usa Hybrid Search (BM25 + Dense) + Self-Query con filtro de número de sección

❌ "El contexto de la sección llega incompleto"
   → Usa Parent-Child Chunking: recupera chunk hijo, retorna chunk padre completo

❌ "Los sub-ítems (a, b, c) pierden referencia a la sección padre"
   → Usa Hierarchical Chunking con metadata enrichment en cada sub-ítem

❌ "El LLM alucina números de sección"
   → Usa Proposition Chunking + metadata con número de sección explícito

❌ "Búsqueda no encuentra por sinónimos o terminología técnica"
   → Usa HyDE o Query Expansion con términos del dominio

❌ "Respuesta correcta pero no cita la fuente"
   → Añade source tracking en metadata + instrucción en system prompt

❌ "PDFs escaneados con mala calidad"
   → Pipeline OCR con pytesseract o AWS Textract + post-procesamiento

❌ "El RAG funciona en dev pero falla en producción"
   → Implementa RAGAS evaluation + monitoring continuo
```

---

## Stack Recomendado (Tu Setup)

```python
# requirements.txt
langchain>=0.3.0
langchain-community>=0.3.0
langchain-openai>=0.2.0
langchain-chroma>=0.1.0
chromadb>=0.5.0
rank-bm25>=0.2.2
sentence-transformers>=3.0.0
pypdf>=4.0.0
pymupdf>=1.24.0          # Mejor que pypdf para PDFs complejos
pytesseract>=0.3.10      # OCR
pillow>=10.0.0           # Procesamiento de imágenes
unstructured>=0.14.0     # Loader universal
ragas>=0.1.0             # Evaluación RAG
pydantic>=2.0.0
```
