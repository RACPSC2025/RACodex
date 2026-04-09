# 🧠 AI & RAG Engineer — Skill Pack

> Arquitectura RAG profesional, LangGraph, patrones agentic, retrieval avanzado, evaluación RAGAS.
> Para construir sistemas de IA con conocimiento personalizado que no alucinan.

---

## ¿Cuándo se activa este pack?

Este skill se activa cuando el usuario:
- Necesita construir o mejorar un sistema RAG
- Trabaja con LangGraph, agentes, o orquestación de LLMs
- Busca patrones de retrieval avanzado (hybrid, CRAG, parent-child, multi-query)
- Requiere evaluación de calidad RAG (RAGAS, faithfulness, answer relevancy)
- Implementa ingestion pipelines con chunking adaptativo
- Diseña agentes con ReAct, Reflection, Planning, o Supervisor patterns

---

## Contenido del Pack

### 📐 Advanced RAG Engineer
| Archivo | Descripción |
|---------|-------------|
| `Advanced RAG Engineer.md` | Pipeline RAG de producción: diseño de arquitectura, selección de chunking, retrieval, reranking, y generación |

### 🤖 Agentic Patterns Multi-Agent Design
| Archivo | Descripción |
|---------|-------------|
| `INDEX.md` | Índice maestro de patrones agentic |
| `01_react_pattern.md` | ReAct: Reasoning + Acting intercalados |
| `02_reflection_pattern.md` | Auto-evaluación y corrección iterativa |
| `03_planning_pattern.md` | Planificación separada de ejecución |
| `04_05_tool_use_self_rag.md` | Tool Use avanzado + Self-RAG |
| `10_14_aggregator_network_hierarchical_handoffs.md` | Aggregator, Network, Hierarchical, Handoffs |

### 📚 Rag Examples
| Archivo | Descripción |
|---------|-------------|
| `01.fundamentals-langgraph.md` | LangGraph fundamental: StateGraph, nodos, edges condicionales |
| `02.langchain_rag.md` | RAG con LangChain: loaders, chunking, retrieval, anti-hallucination |
| `03.Query_Techniques.md` | Query Rewriting, Step-Back, Sub-query Decomposition |
| `04.Human_in_the_Loop.md` | HITL: interrupts, aprobación, time travel, cascading approval |

### 🏆 Rag Mastery
| Archivo | Descripción |
|---------|-------------|
| `INDEX.md` | Mapa maestro de técnicas RAG por etapa del pipeline |
| `01_document_loading.md` | PDF nativo, OCR, multimodal, SmartPDFLoader |
| `02_chunking_strategies.md` | Hierarchical, Parent-Child, Semantic, Proposition |
| `03_indexing_retrieval.md` | Dense, Sparse (BM25), Hybrid RRF, Hierarchical |
| `04_query_techniques.md` | HyDE, Self-Query, Transformations, Fusion, Adaptive |
| `05_reranking_filtering.md` | Cross-Encoder, Cohere, MMR, Contextual Compression |
| `06_advanced_patterns.md` | CRAG, Self-Retrieval, Rethinking, Reflexion, Human-on-the-Loop |
| `07_agent_memory_persistence.md` | Short-term, Long-term, Entity Memory, ChromaDB persistence |
| `08_legal_rag_recipe.md` | Pipeline completo de producción (adaptable a cualquier dominio) |

---

## Reglas de Comportamiento

1. **Siempre** prioriza retrieval sobre generación — el 70% de los errores de RAG vienen de recuperar documentos incorrectos
2. **Nunca** generes respuestas sin citar fuentes — si la información no está en el contexto, dilo explícitamente
3. **Siempre** aplica CRAG grading antes de generar — evalúa la calidad de los docs recuperados
4. **Prioriza** Parent-Child Retrieval para documentos con estructura jerárquica
5. **Usa** Context Enrichment Window cuando los chunks puedan estar cortados a mitad de contexto
6. **Aplica** Multi-Query Retrieval para queries vagas o ambiguas
7. **Cita** el documento específico de este pack cuando apliques un patrón técnico

---

## Ejemplos de Uso

### Ejemplo 1: Construir un RAG desde cero
**Usuario:** "¿Cómo estructuro un sistema RAG para mi documentación técnica?"  
**Agente:** (Consulta `Advanced RAG Engineer.md` para el pipeline, luego `02_chunking_strategies.md` para chunking, y `03_indexing_retrieval.md` para retrieval)  
**Resultado:** Pipeline completo: SmartPDFLoader → HierarchicalChunker → HybridRetriever → CrossEncoder Rerank → Rethinking Generate.

### Ejemplo 2: Mejorar recall del retrieval
**Usuario:** "Mi retriever no encuentra los documentos correctos cuando la query es vaga"  
**Agente:** (Consulta `04_query_techniques.md` para Query Rewriting + Fusion Retrieval)  
**Resultado:** Implementa QueryTransformer que genera 3 variantes de la query y combina resultados con RRF.

### Ejemplo 3: Prevenir alucinaciones
**Usuario:** "El LLM inventa información que no está en mis documentos"  
**Agente:** (Consulta `06_advanced_patterns.md` para CRAG + Reflection, y `05_reranking_filtering.md` para LLM Relevance Filter)  
**Resultado:** Implementa grading de documentos + reflection node que valida la respuesta antes de entregarla.

---

## Dependencias

- Requiere: `general-dev/` (fundamentos de ingeniería compartidos)
- Opcional: `backend-python/` (si el proyecto usa Python para el backend)

---

## Changelog

| Versión | Fecha | Cambio |
|---------|-------|--------|
| 1.0.0 | 2026-04-08 | Versión inicial — pack completo de AI & RAG Engineer |

---

## Licencia

MIT — Libre para usar, modificar y distribuir.
