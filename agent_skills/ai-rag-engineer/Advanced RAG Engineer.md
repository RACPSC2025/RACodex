# Advanced RAG Engineer
## 🎯 Propósito

Construir, evaluar y optimizar sistemas RAG robustos, evitando:

- Hallucinations
- Poor retrieval
- Context overflow
- Baja precisión

## 🧩 Qué hace este Skill

Este skill actúa como un arquitecto + ingeniero RAG:

1. Diseño del pipeline
   Decide:
   - tipo de chunking (semántico vs fijo)
   - estrategia de embeddings
   - tipo de búsqueda:
     - vector
     - híbrida (recomendada)
   - re-ranking

2. Construcción
   Genera código listo para producción
   Integra:
   - vector DB
   - retriever
   - LLM
   - prompt templates

3. Evaluación
   Detecta fallos típicos:
   - retrieval irrelevante
   - falta de contexto
   - respuestas inventadas
   Propone métricas:
   - recall@k
   - faithfulness
   - groundedness

4. Optimización
   Mejora:
   - chunk size dinámico
   - query rewriting
   - reranking
   - caching

## Arquitectura que genera el skill

Pipeline recomendado (Production)
User Query
   ↓
Query Rewriting
   ↓
Hybrid Retrieval (Vector + BM25)
   ↓
Reranker
   ↓
Context Selection
   ↓
LLM Generation (grounded)
   ↓
Answer + Sources

## Arquitectura de Carpetas

```
fenix-rag/
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings/
│   │   │   ├── __init__.py          # Factory: selecciona entorno según APP_ENV
│   │   │   ├── base.py              # Configuración común a todos los entornos
│   │   │   ├── development.py       # Overrides para desarrollo local
│   │   │   ├── production.py        # Overrides para producción
│   │   │   ├── staging.py           # Overrides para pre-producción
│   │   │   └── testing.py           # Overrides para tests (SQLite, Chroma test)
│   │   ├── logging.py               # Structured logging (structlog)
│   │   └── providers.py             # LLM/Embeddings factory con lazy init + pool
│   │
│   ├── ingestion/                   # Responsabilidad: documento → List[Document]
│   │   ├── __init__.py
│   │   ├── base.py                  # Protocol + ABC del loader (tipado estricto)
│   │   ├── registry.py              # LoaderRegistry — auto-discovery por MIME type
│   │   ├── loaders/
│   │   │   ├── __init__.py
│   │   │   ├── pdf_pymupdf.py       # PyMuPDF para texto nativo
│   │   │   ├── pdf_ocr.py           # OCR pipeline: fitz → Tesseract/EasyOCR
│   │   │   ├── pdf_llamaparse.py    # LlamaParse para docs complejos (tablas/imágenes)
│   │   │   ├── docling_loader.py    # IBM Docling: PDF→MD, tablas, layout
│   │   │   ├── excel_loader.py      # openpyxl + pandas → Documents estructurados
│   │   │   └── word_loader.py       # python-docx → Documents con estilos
│   │   ├── detectors/
│   │   │   ├── __init__.py
│   │   │   ├── mime_detector.py     # python-magic: detección real por bytes
│   │   │   └── quality_detector.py  # Detecta PDFs escaneados vs nativos
│   │   ├── processors/
│   │   │   ├── __init__.py
│   │   │   ├── text_cleaner.py      # Limpieza pluggable por dominio
│   │   │   ├── semantic_chunker.py  # Chunking semántico con preservación de estructura
│   │   │   ├── ocr_preprocessor.py  # Deskew, denoise, binarize antes del OCR
│   │   │   └── metadata_extractor.py # Enriquecimiento de metadatos estructurados
│   │   └── pipeline.py              # IngestionPipeline: orquesta detect→load→clean→chunk
│   │
│   ├── retrieval/                   # Responsabilidad: query → List[Document] rankeados
│   │   ├── __init__.py
│   │   ├── base.py                  # RetrieverProtocol
│   │   ├── vector_store.py          # Wrapper Chroma con gestión de colecciones
│   │   ├── bm25_retriever.py        # BM25 con tokenizador español (spaCy/NLTK)
│   │   ├── hybrid_retriever.py      # RRF fusion: vector + BM25
│   │   ├── hierarchical_retriever.py # Summary-level + chunk-level
│   │   ├── reranker.py              # FlashRank / Cohere async wrapper
│   │   └── ensemble.py              # Combina strategies con pesos configurables
│   │
│   ├── agent/                       # Responsabilidad: orquestación agéntica
│   │   ├── __init__.py
│   │   ├── graph.py                 # LangGraph StateGraph principal
│   │   ├── state.py                 # AgentState: TypedDict con todos los campos
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── document_router.py   # Clasifica documento → estrategia de ingestion
│   │   │   ├── ingestion_node.py    # Ejecuta IngestionPipeline
│   │   │   ├── retrieval_node.py    # Ejecuta ensemble retrieval
│   │   │   ├── generation_node.py   # Genera respuesta final con contexto
│   │   │   ├── reflection_node.py   # Self-reflection: evalúa calidad de respuesta
│   │   │   └── supervisor_node.py   # Supervisor pattern: subagentes especializados
│   │   ├── tools/                   # @tool functions invocables por el agente
│   │   │   ├── __init__.py
│   │   │   ├── ingest_tools.py      # ingest_pdf, ingest_excel, ingest_word, ingest_image_pdf
│   │   │   ├── search_tools.py      # semantic_search, hybrid_search, article_lookup
│   │   │   ├── analysis_tools.py    # specialized_analysis, extract_obligations
│   │   │   └── memory_tools.py      # save_context, retrieve_context
│   │   ├── skills/                  # Agent Skills: decisiones de alto nivel
│   │   │   ├── __init__.py
│   │   │   ├── document_classifier.py  # Skill: elige loader óptimo
│   │   │   ├── query_planner.py        # Skill: descompone preguntas complejas
│   │   │   └── answer_validator.py     # Skill: valida respuesta contra fuente
│   │   └── prompts/
│   │       ├── __init__.py
│   │       ├── system.py            # System prompts por rol
│   │       ├── reflection.py        # Prompts de auto-evaluación
│   │       └── domain_templates.py  # Templates específicos por dominio
│   │
│   ├── persistence/                 # Responsabilidad: estado persistente
│   │   ├── __init__.py
│   │   ├── models.py                # SQLAlchemy models: Session, Document, Message, Chunk
│   │   ├── repositories/
│   │   │   ├── __init__.py
│   │   │   ├── document_repo.py     # CRUD documentos ingestionados
│   │   │   ├── session_repo.py      # Conversaciones y contexto de sesión
│   │   │   └── chunk_repo.py        # Registro de chunks y sus metadatos
│   │   ├── migrations/              # Alembic
│   │   │   └── versions/
│   │   └── database.py              # Engine, SessionLocal, get_db()
│   │
│   ├── mcp/                         # Fase 3: MCP server wrapper
│   │   ├── __init__.py
│   │   ├── server.py                # FastMCP server
│   │   └── tools.py                 # Re-exporta agent/tools como MCP tools
│   │
│   └── api/                         # FastAPI — interfaz HTTP
│       ├── __init__.py
│       ├── main.py
│       ├── routes/
│       │   ├── chat.py
│       │   ├── documents.py
│       │   └── health.py
│       └── schemas.py               # Pydantic v2 schemas
│
├── tests/
│   ├── unit/
│   │   ├── ingestion/
│   │   ├── retrieval/
│   │   └── agent/
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_graph.py
│   └── conftest.py
│
├── storage/
│   ├── chroma/                      # Chroma persist directory
│   ├── models/                      # FlashRank / local model cache
│   └── uploads/                     # Archivos subidos (temporal)
│
├── scripts/
│   ├── ingest_batch.py              # CLI para ingestion masiva
│   └── eval_retrieval.py            # Evaluación de precisión retrieval
│
├── .env.example
├── requirements/
│   ├── base.txt                     # Dependencias comunes a todos los entornos
│   ├── development.txt              # Base + dev tools (ruff, black, mypy, ipython)
│   ├── testing.txt                  # Base + pytest, ragas, deepeval, coverage
│   └── production.txt               # Base + gunicorn, sentry, prometheus
├── pyproject.toml
├── docker-compose.yml
└── README.md

```