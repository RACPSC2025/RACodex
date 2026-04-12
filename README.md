<p align="center">
  <h1 align="center">🧠 RACodex</h1>
  <p align="center"><em>Your development assistant with personalized knowledge.</em></p>
  <p align="center">Doesn't invent APIs. Doesn't hallucinate patterns. Cites sources.</p>
</p>

---

## 🌐 Language / Idioma

| 🇺🇸 English | 🇪🇸 Español |
|:---:|:---:|
| 👇 You're reading the English version. | 👇 [Haz clic aquí para la versión en español](#-estado-del-proyecto-español) |
| Scroll down for the Spanish section. | Desplázate hacia abajo para la sección en inglés. |

---

## ⚠️ Project Status

**RACodex is under active development.** The base architecture is implemented and functional, but modules remain before a stable release.

| Metric | Value |
|--------|-------|
| **Overall Progress** | ~57% |
| **Phases Completed** | 7 of 14 |
| **Fixes Applied** | 7 of 7 |
| **Test Files** | 23+ |
| **Skill Packs** | 4 (general-dev, ai-rag-engineer, backend-python, django-ai-engineer) |
| **Estimated Time Remaining** | ~25 weeks |
| **Target Release** | Q3 2026 |

For full progress details, see [`docs/super_agent_architecture.md`](docs/super_agent_architecture.md) and [`docs/action-plan.md`](docs/action-plan.md).

---

## What is RACodex?

RACodex is an **intelligent development agent** that combines **professional RAG** with **agentic cognitive capabilities**. Unlike code assistants that rely solely on pre-trained models, RACodex learns from **YOUR** personalized knowledge: books, internal documentation, company rules, style guides.

**Core philosophy:** The best assistant isn't the one that knows the most. It's the one that knows what YOU need.

---

## Who is it for?

| User | Problem | RACodex Solution |
|------|---------|-----------------|
| 🎓 **Student** | Professor uses a book the LLM doesn't know | Upload the book → agent answers based on THAT book |
| 🚀 **Startup** | Internal conventions nobody documented | Ingest the wiki → agent codes following YOUR rules |
| 🏢 **Enterprise** | Compliance manuals, security policies | Ingest manuals → agent never suggests policy violations |
| 💼 **Freelancer** | Specific stack with favorite patterns | Ingest your docs → agent replicates your style |

---

## How it Works

```
User: "How do I implement JWT authentication?"
      + Uploads documents: [Book_X, Internal_Doc, Company_Rules]

RACodex:
  1. 🧭 Document Router → classifies files, plans ingestion
  2. 🔍 Retrieval Node → Multi-Query Fusion (3 variants) + dedup + context enrichment
  3. 📊 Grade Node (CRAG) → are docs relevant? If no, rewrite query and retry
  4. 🧠 Generation Node (Re2 Conditional) → score >0.8: direct (1 LLM call), 0.5-0.8: Re2 (2 calls), <0.5: Re2 + warning
  5. 🔄 Reflection Node → anti-hallucination: validates answer before responding

Result: "According to your FastAPI docs (ch. 8, p. 142),
         the recommended pattern is JWT with refresh tokens.
         Here's the implementation following your rules..."
```

### Full Pipeline (v2.0)

```
START → Document Router → [ingestion | retrieval]
retrieval → grade (CRAG) → [correct → generation, ambiguous/incorrect → retrieval (rewrite)]
generation (Re2 conditional) → [tools (ReAct) | reflection]
reflection → [END | retrieval (retry, max 2 iterations)]
```

---

## Techniques & Patterns

### Retrieval Pipeline

| Technique | What it does | Why it matters |
|-----------|-------------|----------------|
| **Multi-Query Fusion** | QueryTransformer generates 3 variants → retrieval per variant → deduplication by `source::chunk_index` | +20-30% recall in specialized domains |
| **Hybrid Retrieval (RRF)** | Combines vector search + BM25 with Reciprocal Rank Fusion (k=60) | Finds both semantic meaning and exact terms |
| **Parent-Child Retrieval** | Searches in small chunks, returns full parent document | LLM always sees complete context |
| **Context Enrichment Window** | Adds N neighbor chunks to each retrieved document | Solves "article cut mid-chunk" problem |
| **FlashRank Reranking** | Cross-encoder re-ranks results after retrieval | +10-20% precision in top 5 |
| **Document Augmentation** | Generates 3-5 questions per chunk during ingestion | Better query-document match |

### Cognitive Capabilities

| Pattern | What it does | Why it matters |
|---------|-------------|----------------|
| **CRAG (Corrective RAG)** | Grades retrieved docs: correct → generate, ambiguous → rewrite, incorrect → step-back + retry | Prevents answers based on irrelevant context |
| **Rethinking (Re2 Conditional)** | Two-pass generation when docs are partially relevant (grade 0.5-0.8). Direct generation when docs are clearly relevant (>0.8). | +10-15% precision on multi-hop. ~33% fewer LLM calls. |
| **Self-Reflection** | Auto-evaluates answer before delivery. If score < 0.8, reformulates and retries | Last line of defense against hallucinations |
| **Query Transformation** | Rewriting (more technical) + Step-back (more general) + Decompose (atomic sub-queries) | Bridges gap between natural language and retriever precision |
| **Semantic Routing** | Classifies queries by category using embeddings + cosine similarity (~5ms, 0 tokens) | 10x faster than LLM routing |
| **Supervisor Pattern** | Coordinates specialized sub-agents | Each sub-agent optimized for its domain |

### Ingestion

| Technique | What it does | Why it matters |
|-----------|-------------|----------------|
| **Adaptive Chunking** | Detects document type and applies type-specific chunk separators | Chunks respect the document's natural structure |
| **MIME Detection** | Detects real type by bytes, not extension | Correct handling of mislabeled files |
| **PDF Quality Detection** | Determines if PDF is native or scanned with confidence score | Automatic optimal loader selection |
| **OCR Preprocessing** | 5-step pipeline: deskew → denoise → binarize → upscale → DPI | Improves OCR accuracy for scanned documents |

### Observability (Fix #7)

Every node in the pipeline tracks metrics:

```
=== Pipeline Metrics ===
  document_router: 5ms | docs: 2 | loader_types: ['pymupdf', 'docling']
  retrieval: 250ms | docs: 15 | variants: 3, dedup_removed: 5
  generation: 1200ms | docs: 1 | mode: direct, llm_calls: 1
  reflection: 10ms | docs: 0 | score: 0.95, reason: valid_response
  TOTAL: 1465ms
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Orchestration** | LangGraph 0.2+ (ReAct + Self-Reflection + CRAG + Supervisor) |
| **LLM** | AWS Bedrock (Amazon Nova Pro / Lite) |
| **Embeddings** | Amazon Titan Embed v2 |
| **Retrieval** | Chroma + BM25 (spaCy) + RRF Fusion + FlashRank |
| **Ingestion** | PyMuPDF · EasyOCR · IBM Docling · python-docx · pandas |
| **Persistence** | PostgreSQL + SQLAlchemy 2 async + Alembic |
| **API** | FastAPI + uvicorn |
| **MCP** | FastMCP (Claude Desktop + remote agents) |

---

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/RACPSC2025/RACodex.git
cd RACodex

python -m venv .venv && source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  (Windows)

pip install -r requirements/development.txt

# Install spaCy Spanish model
python -m spacy download es_core_news_sm
```

### 2. Environment variables

```bash
cp .env.example .env
# Edit .env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, DATABASE_URL
```

### 3. Database

```bash
# With Docker (recommended for development)
docker compose up postgres -d

# Apply migrations
alembic upgrade head
```

### 4. Start services

```bash
# API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs
```

### 5. First query

```bash
# Create session
SESSION=$(curl -s -X POST http://localhost:8000/api/v1/sessions/ \
  -H "Content-Type: application/json" \
  -d '{"user_identifier": "dev@example.com"}' | jq -r .id)

# Upload document
curl -X POST "http://localhost:8000/api/v1/documents/upload/$SESSION" \
  -F "file=@/path/to/your/document.pdf"

# Query
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"query\": \"How do I implement JWT auth?\"}"
```

---

## Project Structure

```
racodex/
├── src/
│   ├── config/settings/          # Per-environment config (dev/prod/staging/test)
│   ├── agent/                    # Agentic layer (LangGraph StateGraph)
│   │   ├── graph.py              # StateGraph builder + singleton
│   │   ├── state.py              # AgentState TypedDict + pipeline_metrics
│   │   ├── metrics.py            # NodeTimer: structured per-node timing
│   │   ├── nodes/                # 6 nodes with real implementations
│   │   ├── tools/                # 12 tools (ingest, search, memory)
│   │   ├── skills/               # 8 skills + SkillRegistry
│   │   └── prompts/              # System prompts per role
│   ├── ingestion/                # Document ingestion pipeline
│   ├── retrieval/                # 6 retrieval strategies + ensemble
│   ├── persistence/              # PostgreSQL + SQLAlchemy async
│   ├── api/                      # FastAPI REST API
│   └── mcp/                      # Model Context Protocol
├── tests/                        # Unit + integration tests (23+ files)
├── agent_skills/                 # Skill Packs (modular agent knowledge)
│   ├── registry.json             # Profile registry (4 packs)
│   ├── general-dev/              # Base fallback profile
│   ├── ai-rag-engineer/          # RAG professional profile
│   ├── backend-python/           # Backend Python (in progress)
│   └── django-ai-engineer/      # Django + AI Engineer (dual mentoring)
├── docs/                         # Project documentation
│   ├── super_agent_architecture.md    # Full architecture + 14-phase plan
│   ├── action-plan.md                 # Complete action plan with priorities
│   ├── token_optimization.md          # Token optimization analysis (5 projects)
│   ├── dependencies.md                # 40+ dependencies explained
│   └── 08012026.md                    # Development diary
├── examples/                     # Reference projects analyzed
│   ├── token_optimization/            # 5 token optimization projects
│   └── ...
└── code/examples/                # LangGraph official examples (40+ notebooks)
```

---

## Documentation

| Document | Content | Language |
|----------|---------|----------|
| [`docs/super_agent_architecture.md`](docs/super_agent_architecture.md) | Full architecture, stack, 14-phase action plan, glossary | 🇪🇸 |
| [`docs/action-plan.md`](docs/action-plan.md) | Complete action plan with priorities and next steps | 🇪🇸 |
| [`docs/token_optimization.md`](docs/token_optimization.md) | Token optimization analysis (5 projects, 8 techniques) | 🇪🇸 |
| [`docs/dependencies.md`](docs/dependencies.md) | 40+ dependencies explained individually | 🇪🇸 |
| [`docs/08012026.md`](docs/08012026.md) | Development diary — changes, decisions, findings | 🇪🇸 |
| [`agent_skills/`](agent_skills/) | 📚 Reference technical docs — agent skills (RAG Mastery, Agentic Patterns, Engineering Fundamentals) | 🇪🇸 |
| [`agent.md`](agent.md) | 🧠 Operating instructions for the agent assistant | 🇪🇸 |

> **Note:** Most documentation is currently in Spanish. English translations are planned. For technical questions, feel free to open an issue in English.

---

## License

MIT — Free to use, modify, and distribute. The community is welcome.

---

> *"The best code assistant isn't the one that knows the most. It's the one that knows what YOU need."*
>
> — **RAC**, creator of RACodex

---
---

<br>

<div id="estado-del-proyecto-español"></div>

## 🌐 Idioma / Language

| 🇪🇸 Español | 🇺🇸 English |
|:---:|:---:|
| 👇 Estás leyendo la versión en español. | 👇 [Click here for the English version](#-racodex) |

---

## ⚠️ Estado del Proyecto

**RACodex está en desarrollo activo.** La arquitectura base está implementada y funcional, pero quedan módulos por completar antes de un release estable.

| Métrica | Valor |
|---------|-------|
| **Progreso general** | ~57% |
| **Fases completadas** | 7 de 14 |
| **Fixes aplicados** | 7 de 7 |
| **Archivos de tests** | 23+ |
| **Skill Packs** | 4 (general-dev, ai-rag-engineer, backend-python, django-ai-engineer) |
| **Tiempo estimado restante** | ~25 semanas |
| **Release objetivo** | Q3 2026 |

Para detalles completos del progreso, consulta [`docs/super_agent_architecture.md`](docs/super_agent_architecture.md) y [`docs/action-plan.md`](docs/action-plan.md).

---

## ¿Qué es RACodex?

RACodex es un **agente de desarrollo inteligente** que combina **RAG profesional** con **capacidades cognitivas agenticas**. A diferencia de los asistentes de código que dependen exclusivamente del modelo preentrenado, RACodex se alimenta de **tu** conocimiento personalizado: libros, documentación interna, reglas de empresa, guías de estilo.

**Filosofía central:** El mejor asistente no es el que más sabe. Es el que sabe lo que TÚ necesitas.

---

## ¿Para quién?

| Usuario | Problema | Solución RACodex |
|---------|----------|-----------------|
| 🎓 **Estudiante** | El profesor usa un libro que el LLM no conoce | Carga el libro → el agente responde basado en ESE libro |
| 🚀 **Startup** | Convenciones internas que nadie documenta | Ingesta la wiki → el agente codea siguiendo TUS reglas |
| 🏢 **Empresa** | Manuales de compliance, políticas de seguridad | Ingesta los manuales → el agente nunca sugiere algo que viole políticas |
| 💼 **Freelancer** | Stack específico con patrones favoritos | Ingesta tu documentación → el agente replica tu estilo |

---

## Cómo funciona

```
Usuario: "¿Cómo implemento autenticación JWT?"
         + Carga documentos: [Libro_X, Doc_Interna, Reglas_Empresa]

RACodex:
  1. 🧭 Document Router → clasifica archivos, planifica ingestión
  2. 🔍 Retrieval Node → Multi-Query Fusion (3 variantes) + deduplicación + context enrichment
  3. 📊 Grade Node (CRAG) → ¿los docs son relevantes? Si no, reformula y reintenta
  4. 🧠 Generation Node (Re2 Condicional) → score >0.8: directo (1 llamada LLM), 0.5-0.8: Re2 (2 llamadas), <0.5: Re2 + advertencia
  5. 🔄 Reflection Node → anti-alucinación: valida la respuesta antes de responder

Resultado: "Según tu documentación de FastAPI (cap. 8, p. 142),
            el patrón recomendado es JWT con refresh tokens.
            Aquí está la implementación siguiendo tus reglas..."
```

### Pipeline Completo (v2.0)

```
START → Document Router → [ingestion | retrieval]
retrieval → grade (CRAG) → [correct → generation, ambiguous/incorrect → retrieval (rewrite)]
generation (Re2 condicional) → [tools (ReAct) | reflection]
reflection → [END | retrieval (retry, máx 2 iteraciones)]
```

---

## Técnicas y Patrones Implementados

### Pipeline de Retrieval

| Técnica | Qué hace | Por qué importa |
|---------|----------|-----------------|
| **Multi-Query Fusion** | QueryTransformer genera 3 variantes → retrieval por variante → deduplicación por `source::chunk_index` | Recall +20-30% en dominios especializados |
| **Hybrid Retrieval (RRF)** | Combina búsqueda vectorial + BM25 con Reciprocal Rank Fusion (k=60) | Encuentra tanto por significado semántico como por términos exactos |
| **Parent-Child Retrieval** | Busca en chunks pequeños, retorna el documento padre completo | El LLM siempre ve el contexto completo, no fragmentos sueltos |
| **Context Enrichment Window** | Agrega N chunks vecinos a cada documento recuperado | Resuelve el problema de "artículo cortado a mitad del chunk" |
| **FlashRank Reranking** | Cross-encoder reordena los resultados después del retrieval | +10-20% precisión en top 5 |
| **Document Augmentation** | Genera 3-5 preguntas por chunk durante la ingestión y las indexa | Mejor match query-documento |

### Capacidades Cognitivas

| Patrón | Qué hace | Por qué importa |
|--------|----------|-----------------|
| **CRAG (Corrective RAG)** | Evalúa la calidad de los docs recuperados: correct → genera, ambiguous → reformula, incorrect → step-back + reintenta | Evita generar respuestas basadas en contexto irrelevante |
| **Rethinking (Re2 Condicional)** | Two-pass generation cuando docs son parcialmente relevantes (grade 0.5-0.8). Generación directa cuando docs son claramente relevantes (>0.8). | +10-15% precisión en multi-hop. ~33% menos llamadas LLM. |
| **Self-Reflection** | Auto-evaluación de la respuesta antes de entregarla. Si score < 0.8, reintenta | Última línea de defensa contra alucinaciones |
| **Query Transformation** | Rewriting (más técnica) + Step-back (más general) + Decompose (sub-queries atómicas) | Cierra la brecha entre lenguaje natural del usuario y precisión del retriever |
| **Semantic Routing** | Clasifica queries por categoría usando embeddings + cosine similarity (~5ms, 0 tokens) | 10x más rápido que LLM routing |
| **Supervisor Pattern** | Coordina subagentes especializados | Cada subagente se optimiza para su dominio |

### Ingestión

| Técnica | Qué hace | Por qué importa |
|---------|----------|-----------------|
| **Adaptive Chunking** | Detecta el tipo de documento y aplica separadores de chunking específicos | Chunks que respetan la estructura natural del documento |
| **MIME Detection** | Detecta el tipo real por bytes, no por extensión | Manejo correcto de archivos sin extensión o con extensión incorrecta |
| **PDF Quality Detection** | Determina si un PDF es nativo o escaneado con confidence score | Selección automática del loader óptimo |
| **OCR Preprocessing** | Pipeline de 5 pasos: deskew → denoise → binarize → upscale → DPI | Mejora la precisión del OCR para documentos escaneados |

### Observabilidad (Fix #7)

Cada nodo del pipeline registra métricas:

```
=== Pipeline Metrics ===
  document_router: 5ms | docs: 2 | loader_types: ['pymupdf', 'docling']
  retrieval: 250ms | docs: 15 | variants: 3, dedup_removed: 5
  generation: 1200ms | docs: 1 | mode: direct, llm_calls: 1
  reflection: 10ms | docs: 0 | score: 0.95, reason: valid_response
  TOTAL: 1465ms
```

---

## Stack Técnico

| Capa | Tecnología |
|------|-----------|
| **Orquestación** | LangGraph 0.2+ (ReAct + Self-Reflection + CRAG + Supervisor) |
| **LLM** | AWS Bedrock (Amazon Nova Pro / Lite) |
| **Embeddings** | Amazon Titan Embed v2 |
| **Retrieval** | Chroma + BM25 (spaCy) + RRF Fusion + FlashRank |
| **Ingestion** | PyMuPDF · EasyOCR · IBM Docling · python-docx · pandas |
| **Persistencia** | PostgreSQL + SQLAlchemy 2 async + Alembic |
| **API** | FastAPI + uvicorn |
| **MCP** | FastMCP (Claude Desktop + agentes remotos) |

---

## Inicio Rápido

### 1. Clonar y configurar entorno

```bash
git clone https://github.com/RACPSC2025/RACodex.git
cd RACodex

python -m venv .venv && source .venv/bin/activate  # Linux/macOS
# o: .venv\Scripts\activate  (Windows)

pip install -r requirements/development.txt

# Instalar modelo spaCy español
python -m spacy download es_core_news_sm
```

### 2. Variables de entorno

```bash
cp .env.example .env
# Editar .env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, DATABASE_URL
```

### 3. Base de datos

```bash
# Con Docker (recomendado para desarrollo)
docker compose up postgres -d

# Aplicar migraciones
alembic upgrade head
```

### 4. Iniciar servicios

```bash
# API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs
```

### 5. Primera consulta

```bash
# Crear sesión
SESSION=$(curl -s -X POST http://localhost:8000/api/v1/sessions/ \
  -H "Content-Type: application/json" \
  -d '{"user_identifier": "dev@ejemplo.com"}' | jq -r .id)

# Subir documento
curl -X POST "http://localhost:8000/api/v1/documents/upload/$SESSION" \
  -F "file=@/ruta/a/tu/documento.pdf"

# Consultar
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"query\": \"¿Cómo implemento autenticación JWT?\"}"
```

---

## Estructura del Proyecto

```
racodex/
├── src/
│   ├── config/settings/          # Configuración por entorno (dev/prod/staging/test)
│   ├── agent/                    # Capa agéntica (LangGraph StateGraph)
│   │   ├── graph.py              # StateGraph builder + singleton
│   │   ├── state.py              # AgentState TypedDict + pipeline_metrics
│   │   ├── metrics.py            # NodeTimer: timing estructurado por nodo
│   │   ├── nodes/                # 6 nodos con implementación real
│   │   ├── tools/                # 12 tools (ingest, search, memory)
│   │   ├── skills/               # 8 skills + SkillRegistry
│   │   └── prompts/              # System prompts por rol
│   ├── ingestion/                # Pipeline de ingestión (document_type propagation, batching LLM)
│   ├── retrieval/                # 6 estrategias de retrieval + ensemble
│   ├── persistence/              # PostgreSQL + SQLAlchemy async
│   ├── api/                      # FastAPI REST API
│   └── mcp/                      # Model Context Protocol
├── tests/                        # Tests unitarios e integración (23+ archivos)
├── agent_skills/                 # Skill Packs (conocimiento modular del agente)
│   ├── registry.json             # Registro de perfiles (4 packs)
│   ├── general-dev/              # Perfil base fallback
│   ├── ai-rag-engineer/          # Perfil RAG profesional
│   ├── backend-python/           # Backend Python (en progreso)
│   └── django-ai-engineer/      # Django + AI Engineer (mentoría dual)
├── docs/                         # Documentación del proyecto
│   ├── super_agent_architecture.md    # Arquitectura completa + plan de 14 fases
│   ├── action-plan.md                 # Plan de acción completo con prioridades
│   ├── token_optimization.md          # Análisis de optimización de tokens (5 proyectos)
│   ├── dependencies.md                # 40+ dependencias explicadas
│   └── 08012026.md                    # Diario de desarrollo
├── examples/                     # Proyectos de referencia analizados
│   ├── token_optimization/            # 5 proyectos de optimización de tokens
│   └── ...
└── code/examples/                # Ejemplos oficiales de LangGraph (40+ notebooks)
```

---

## Documentación

| Documento | Contenido | Idioma |
|-----------|-----------|--------|
| [`docs/super_agent_architecture.md`](docs/super_agent_architecture.md) | Arquitectura completa, stack, plan de acción de 14 fases, glosario | 🇪🇸 |
| [`docs/action-plan.md`](docs/action-plan.md) | Plan de acción completo con prioridades y próximos pasos | 🇪🇸 |
| [`docs/token_optimization.md`](docs/token_optimization.md) | Análisis de optimización de tokens (5 proyectos, 8 técnicas) | 🇪🇸 |
| [`docs/dependencies.md`](docs/dependencies.md) | 40+ dependencias explicadas individualmente | 🇪🇸 |
| [`docs/08012026.md`](docs/08012026.md) | Diario de desarrollo — cambios, decisiones, hallazgos | 🇪🇸 |
| [`agent_skills/`](agent_skills/) | 📚 Documentación técnica de referencia — skills del agente | 🇪🇸 |
| [`agent.md`](agent.md) | 🧠 Instrucciones operativas del agente asistente | 🇪🇸 |

> **Nota:** La mayor parte de la documentación está actualmente en español. Se planean traducciones al inglés. Para preguntas técnicas, no dudes en abrir un issue en inglés.

---

## Licencia

MIT — Libre para usar, modificar y distribuir. La comunidad es bienvenida.

---

> *"El mejor asistente de código no es el que más sabe. Es el que sabe lo que TÚ necesitas."*
>
> — **RAC**, creador de RACodex
