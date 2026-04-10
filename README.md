# 🧠 RACodex

> **Tu asistente de desarrollo con conocimiento personalizado.**
> No inventa APIs. No alucina patrones. Cita fuentes.

---

## ⚠️ Estado del Proyecto

**RACodex está en desarrollo activo.** La arquitectura base está implementada y funcional, pero quedan módulos por completar antes de un release estable.

| Progreso actual | ~65% |
|----------------|------|
| **Fases completadas** | 1-8 (de 13) |
| **Tiempo estimado restante** | ~17 semanas |
| **Release objetivo** | Q3 2026 |

Para ver el detalle completo del progreso, consulta [`docs/super_agent_architecture.md`](docs/super_agent_architecture.md).

---

## ¿Qué es?

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
  1. 🧭 Semantic Router → clasifica la query (5ms, 0 tokens)
  2. ✍️ Query Transformer → [original, rewritten, step-back]
  3. 🔍 Multi-Query Retrieval → busca en TUS documentos con 6 estrategias
  4. 🪟 Context Enrichment → agrega contexto de chunks vecinos
  5. 📊 CRAG Grade → ¿los docs son relevantes? Si no, reformula y reintenta
  6. 🧠 Rethinking Generation → 2 pasadas de lectura para precisión
  7. 🔄 Reflection → anti-alucinación: valida la respuesta antes de responder

Resultado: "Según tu documentación de FastAPI (cap. 8, p. 142),
            el patrón recomendado es JWT con refresh tokens.
            Aquí está la implementación siguiendo tus reglas..."
```

---

## Técnicas y Patrones Implementados

### Pipeline de Retrieval

| Técnica | Qué hace | Por qué importa |
|---------|----------|-----------------|
| **Hybrid Retrieval (RRF)** | Combina búsqueda vectorial + BM25 con Reciprocal Rank Fusion (k=60) | Encuentra tanto por significado semántico como por términos exactos (números de sección, siglas) |
| **Parent-Child Retrieval** | Busca en chunks pequeños, retorna el documento padre completo | El LLM siempre ve el contexto completo, no fragmentos sueltos |
| **Context Enrichment Window** | Agrega N chunks vecinos a cada documento recuperado | Resuelve el problema de "artículo cortado a mitad del chunk" |
| **Multi-Query Retrieval** | Ejecuta retrieval con múltiples variantes de la query (rewritten, step-back) y combina resultados | Recall +20-30% en dominios especializados |
| **FlashRank Reranking** | Cross-encoder reordena los resultados después del retrieval | +10-20% precisión en top 5 |
| **Document Augmentation** | Genera 3-5 preguntas por chunk durante la ingestión y las indexa | Mejor match query-documento porque la query del usuario naturalmente se parece a una pregunta |

### Capacidades Cognitivas

| Patrón | Qué hace | Por qué importa |
|--------|----------|-----------------|
| **ReAct** | El agente intercala pensamiento (Reasoning) con acción (Acting) — piensa → usa tool → observa → piensa de nuevo | Decisiones dinámicas, no flujo predefinido |
| **CRAG (Corrective RAG)** | Evalúa la calidad de los docs recuperados: correct → genera, ambiguous → reformula, incorrect → step-back + reintenta | Evita generar respuestas basadas en contexto irrelevante |
| **Rethinking (Re2)** | Two-pass generation: 1ra pasada identifica pasajes clave, 2da genera la respuesta citando fuentes | +10-15% precisión en respuestas multi-hop |
| **Self-Reflection** | Auto-evaluación de la respuesta antes de entregarla. Si score < 0.8, reintenta | Última línea de defensa contra alucinaciones |
| **Semantic Routing** | Clasifica queries por categoría usando embeddings + cosine similarity (~5ms, 0 tokens) | 10x más rápido que LLM routing |
| **Query Transformation** | Rewriting (más técnica) + Step-back (más general) + Decompose (sub-queries atómicas) | Cierra la brecha entre lenguaje natural del usuario y precisión del retriever |
| **Supervisor Pattern** | Coordina subagentes especializados | Cada subagente se optimiza para su dominio |

### Ingestión

| Técnica | Qué hace | Por qué importa |
|---------|----------|-----------------|
| **Adaptive Chunking** | Detecta el tipo de documento y aplica separadores de chunking específicos | Chunks que respetan la estructura natural del documento |
| **Hierarchical Chunking** | Preserva la jerarquía del documento (documento → sección → sub-sección) | Metadata breadcrumb en cada chunk |
| **MIME Detection** | Detecta el tipo real por bytes, no por extensión | Manejo correcto de archivos sin extensión o con extensión incorrecta |
| **PDF Quality Detection** | Determina si un PDF es nativo o escaneado con confidence score | Selección automática del loader óptimo |

---

## Stack Técnico

| Capa | Tecnología |
|------|-----------|
| **Orquestación** | LangGraph 0.2+ (ReAct + Self-Reflection + CRAG + Supervisor) |
| **LLM** | AWS Bedrock (Amazon Nova Pro / Lite) |
| **Embeddings** | Amazon Titan Embed v2 |
| **Retrieval** | Chroma + BM25 (spaCy español) + RRF Fusion + FlashRank |
| **Ingestion** | PyMuPDF · EasyOCR · IBM Docling · python-docx · pandas |
| **Persistencia** | PostgreSQL + SQLAlchemy 2 async + Alembic |
| **API** | FastAPI + uvicorn |
| **MCP** | FastMCP (Claude Desktop + agentes remotos) |

---

## Inicio Rápido

### 1. Clonar y configurar entorno

```bash
git clone https://github.com/RACPSC2025/RACodex.git
cd racodex

python -m venv .venv && source .venv/bin/activate
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
│   │   ├── state.py              # AgentState TypedDict
│   │   ├── nodes/                # 6 nodos del grafo
│   │   ├── tools/                # 8 tools LangChain
│   │   ├── skills/               # 7 skills de decisión
│   │   └── prompts/              # System prompts
│   ├── ingestion/                # Pipeline de ingestión
│   ├── retrieval/                # 6 estrategias de retrieval
│   ├── persistence/              # PostgreSQL + SQLAlchemy async
│   ├── api/                      # FastAPI REST API
│   └── mcp/                      # Model Context Protocol
├── tests/                        # Tests unitarios e integración
├── agent_skills/                 # Documentación de skills del agente
├── docs/                         # Documentación del proyecto
│   ├── super_agent_architecture.md    # Arquitectura completa + plan de acción
│   └── 08012026.md                    # Diario de desarrollo
└── storage/                      # ChromaDB, uploads, models
```

---

## Documentación

| Documento | Contenido |
|-----------|-----------|
| [`docs/super_agent_architecture.md`](docs/super_agent_architecture.md) | Arquitectura completa, stack, plan de acción por fases (13 fases), glosario de conceptos |
| [`docs/08012026.md`](docs/08012026.md) | Diario de desarrollo — registro de cambios, decisiones, hallazgos |
| [`agent_skills/`](agent_skills/) | 📚 **Documentación técnica de referencia** — skills del agente (RAG Mastery, Agentic Patterns, Engineering Fundamentals). Úsalas como guía para tu propio RAG. |
| [`agent.md`](agent.md) | 🧠 Instrucciones operativas del agente asistente — cómo piensa, cómo trabaja, cómo documenta |

---

## Licencia

MIT — Libre para usar, modificar y distribuir. La comunidad es bienvenida.

---

> *"El mejor asistente de código no es el que más sabe. Es el que sabe lo que TÚ necesitas."*
>
> — **RAC**, creador de RACodex
