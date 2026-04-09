# Skill: Agentic Patterns & Multi-Agent Design
**Versión:** 1.0.0 | **Stack principal:** LangGraph · Python 3.12+ | **LLMs:** OpenAI / Anthropic / Google

---

## Propósito de esta Skill

Esta skill es la referencia técnica completa del Agente Fenix Tech Líder para el diseño e implementación de sistemas agenticos. Contiene patrones probados en producción con ejemplos reales y ejecutables.

**Antes de diseñar cualquier agente, consulta esta skill.**

---

## Índice de Patrones

### 🧠 Patrones Cognitivos Fundamentales
| Patrón | Archivo | Descripción |
|--------|---------|-------------|
| ReAct | `01_react_pattern.md` | Razonamiento + Acción intercalados con observaciones |
| Reflection | `02_reflection_pattern.md` | Auto-evaluación y corrección iterativa |
| Planning | `03_planning_pattern.md` | Planificación separada de ejecución |
| Tool Use | `04_tool_use_pattern.md` | Uso avanzado de herramientas y function calling |
| Self-RAG | `05_self_rag_pattern.md` | El agente decide cuándo recuperar contexto |

### 🏗️ Patrones de Diseño Multi-Agente
| Patrón | Archivo | Descripción |
|--------|---------|-------------|
| Parallel Agents | `06_parallel_agents.md` | Múltiples agentes ejecutando en paralelo |
| Sequential Agents | `07_sequential_agents.md` | Cadena de agentes en pipeline lineal |
| Loop Agents | `08_loop_agents.md` | Ciclo iterativo hasta condición de parada |
| Router Agent | `09_router_agent.md` | Agente que enruta a especialistas |
| Aggregator Agent | `10_aggregator_agent.md` | Consolida outputs de múltiples agentes |
| Network Agent | `11_network_agent.md` | Malla de agentes con comunicación libre |
| Hierarchical Agents | `12_hierarchical_agents.md` | Supervisor + Workers especializados |
| Subagents | `13_subagents.md` | Agentes anidados como herramientas |
| Handoffs | `14_handoffs_pattern.md` | Transferencia de control entre agentes |

---

## Setup Base Común

Todos los ejemplos asumen este setup de dependencias:

```bash
pip install langgraph langchain langchain-openai langchain-anthropic \
            langchain-google-genai pydantic python-dotenv
```

```python
# config/llm_factory.py
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


def get_llm(
    provider: LLMProvider = LLMProvider.OPENAI,
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs,
) -> BaseChatModel:
    """Factory para obtener LLM de cualquier proveedor."""
    match provider:
        case LLMProvider.OPENAI:
            return ChatOpenAI(
                model=model or "gpt-4o",
                temperature=temperature,
                **kwargs,
            )
        case LLMProvider.ANTHROPIC:
            return ChatAnthropic(
                model=model or "claude-3-5-sonnet-20241022",
                temperature=temperature,
                **kwargs,
            )
        case LLMProvider.GOOGLE:
            return ChatGoogleGenerativeAI(
                model=model or "gemini-1.5-pro",
                temperature=temperature,
                **kwargs,
            )
        case _:
            raise ValueError(f"Provider no soportado: {provider}")
```

---

## Guía de Selección de Patrón

```
¿Tarea simple y lineal?
  → Sequential Agents o Planning Pattern

¿Necesita razonamiento + acción en ciclo?
  → ReAct Pattern

¿Necesita auto-corrección?
  → Reflection Pattern o Loop Agents

¿Múltiples subtareas independientes?
  → Parallel Agents

¿Múltiples tipos de consultas distintas?
  → Router Agent

¿Necesita combinar resultados de varios agentes?
  → Aggregator Agent

¿Sistema grande con especialistas?
  → Hierarchical Agents

¿Agente que delega a otro agente?
  → Subagents o Handoffs

¿Agentes que necesitan comunicarse entre sí?
  → Network Agent

¿El agente decide si buscar información?
  → Self-RAG Pattern
```

---

## Convenciones de esta Skill

- **State**: Siempre `TypedDict` con type hints completos.
- **Nodes**: Funciones puras que reciben `State` y retornan `dict`.
- **Edges**: Condicionales usan `Literal` para type safety.
- **LLM**: Agnóstico — usa `get_llm()` del factory.
- **Structured Output**: `pydantic.BaseModel` + `.with_structured_output()`.
- **Errores**: Siempre tipados y manejados en el grafo, nunca silenciados.
