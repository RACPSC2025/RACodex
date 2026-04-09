# Patrón 02: Reflection (Auto-Evaluación y Corrección)

## Concepto

El agente genera una respuesta inicial y luego la **evalúa críticamente** con un segundo paso de reflexión. Puede iterar hasta que la calidad sea suficiente o se alcance el límite de intentos.

```
Generate → Reflect → [OK → End] | [Needs improvement → Revise → Reflect → ...]
```

## Cuándo Usarlo

- Generación de código, textos, análisis que requieren alta calidad.
- Tareas donde "bueno a la primera" no es suficiente.
- Cuando puedes definir criterios de calidad evaluables.
- Reducir alucinaciones con auto-corrección.

---

## Implementación con LangGraph

```python
# patterns/reflection_agent.py
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# ─── Estado ───────────────────────────────────────────────────────────────────

class ReflectionState(TypedDict):
    task: str
    draft: str
    feedback: str
    final_output: str
    iteration: int
    max_iterations: int
    approved: bool


# ─── Schemas de Structured Output ─────────────────────────────────────────────

class DraftOutput(BaseModel):
    content: str = Field(description="El contenido generado.")
    reasoning: str = Field(description="Razonamiento breve sobre las decisiones tomadas.")


class ReflectionOutput(BaseModel):
    approved: bool = Field(description="True si el draft es suficientemente bueno.")
    score: int = Field(description="Puntuación de calidad del 1 al 10.", ge=1, le=10)
    feedback: str = Field(description="Feedback específico y accionable para mejorar.")
    strengths: list[str] = Field(description="Aspectos positivos del draft.")
    weaknesses: list[str] = Field(description="Aspectos a mejorar del draft.")


# ─── LLM Factory ──────────────────────────────────────────────────────────────

def get_llm(provider: str = "openai"):
    match provider:
        case "openai":    return ChatOpenAI(model="gpt-4o", temperature=0.3)
        case "anthropic": return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
        case "google":    return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
        case _: raise ValueError(f"Provider no soportado: {provider}")


# ─── Nodos ────────────────────────────────────────────────────────────────────

GENERATOR_SYSTEM = """Eres un experto generador de contenido de alta calidad.
Genera el mejor output posible para la tarea dada.
Si hay feedback previo, incorpóralo completamente en esta versión mejorada."""

REFLECTOR_SYSTEM = """Eres un crítico experto y riguroso.
Evalúa el draft con criterios profesionales:
- Corrección técnica y factual
- Claridad y estructura
- Completitud
- Calidad general
Sé específico en el feedback. Aprueba (score >= 8) solo si está verdaderamente listo para producción."""


def generate_node(state: ReflectionState) -> dict:
    """Genera o mejora el draft basándose en el feedback."""
    llm = get_llm().with_structured_output(DraftOutput)

    messages = [SystemMessage(content=GENERATOR_SYSTEM)]

    if state["feedback"]:
        user_content = (
            f"Tarea: {state['task']}\n\n"
            f"Draft anterior:\n{state['draft']}\n\n"
            f"Feedback del crítico:\n{state['feedback']}\n\n"
            "Por favor, mejora el draft incorporando todo el feedback."
        )
    else:
        user_content = f"Tarea: {state['task']}"

    messages.append(HumanMessage(content=user_content))

    result: DraftOutput = llm.invoke(messages)
    print(f"\n✏️  [Iteración {state['iteration'] + 1}] Draft generado (score previo implícito)")

    return {
        "draft": result.content,
        "iteration": state["iteration"] + 1,
    }


def reflect_node(state: ReflectionState) -> dict:
    """Evalúa el draft y genera feedback estructurado."""
    llm = get_llm().with_structured_output(ReflectionOutput)

    messages = [
        SystemMessage(content=REFLECTOR_SYSTEM),
        HumanMessage(content=(
            f"Tarea original: {state['task']}\n\n"
            f"Draft a evaluar (iteración {state['iteration']}):\n\n"
            f"{state['draft']}"
        )),
    ]

    result: ReflectionOutput = llm.invoke(messages)

    print(f"   🔍 Reflection: score={result.score}/10, approved={result.approved}")
    if result.weaknesses:
        print(f"   ⚠️  Debilidades: {', '.join(result.weaknesses[:2])}")

    return {
        "feedback": result.feedback,
        "approved": result.approved or state["iteration"] >= state["max_iterations"],
        "final_output": state["draft"] if result.approved else "",
    }


def finalize_node(state: ReflectionState) -> dict:
    """Consolida el output final."""
    print(f"\n✅ Output aprobado tras {state['iteration']} iteración(es).")
    return {"final_output": state["draft"]}


# ─── Enrutamiento ─────────────────────────────────────────────────────────────

def should_continue(state: ReflectionState) -> Literal["generate", "finalize"]:
    """Continúa iterando o finaliza según aprobación."""
    if state["approved"]:
        return "finalize"
    return "generate"


# ─── Construcción del Grafo ────────────────────────────────────────────────────

def build_reflection_graph():
    graph = StateGraph(ReflectionState)

    graph.add_node("generate", generate_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "reflect")
    graph.add_conditional_edges(
        "reflect",
        should_continue,
        {"generate": "generate", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


# ─── Uso ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_reflection_graph()

    result = app.invoke({
        "task": (
            "Escribe una función Python que implemente búsqueda binaria. "
            "Debe incluir: type hints, docstring, manejo de edge cases y tests."
        ),
        "draft": "",
        "feedback": "",
        "final_output": "",
        "iteration": 0,
        "max_iterations": 3,
        "approved": False,
    })

    print("\n" + "="*60)
    print("OUTPUT FINAL:")
    print("="*60)
    print(result["final_output"])
```

---

## Variante: Reflection con Múltiples Críticos

Para tareas críticas, usa múltiples perspectivas de evaluación:

```python
class MultiReflectorState(TypedDict):
    task: str
    draft: str
    technical_feedback: str   # Crítico técnico
    style_feedback: str       # Crítico de estilo
    security_feedback: str    # Crítico de seguridad
    combined_feedback: str
    iteration: int
    approved: bool

# Cada critic_node evalúa desde su perspectiva especializada
# Un aggregator_node combina todos los feedbacks
```

---

## Anti-Patterns

| Anti-pattern | Problema | Solución |
|---|---|---|
| Sin `max_iterations` | Loop infinito | Siempre define límite y lo respeta |
| Feedback vago ("mejóralo") | El generador no sabe qué cambiar | Structured output con campos específicos |
| Mismo LLM genera y refleja | Confirma sus propios sesgos | Usa modelos distintos o temps diferentes |
| No guardar historial de drafts | No puedes hacer rollback | Mantén lista de drafts en el estado |
