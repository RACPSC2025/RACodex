# Patrón 03: Planning (Plan-and-Execute)

## Concepto

Separa la **planificación** de la **ejecución**. Un Planner genera un plan estructurado de pasos y luego un Executor los ejecuta secuencialmente, con posibilidad de re-planificar si algo falla.

```
Task → Planner → [Step 1, Step 2, ..., Step N] → Executor(Step 1) → ... → Executor(Step N) → Synthesizer → Answer
```

## Cuándo Usarlo

- Tareas complejas que requieren múltiples pasos coordinados.
- Cuando quieres visibilidad completa del plan antes de ejecutar.
- Proyectos de investigación, análisis multi-etapa, generación de reportes.
- Necesitas aprobación humana del plan antes de ejecutar (Human-in-the-Loop).

---

## Implementación con LangGraph

```python
# patterns/planning_agent.py
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
import operator


# ─── Schemas ──────────────────────────────────────────────────────────────────

class Step(BaseModel):
    step_id: int = Field(description="Número de paso (1-based).")
    description: str = Field(description="Qué hace este paso.")
    tool_to_use: str | None = Field(None, description="Herramienta a usar, si aplica.")
    depends_on: list[int] = Field(default_factory=list, description="IDs de pasos previos requeridos.")


class Plan(BaseModel):
    goal: str = Field(description="El objetivo final del plan.")
    steps: list[Step] = Field(description="Lista ordenada de pasos a ejecutar.")
    estimated_complexity: str = Field(description="low | medium | high")


class StepResult(BaseModel):
    step_id: int
    output: str
    success: bool
    error: str | None = None


class FinalAnswer(BaseModel):
    answer: str = Field(description="Respuesta final sintetizada.")
    sources: list[str] = Field(description="Pasos que contribuyeron a la respuesta.")


# ─── Estado ───────────────────────────────────────────────────────────────────

class PlanExecuteState(TypedDict):
    task: str
    plan: Plan | None
    completed_steps: Annotated[list[StepResult], operator.add]
    current_step_idx: int
    final_answer: str
    replan_count: int
    max_replans: int


# ─── Herramientas de Ejemplo ──────────────────────────────────────────────────

@tool
def search(query: str) -> str:
    """Busca información sobre un tema."""
    return f"Resultados de búsqueda para '{query}': [datos relevantes]"


@tool
def analyze_data(data: str) -> str:
    """Analiza un conjunto de datos y extrae insights."""
    return f"Análisis de '{data}': [insights extraídos]"


@tool
def write_report(content: str, title: str) -> str:
    """Genera un reporte estructurado."""
    return f"Reporte '{title}' generado exitosamente con {len(content)} caracteres."


TOOLS = {t.name: t for t in [search, analyze_data, write_report]}


# ─── LLM Factory ──────────────────────────────────────────────────────────────

def get_llm(provider: str = "openai", temperature: float = 0.0):
    match provider:
        case "openai":    return ChatOpenAI(model="gpt-4o", temperature=temperature)
        case "anthropic": return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=temperature)
        case "google":    return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=temperature)
        case _: raise ValueError(f"Provider no soportado: {provider}")


# ─── Nodos ────────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """Eres un experto en descomposición de tareas complejas.
Crea planes claros, mínimos y ejecutables. Cada paso debe ser atómico y verificable.
Disponibles: search, analyze_data, write_report.
Principio YAGNI: no planees pasos innecesarios."""


def planner_node(state: PlanExecuteState) -> dict:
    """Genera un plan estructurado para la tarea."""
    llm = get_llm().with_structured_output(Plan)

    context = ""
    if state["completed_steps"]:
        context = "\n\nResultados de pasos previos:\n" + "\n".join(
            f"Paso {r.step_id}: {r.output}" for r in state["completed_steps"]
        )

    plan: Plan = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"Tarea: {state['task']}{context}"),
    ])

    print(f"\n📋 Plan generado ({len(plan.steps)} pasos, complejidad: {plan.estimated_complexity})")
    for step in plan.steps:
        tool_info = f" [tool: {step.tool_to_use}]" if step.tool_to_use else ""
        print(f"   Paso {step.step_id}: {step.description}{tool_info}")

    return {
        "plan": plan,
        "current_step_idx": 0,
    }


def executor_node(state: PlanExecuteState) -> dict:
    """Ejecuta el paso actual del plan."""
    plan = state["plan"]
    idx = state["current_step_idx"]

    if idx >= len(plan.steps):
        return {}

    step = plan.steps[idx]
    print(f"\n⚙️  Ejecutando paso {step.step_id}: {step.description}")

    # Construir contexto de pasos previos
    prev_context = "\n".join(
        f"Paso {r.step_id} output: {r.output}"
        for r in state["completed_steps"]
        if r.step_id in step.depends_on
    )

    llm = get_llm(temperature=0.1)

    if step.tool_to_use and step.tool_to_use in TOOLS:
        # Ejecutar herramienta
        try:
            tool_input_prompt = (
                f"Para ejecutar el paso: '{step.description}'\n"
                f"Contexto previo: {prev_context}\n"
                "Genera el input apropiado para la herramienta en texto plano."
            )
            tool_input = llm.invoke([HumanMessage(content=tool_input_prompt)]).content
            output = TOOLS[step.tool_to_use].invoke({"query": tool_input})
            result = StepResult(step_id=step.step_id, output=str(output), success=True)
        except Exception as e:
            result = StepResult(step_id=step.step_id, output="", success=False, error=str(e))
    else:
        # Ejecutar con LLM directamente
        response = llm.invoke([
            SystemMessage(content="Ejecuta el paso de forma precisa y concisa."),
            HumanMessage(content=f"Paso a ejecutar: {step.description}\nContexto: {prev_context}"),
        ])
        result = StepResult(step_id=step.step_id, output=response.content, success=True)

    print(f"   {'✅' if result.success else '❌'} Resultado: {result.output[:80]}...")

    return {
        "completed_steps": [result],
        "current_step_idx": idx + 1,
    }


def synthesizer_node(state: PlanExecuteState) -> dict:
    """Sintetiza todos los resultados en una respuesta final."""
    llm = get_llm().with_structured_output(FinalAnswer)

    steps_summary = "\n".join(
        f"Paso {r.step_id}: {r.output}"
        for r in state["completed_steps"]
        if r.success
    )

    result: FinalAnswer = llm.invoke([
        SystemMessage(content="Sintetiza los resultados en una respuesta completa y coherente."),
        HumanMessage(content=f"Tarea original: {state['task']}\n\nResultados:\n{steps_summary}"),
    ])

    return {"final_answer": result.answer}


# ─── Enrutamiento ─────────────────────────────────────────────────────────────

def route_execution(state: PlanExecuteState) -> str:
    """Determina si continuar ejecutando, re-planificar o sintetizar."""
    plan = state["plan"]
    idx = state["current_step_idx"]

    if idx >= len(plan.steps):
        return "synthesize"

    # Verificar si el último paso falló
    if state["completed_steps"]:
        last = state["completed_steps"][-1]
        if not last.success and state["replan_count"] < state["max_replans"]:
            return "replan"

    return "execute"


# ─── Construcción del Grafo ────────────────────────────────────────────────────

def build_plan_execute_graph():
    graph = StateGraph(PlanExecuteState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges(
        "executor",
        route_execution,
        {
            "execute": "executor",
            "synthesize": "synthesizer",
            "replan": "planner",
        },
    )
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ─── Uso ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_plan_execute_graph()

    result = app.invoke({
        "task": "Investiga el estado actual de LangGraph en 2025, analiza sus casos de uso principales y genera un reporte ejecutivo.",
        "plan": None,
        "completed_steps": [],
        "current_step_idx": 0,
        "final_answer": "",
        "replan_count": 0,
        "max_replans": 2,
    })

    print("\n" + "="*60)
    print("RESPUESTA FINAL:")
    print(result["final_answer"])
```

---

## Variante: Planning con Human-in-the-Loop

```python
from langgraph.checkpoint.memory import MemorySaver

# Agrega interrupt ANTES del executor para aprobación humana
graph.add_edge("planner", "human_approval")  # nodo de pausa
graph.add_conditional_edges("human_approval", human_approves, {...})

# Compilar con checkpointer para persistir estado
memory = MemorySaver()
app = graph.compile(checkpointer=memory, interrupt_before=["executor"])

# Flujo:
# 1. Invocar → genera plan → pausa
# 2. Humano revisa plan impreso
# 3. app.invoke(None, config) → continúa ejecución
```
