# Patrones Multi-Agente: Parallel · Sequential · Loop · Router

---

# Patrón 06: Parallel Agents

## Concepto

Múltiples agentes ejecutan **simultáneamente** tareas independientes. Un Fan-Out distribuye trabajo, un Fan-In consolida resultados.

```
Task → Fan-Out → [Agent A ∥ Agent B ∥ Agent C] → Fan-In → Aggregated Result
```

```python
# patterns/parallel_agents.py
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
import operator


class ParallelState(TypedDict):
    task: str
    subtasks: list[str]
    results: Annotated[list[dict], operator.add]  # Acumulador thread-safe
    final_summary: str


class SubtaskState(TypedDict):
    task: str
    subtask: str
    result: str


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)


def fan_out_node(state: ParallelState) -> list[Send]:
    """
    Distribuye el trabajo entre agentes paralelos.
    Retorna lista de Send() — LangGraph los ejecuta en paralelo.
    """
    # En producción: las subtasks podrían venir del plan o del propio LLM
    subtasks = state["subtasks"]
    print(f"\n🔀 Fan-out: {len(subtasks)} tareas en paralelo")
    return [
        Send("worker_agent", {"task": state["task"], "subtask": st, "result": ""})
        for st in subtasks
    ]


def worker_agent_node(state: SubtaskState) -> dict:
    """Agente worker que ejecuta una subtarea específica."""
    print(f"   ⚙️  Worker procesando: {state['subtask'][:50]}")
    response = llm.invoke([
        SystemMessage(content=f"Eres un especialista. Tu tarea específica es: {state['subtask']}"),
        HumanMessage(content=f"Contexto general: {state['task']}"),
    ])
    return {"results": [{"subtask": state["subtask"], "result": response.content}]}


def fan_in_node(state: ParallelState) -> dict:
    """Consolida los resultados de todos los workers."""
    print(f"\n🔁 Fan-in: consolidando {len(state['results'])} resultados")
    results_text = "\n\n".join(
        f"### {r['subtask']}\n{r['result']}"
        for r in state["results"]
    )
    summary = llm.invoke([
        SystemMessage(content="Sintetiza estos resultados en un reporte cohesivo y bien estructurado."),
        HumanMessage(content=f"Tarea original: {state['task']}\n\nResultados:\n{results_text}"),
    ]).content
    return {"final_summary": summary}


def build_parallel_graph():
    graph = StateGraph(ParallelState)
    graph.add_node("fan_out", fan_out_node)
    graph.add_node("worker_agent", worker_agent_node)
    graph.add_node("fan_in", fan_in_node)

    graph.add_edge(START, "fan_out")
    graph.add_edge("fan_out", "worker_agent")  # LangGraph maneja el Send automáticamente
    graph.add_edge("worker_agent", "fan_in")
    graph.add_edge("fan_in", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_parallel_graph()
    result = app.invoke({
        "task": "Analiza el ecosistema de frameworks de agentes en Python 2025",
        "subtasks": [
            "Analiza LangGraph: ventajas, limitaciones y casos de uso",
            "Analiza CrewAI: ventajas, limitaciones y casos de uso",
            "Analiza AutoGen: ventajas, limitaciones y casos de uso",
        ],
        "results": [],
        "final_summary": "",
    })
    print(result["final_summary"])
```

---

# Patrón 07: Sequential Agents

## Concepto

Agentes especializados en **pipeline lineal**. El output de cada agente es el input del siguiente. Ideal para workflows de transformación de datos.

```
Input → Agent A → Agent B → Agent C → Output
```

```python
# patterns/sequential_agents.py
from typing import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


class PipelineState(TypedDict):
    raw_input: str
    researched_content: str
    structured_outline: str
    draft_content: str
    reviewed_content: str
    final_output: str


llm = ChatOpenAI(model="gpt-4o", temperature=0.1)


def researcher_agent(state: PipelineState) -> dict:
    """Agente 1: Investiga y recopila información."""
    print("🔍 Researcher: recopilando información...")
    result = llm.invoke([
        SystemMessage(content="Eres un investigador experto. Recopila y sintetiza información relevante y actualizada."),
        HumanMessage(content=f"Investiga en profundidad: {state['raw_input']}"),
    ]).content
    return {"researched_content": result}


def outliner_agent(state: PipelineState) -> dict:
    """Agente 2: Crea estructura del contenido."""
    print("📋 Outliner: estructurando contenido...")
    result = llm.invoke([
        SystemMessage(content="Eres un arquitecto de contenido. Crea outlines claros, lógicos y bien jerarquizados."),
        HumanMessage(content=f"Crea un outline detallado basado en:\n{state['researched_content']}"),
    ]).content
    return {"structured_outline": result}


def writer_agent(state: PipelineState) -> dict:
    """Agente 3: Redacta el contenido."""
    print("✍️  Writer: redactando...")
    result = llm.invoke([
        SystemMessage(content="Eres un escritor técnico experto. Redacta contenido claro, preciso y profesional."),
        HumanMessage(content=f"Redacta siguiendo este outline:\n{state['structured_outline']}"),
    ]).content
    return {"draft_content": result}


def reviewer_agent(state: PipelineState) -> dict:
    """Agente 4: Revisa y mejora el draft."""
    print("🔎 Reviewer: revisando calidad...")
    result = llm.invoke([
        SystemMessage(content="Eres un editor senior. Mejora claridad, corrección técnica y estilo. Produce la versión final."),
        HumanMessage(content=f"Revisa y mejora:\n{state['draft_content']}"),
    ]).content
    return {"reviewed_content": result, "final_output": result}


def build_sequential_pipeline():
    graph = StateGraph(PipelineState)
    for name, fn in [
        ("researcher", researcher_agent),
        ("outliner", outliner_agent),
        ("writer", writer_agent),
        ("reviewer", reviewer_agent),
    ]:
        graph.add_node(name, fn)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "outliner")
    graph.add_edge("outliner", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_edge("reviewer", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_sequential_pipeline()
    result = app.invoke({
        "raw_input": "Mejores prácticas de seguridad en APIs con FastAPI en 2025",
        "researched_content": "", "structured_outline": "",
        "draft_content": "", "reviewed_content": "", "final_output": "",
    })
    print("\nOUTPUT FINAL:\n" + result["final_output"])
```

---

# Patrón 08: Loop Agent

## Concepto

El agente **itera** sobre su tarea hasta satisfacer una condición de convergencia o alcanzar el límite de iteraciones.

```
Task → Agent → Check Condition → [Met → End] | [Not Met → Agent → Check → ...]
```

```python
# patterns/loop_agent.py
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END


class LoopState(TypedDict):
    task: str
    current_solution: str
    quality_score: float
    iteration: int
    max_iterations: int
    convergence_threshold: float
    history: list[str]


class QualityEvaluation(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Puntuación de calidad entre 0 y 1.")
    meets_threshold: bool = Field(description="True si supera el umbral de calidad.")
    improvement_needed: str = Field(description="Qué mejorar en la próxima iteración.")


llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


def refine_node(state: LoopState) -> dict:
    """Genera o mejora la solución."""
    history_ctx = ""
    if state["history"]:
        last = state["history"][-1]
        history_ctx = f"\n\nVersión anterior (score={state['quality_score']:.2f}):\n{last}\n\nMejora esta versión."

    result = llm.invoke([
        SystemMessage(content="Eres un experto que mejora soluciones iterativamente hasta la perfección."),
        HumanMessage(content=f"Tarea: {state['task']}{history_ctx}"),
    ]).content

    print(f"\n🔄 Iteración {state['iteration'] + 1}: solución generada")
    return {
        "current_solution": result,
        "iteration": state["iteration"] + 1,
        "history": state["history"] + [result],
    }


def evaluate_node(state: LoopState) -> dict:
    """Evalúa la calidad de la solución actual."""
    evaluation: QualityEvaluation = llm.with_structured_output(QualityEvaluation).invoke([
        SystemMessage(content=(
            f"Evalúa la calidad de esta solución para la tarea dada. "
            f"El umbral de aprobación es {state['convergence_threshold']:.0%}."
        )),
        HumanMessage(content=f"Tarea: {state['task']}\n\nSolución:\n{state['current_solution']}"),
    ])
    print(f"   📊 Score: {evaluation.score:.2f} | ¿Aprobado?: {evaluation.meets_threshold}")
    return {
        "quality_score": evaluation.score,
        "task": state["task"] + (
            f"\n[Mejora requerida: {evaluation.improvement_needed}]"
            if not evaluation.meets_threshold else ""
        ),
    }


def should_continue_loop(state: LoopState) -> Literal["refine", "end"]:
    if (
        state["quality_score"] >= state["convergence_threshold"]
        or state["iteration"] >= state["max_iterations"]
    ):
        reason = "calidad alcanzada" if state["quality_score"] >= state["convergence_threshold"] else "límite de iteraciones"
        print(f"\n✅ Loop terminado ({reason}) tras {state['iteration']} iteración(es)")
        return "end"
    return "refine"


def build_loop_graph():
    graph = StateGraph(LoopState)
    graph.add_node("refine", refine_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_edge(START, "refine")
    graph.add_edge("refine", "evaluate")
    graph.add_conditional_edges("evaluate", should_continue_loop, {"refine": "refine", "end": END})
    return graph.compile()


if __name__ == "__main__":
    app = build_loop_graph()
    result = app.invoke({
        "task": "Escribe una función Python para validar emails con regex, type hints y tests.",
        "current_solution": "", "quality_score": 0.0,
        "iteration": 0, "max_iterations": 4,
        "convergence_threshold": 0.85, "history": [],
    })
    print("\nSOLUCIÓN FINAL:\n" + result["current_solution"])
```

---

# Patrón 09: Router Agent

## Concepto

Un agente **clasificador** analiza la consulta y la dirige al agente especialista más adecuado.

```
Query → Router → [Technical Agent | Creative Agent | Data Agent | Support Agent]
```

```python
# patterns/router_agent.py
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END


AgentType = Literal["technical", "creative", "data_analysis", "customer_support", "unknown"]


class RouterDecision(BaseModel):
    agent_type: AgentType = Field(description="Tipo de agente más adecuado para la consulta.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confianza en la decisión (0-1).")
    reasoning: str = Field(description="Por qué este agente es el más adecuado.")


class RouterState(TypedDict):
    query: str
    routed_to: AgentType
    response: str
    routing_confidence: float


SPECIALIST_PROMPTS: dict[AgentType, str] = {
    "technical": "Eres un ingeniero senior experto en Python, arquitectura de software y sistemas. Responde con precisión técnica.",
    "creative": "Eres un experto en escritura creativa, marketing y comunicación. Responde con creatividad y claridad.",
    "data_analysis": "Eres un data scientist experto en análisis estadístico, visualización y ML. Responde con rigor analítico.",
    "customer_support": "Eres un agente de soporte empático y solucionador de problemas. Responde con claridad y amabilidad.",
    "unknown": "Eres un asistente general. Responde de la mejor forma posible.",
}

llm = ChatOpenAI(model="gpt-4o", temperature=0)
router_llm = llm.with_structured_output(RouterDecision)


def router_node(state: RouterState) -> dict:
    """Clasifica la consulta y decide el agente destino."""
    decision: RouterDecision = router_llm.invoke([
        SystemMessage(content=(
            "Clasifica esta consulta en uno de estos tipos: "
            "technical (código, arquitectura, sistemas), "
            "creative (escritura, marketing, diseño), "
            "data_analysis (estadística, ML, visualización), "
            "customer_support (soporte, quejas, información). "
            "Si no encaja en ninguno: unknown."
        )),
        HumanMessage(content=state["query"]),
    ])
    print(f"\n🧭 Router → {decision.agent_type} (confianza: {decision.confidence:.0%})")
    return {"routed_to": decision.agent_type, "routing_confidence": decision.confidence}


def specialist_node(state: RouterState) -> dict:
    """Agente especialista — el mismo nodo sirve a todos los tipos vía estado."""
    agent_type = state["routed_to"]
    system_prompt = SPECIALIST_PROMPTS.get(agent_type, SPECIALIST_PROMPTS["unknown"])
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["query"]),
    ]).content
    print(f"   💬 [{agent_type}] respondiendo...")
    return {"response": response}


def route_to_specialist(state: RouterState) -> AgentType:
    return state["routed_to"]


def build_router_graph():
    graph = StateGraph(RouterState)
    graph.add_node("router", router_node)

    # Un nodo por especialista (en producción pueden ser grafos completos)
    for agent_type in AgentType.__args__:
        graph.add_node(agent_type, specialist_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_to_specialist,
        {t: t for t in AgentType.__args__},
    )
    for agent_type in AgentType.__args__:
        graph.add_edge(agent_type, END)

    return graph.compile()


if __name__ == "__main__":
    app = build_router_graph()
    queries = [
        "¿Cómo implemento un singleton thread-safe en Python?",
        "Escribe un slogan para una startup de IA en salud.",
        "Analiza la correlación entre estas variables: temperatura y ventas.",
    ]
    for q in queries:
        result = app.invoke({"query": q, "routed_to": "unknown", "response": "", "routing_confidence": 0.0})
        print(f"\n[{result['routed_to']}] {result['response'][:100]}...\n")
```
