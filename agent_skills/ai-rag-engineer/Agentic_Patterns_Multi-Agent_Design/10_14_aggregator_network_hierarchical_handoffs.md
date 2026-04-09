# Patrones Multi-Agente: Aggregator · Network · Hierarchical · Subagents · Handoffs

---

# Patrón 10: Aggregator Agent

## Concepto

Consolida y sintetiza outputs de **múltiples fuentes o agentes** en un resultado unificado y coherente.

```
[Agent A result] ─┐
[Agent B result] ─┤→ Aggregator → Unified Output
[Agent C result] ─┘
```

```python
# patterns/aggregator_agent.py
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
import operator


class AgentOutput(BaseModel):
    source: str
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class AggregatorState(TypedDict):
    query: str
    sources: list[str]
    agent_outputs: Annotated[list[AgentOutput], operator.add]
    aggregated_result: str
    conflicts_detected: bool


class WorkerState(TypedDict):
    query: str
    source: str
    agent_outputs: list[AgentOutput]


class AggregatedOutput(BaseModel):
    unified_answer: str = Field(description="Respuesta unificada y coherente.")
    conflicts: list[str] = Field(description="Conflictos o contradicciones detectadas entre fuentes.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confianza general del resultado.")
    sources_used: list[str] = Field(description="Fuentes que contribuyeron al resultado final.")


llm = ChatOpenAI(model="gpt-4o", temperature=0)


def dispatch_agents(state: AggregatorState) -> list[Send]:
    """Fan-out: cada source recibe su propio agente worker."""
    print(f"\n📡 Dispatching a {len(state['sources'])} fuentes en paralelo")
    return [
        Send("source_agent", {"query": state["query"], "source": src, "agent_outputs": []})
        for src in state["sources"]
    ]


def source_agent_node(state: WorkerState) -> dict:
    """Agente especializado en una fuente específica."""
    print(f"   🔍 Consultando fuente: {state['source']}")
    response = llm.invoke([
        SystemMessage(content=f"Eres un especialista en {state['source']}. Proporciona información precisa y confiable."),
        HumanMessage(content=state["query"]),
    ])
    output = AgentOutput(
        source=state["source"],
        content=response.content,
        confidence=0.85,  # En producción: calcular confidence real
    )
    return {"agent_outputs": [output]}


def aggregator_node(state: AggregatorState) -> dict:
    """Consolida todos los outputs en una respuesta unificada."""
    print(f"\n🔀 Aggregator: consolidando {len(state['agent_outputs'])} outputs")

    sources_text = "\n\n".join(
        f"=== Fuente: {o.source} (confianza: {o.confidence:.0%}) ===\n{o.content}"
        for o in state["agent_outputs"]
    )

    result: AggregatedOutput = llm.with_structured_output(AggregatedOutput).invoke([
        SystemMessage(content=(
            "Eres un experto en síntesis de información. "
            "Consolida estas respuestas de múltiples fuentes en una respuesta unificada. "
            "Detecta conflictos, evalúa confiabilidad y prioriza las fuentes más fiables."
        )),
        HumanMessage(content=f"Consulta: {state['query']}\n\nRespuestas:\n{sources_text}"),
    ])

    print(f"   {'⚠️ Conflictos detectados' if result.conflicts else '✅ Sin conflictos'}")
    return {
        "aggregated_result": result.unified_answer,
        "conflicts_detected": len(result.conflicts) > 0,
    }


def build_aggregator_graph():
    graph = StateGraph(AggregatorState)
    graph.add_node("dispatch", dispatch_agents)
    graph.add_node("source_agent", source_agent_node)
    graph.add_node("aggregator", aggregator_node)

    graph.add_edge(START, "dispatch")
    graph.add_edge("dispatch", "source_agent")
    graph.add_edge("source_agent", "aggregator")
    graph.add_edge("aggregator", END)
    return graph.compile()
```

---

# Patrón 11: Network Agent (Malla de Agentes)

## Concepto

Agentes que pueden **comunicarse entre sí libremente** sin jerarquía fija. Cada agente decide con quién colaborar.

```
Agent A ←→ Agent B
    ↕           ↕
Agent C ←→ Agent D
```

```python
# patterns/network_agent.py
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import random


AgentRole = Literal["analyst", "researcher", "critic", "synthesizer"]


class NetworkState(TypedDict):
    task: str
    messages: Annotated[list, add_messages]
    round: int
    max_rounds: int
    final_output: str


AGENT_PERSONAS: dict[AgentRole, str] = {
    "analyst": "Eres un analista que examina datos y extrae patrones. Critica los puntos débiles de los demás.",
    "researcher": "Eres un investigador que aporta contexto y evidencia. Complementas lo que otros han dicho.",
    "critic": "Eres un crítico riguroso que identifica fallas y propone mejoras. Sé constructivo pero directo.",
    "synthesizer": "Eres un sintetizador que integra perspectivas diversas en conclusiones coherentes.",
}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def create_agent_node(role: AgentRole):
    """Factory: crea un nodo de agente para un rol específico."""
    def agent_node(state: NetworkState) -> dict:
        # Contexto de la conversación entre agentes
        conversation = "\n".join(
            f"{m.type.upper()}: {m.content[:200]}"
            for m in state["messages"][-6:]  # Últimos 6 mensajes
        )
        response = llm.invoke([
            SystemMessage(content=AGENT_PERSONAS[role]),
            HumanMessage(content=(
                f"Tarea del equipo: {state['task']}\n\n"
                f"Conversación hasta ahora:\n{conversation}\n\n"
                f"Aporta tu perspectiva como {role}. Sé conciso (máx 150 palabras)."
            )),
        ])
        print(f"   💬 [{role.upper()}]: {response.content[:80]}...")
        return {
            "messages": [AIMessage(content=f"[{role.upper()}]: {response.content}")],
            "round": state["round"] + 1,
        }
    return agent_node


def synthesizer_final_node(state: NetworkState) -> dict:
    """Síntesis final de toda la conversación de la red."""
    conversation = "\n".join(m.content for m in state["messages"])
    final = llm.invoke([
        SystemMessage(content="Sintetiza esta discusión multi-agente en una conclusión definitiva."),
        HumanMessage(content=f"Tarea: {state['task']}\n\nDiscusión:\n{conversation}"),
    ]).content
    return {"final_output": final}


def route_network(state: NetworkState) -> str:
    """Enruta de forma round-robin entre agentes o finaliza."""
    if state["round"] >= state["max_rounds"]:
        return "final_synthesis"
    # Round-robin entre analista, investigador y crítico
    roles = ["analyst", "researcher", "critic"]
    return roles[state["round"] % len(roles)]


def build_network_graph():
    graph = StateGraph(NetworkState)
    for role in ["analyst", "researcher", "critic"]:
        graph.add_node(role, create_agent_node(role))
    graph.add_node("final_synthesis", synthesizer_final_node)

    graph.add_edge(START, "analyst")
    for role in ["analyst", "researcher", "critic"]:
        graph.add_conditional_edges(
            role, route_network,
            {"analyst": "analyst", "researcher": "researcher",
             "critic": "critic", "final_synthesis": "final_synthesis"},
        )
    graph.add_edge("final_synthesis", END)
    return graph.compile()
```

---

# Patrón 12 & 13: Hierarchical Agents + Subagents

## Concepto

Un **Supervisor** orquesta múltiples **Workers** especializados. Los Subagents son agentes completos invocados como herramientas.

```
Supervisor
├── Worker Agent A (especialista en X)
├── Worker Agent B (especialista en Y)  ← invocados como tools o Send()
└── Worker Agent C (especialista en Z)
```

```python
# patterns/hierarchical_agents.py
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
import operator


WorkerType = Literal["code_agent", "research_agent", "review_agent"]


class SupervisorDecision(BaseModel):
    next_worker: WorkerType | Literal["FINISH"] = Field(
        description="Qué worker invocar a continuación, o FINISH si está completo."
    )
    task_for_worker: str = Field(description="Instrucción específica para el worker.")
    reasoning: str = Field(description="Por qué elegiste este worker.")


class HierarchicalState(TypedDict):
    original_task: str
    supervisor_instructions: str
    worker_results: Annotated[list[dict], operator.add]
    next_worker: WorkerType | Literal["FINISH"]
    final_output: str
    iterations: int


class WorkerState(TypedDict):
    task: str
    worker_type: WorkerType
    context: str
    worker_results: list[dict]


WORKER_PROMPTS: dict[WorkerType, str] = {
    "code_agent": "Eres un senior Python engineer. Escribe código limpio, tipado y con type hints.",
    "research_agent": "Eres un investigador técnico. Provee información precisa con fuentes cuando sea posible.",
    "review_agent": "Eres un code reviewer experto. Revisa código y detecta bugs, mejoras de SOLID y seguridad.",
}

supervisor_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(SupervisorDecision)
worker_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)


def supervisor_node(state: HierarchicalState) -> dict:
    """Supervisor: analiza progreso y decide próximo worker."""
    results_summary = "\n".join(
        f"- [{r['worker']}]: {r['result'][:100]}..."
        for r in state.get("worker_results", [])
    )

    decision: SupervisorDecision = supervisor_llm.invoke([
        SystemMessage(content=(
            "Eres un Tech Lead que orquesta un equipo. "
            "Workers disponibles: code_agent, research_agent, review_agent. "
            "Decide quién debe actuar a continuación o si la tarea está completa (FINISH)."
        )),
        HumanMessage(content=(
            f"Tarea: {state['original_task']}\n\n"
            f"Trabajo completado:\n{results_summary or 'Ninguno aún'}"
        )),
    ])

    print(f"\n🎯 Supervisor → {decision.next_worker}: {decision.task_for_worker[:60]}")
    return {
        "next_worker": decision.next_worker,
        "supervisor_instructions": decision.task_for_worker,
        "iterations": state["iterations"] + 1,
    }


def create_worker_node(worker_type: WorkerType):
    def worker_node(state: HierarchicalState) -> dict:
        response = worker_llm.invoke([
            SystemMessage(content=WORKER_PROMPTS[worker_type]),
            HumanMessage(content=(
                f"Instrucción del supervisor: {state['supervisor_instructions']}\n"
                f"Tarea general: {state['original_task']}"
            )),
        ]).content
        print(f"   👷 [{worker_type}] completado")
        return {
            "worker_results": [{"worker": worker_type, "result": response}],
        }
    return worker_node


def finalize_node(state: HierarchicalState) -> dict:
    all_results = "\n\n".join(
        f"[{r['worker']}]:\n{r['result']}" for r in state["worker_results"]
    )
    final = worker_llm.invoke([
        SystemMessage(content="Consolida el trabajo del equipo en un output final cohesivo."),
        HumanMessage(content=f"Tarea: {state['original_task']}\n\nResultados:\n{all_results}"),
    ]).content
    return {"final_output": final}


def route_supervisor(state: HierarchicalState) -> str:
    if state["next_worker"] == "FINISH" or state["iterations"] >= 5:
        return "finalize"
    return state["next_worker"]


def build_hierarchical_graph():
    graph = StateGraph(HierarchicalState)
    graph.add_node("supervisor", supervisor_node)
    for wt in ["code_agent", "research_agent", "review_agent"]:
        graph.add_node(wt, create_worker_node(wt))
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", route_supervisor,
        {"code_agent": "code_agent", "research_agent": "research_agent",
         "review_agent": "review_agent", "finalize": "finalize"},
    )
    for wt in ["code_agent", "research_agent", "review_agent"]:
        graph.add_edge(wt, "supervisor")
    graph.add_edge("finalize", END)
    return graph.compile()
```

---

# Patrón 14: Handoffs (Transferencia de Control)

## Concepto

Un agente **transfiere explícitamente** el control a otro agente con contexto completo. Diferente al Router (que solo enruta la query): el Handoff transfiere el estado de conversación completo.

```
Agent A (handles inicial) → Handoff(context, reason) → Agent B (continúa)
```

```python
# patterns/handoffs.py
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent


class HandoffRequest(BaseModel):
    target_agent: str = Field(description="Nombre del agente destino.")
    reason: str = Field(description="Por qué se hace el handoff.")
    context_summary: str = Field(description="Resumen del contexto para el agente destino.")
    priority: Literal["low", "medium", "high", "urgent"] = "medium"


class HandoffState(TypedDict):
    messages: Annotated[list, add_messages]
    current_agent: str
    handoff_context: str
    resolved: bool


llm = ChatOpenAI(model="gpt-4o", temperature=0)


def create_handoff_tool(target_agent: str, description: str):
    """Factory: crea una tool de handoff hacia un agente específico."""
    @tool(name=f"handoff_to_{target_agent}")
    def handoff_tool(reason: str, context_summary: str) -> str:
        f"""
        {description}
        Usa cuando necesites transferir esta conversación a {target_agent}.
        """
        return f"HANDOFF_REQUESTED:{target_agent}:{reason}:{context_summary}"
    return handoff_tool


def create_agent_with_handoffs(
    agent_name: str,
    system_prompt: str,
    available_handoffs: list[str],
) -> object:
    """Crea un agente con capacidad de hacer handoffs."""
    handoff_tools = [
        create_handoff_tool(
            target,
            f"Transfiere a {target} cuando la consulta requiere su especialidad."
        )
        for target in available_handoffs
    ]
    agent_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return create_react_agent(agent_llm, handoff_tools, prompt=system_prompt)


# Agentes especializados con handoff capabilities
AGENTS = {
    "triage": create_agent_with_handoffs(
        "triage",
        "Eres el agente de triaje. Clasifica la consulta y deriva al especialista correcto.",
        ["technical_support", "billing", "escalation"],
    ),
    "technical_support": create_agent_with_handoffs(
        "technical_support",
        "Eres soporte técnico senior. Resuelves problemas técnicos. Si requiere facturación, transfiere.",
        ["billing", "escalation"],
    ),
    "billing": create_agent_with_handoffs(
        "billing",
        "Eres el agente de facturación. Manejas pagos y suscripciones. Escala disputas complejas.",
        ["escalation"],
    ),
    "escalation": create_agent_with_handoffs(
        "escalation",
        "Eres el supervisor de escalación. Manejas casos complejos que otros no pudieron resolver.",
        [],  # Sin más handoffs — este es el final
    ),
}


def agent_node(state: HandoffState) -> dict:
    """Ejecuta el agente actual y detecta handoff requests."""
    current_agent_name = state["current_agent"]
    agent = AGENTS.get(current_agent_name)

    if not agent:
        return {"messages": [AIMessage(content=f"Agente '{current_agent_name}' no encontrado.")], "resolved": True}

    print(f"\n🤝 [{current_agent_name.upper()}] procesando...")

    # Preparar mensajes con contexto de handoff si existe
    messages = list(state["messages"])
    if state["handoff_context"]:
        messages.insert(0, SystemMessage(content=f"Contexto recibido: {state['handoff_context']}"))

    result = agent.invoke({"messages": messages})
    last_message = result["messages"][-1].content

    # Detectar si hay un handoff request en el output
    if "HANDOFF_REQUESTED:" in last_message:
        parts = last_message.split(":")
        target = parts[1]
        reason = parts[2] if len(parts) > 2 else ""
        context = parts[3] if len(parts) > 3 else ""
        print(f"   ↗️  Handoff → {target}: {reason[:50]}")
        return {
            "messages": [AIMessage(content=f"Transfiriendo a {target}: {reason}")],
            "current_agent": target,
            "handoff_context": context,
        }

    return {
        "messages": [AIMessage(content=last_message)],
        "resolved": True,
    }


def should_continue_handoff(state: HandoffState) -> Literal["continue", "end"]:
    return "end" if state["resolved"] else "continue"


def build_handoff_graph():
    graph = StateGraph(HandoffState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue_handoff,
        {"continue": "agent", "end": END},
    )
    return graph.compile()


if __name__ == "__main__":
    app = build_handoff_graph()
    result = app.invoke({
        "messages": [HumanMessage(content="Mi pago falló pero necesito acceso a la plataforma urgente.")],
        "current_agent": "triage",
        "handoff_context": "",
        "resolved": False,
    })
    print("\n" + "="*50)
    print("RESOLUCIÓN FINAL:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"  {msg.content[:200]}")
```

---

## Comparativa de Patrones Multi-Agente

| Patrón | Complejidad | Latencia | Uso ideal |
|--------|-------------|----------|-----------|
| Sequential | Baja | Alta | Pipelines ETL, generación de contenido |
| Parallel | Media | Baja | Análisis multi-fuente, tareas independientes |
| Loop | Media | Variable | Optimización iterativa, validación con retry |
| Router | Media | Baja | Clasificación de consultas, multi-dominio |
| Aggregator | Media-Alta | Baja | Consolidación de múltiples opiniones/fuentes |
| Network | Alta | Media | Discusión colaborativa, brainstorming |
| Hierarchical | Alta | Media | Sistemas enterprise, workflows complejos |
| Handoffs | Media | Baja | Customer service, soporte multi-nivel |
