# Patrón 01: ReAct (Reasoning + Acting)

## Concepto

ReAct intercala pasos de **razonamiento** (Thought) con **acciones** (Act) y **observaciones** (Observe) en un ciclo continuo. El modelo "piensa en voz alta" antes de cada acción, lo que mejora drásticamente la calidad de las decisiones.

```
Thought → Action → Observation → Thought → Action → Observation → ... → Final Answer
```

## Cuándo Usarlo

- El agente necesita múltiples herramientas para resolver una tarea.
- El problema requiere decisiones encadenadas donde cada paso depende del anterior.
- Necesitas trazabilidad completa del razonamiento.
- Tareas de Q&A complejo, investigación, debugging.

## Cuándo NO Usarlo

- Tareas deterministas sin necesidad de razonamiento dinámico.
- Cuando el número de pasos es fijo y conocido de antemano (usa Planning).
- Latencia crítica donde el paso de razonamiento extra es costoso.

---

## Implementación con LangGraph

```python
# patterns/react_agent.py
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import httpx
import json


# ─── Estado ───────────────────────────────────────────────────────────────────

class ReActState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    max_iterations: int


# ─── Herramientas ─────────────────────────────────────────────────────────────

@tool
def search_web(query: str) -> str:
    """Busca información en la web sobre un tema dado."""
    # En producción: integrar con Tavily, SerpAPI, etc.
    return f"[Resultado de búsqueda para '{query}']: Información relevante encontrada."


@tool
def calculate(expression: str) -> str:
    """Evalúa una expresión matemática de forma segura."""
    try:
        # NOTA: En producción usar ast.literal_eval o sympy, nunca eval() directo
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "Error: expresión contiene caracteres no permitidos."
        result = eval(expression)  # noqa: S307 — solo en ejemplo controlado
        return str(result)
    except Exception as e:
        return f"Error al calcular: {e}"


@tool
def get_weather(city: str) -> str:
    """Obtiene el clima actual de una ciudad."""
    # En producción: conectar con OpenWeatherMap, WeatherAPI, etc.
    return f"El clima en {city} es 22°C, parcialmente nublado."


TOOLS = [search_web, calculate, get_weather]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ─── Nodos ────────────────────────────────────────────────────────────────────

def get_react_llm(provider: str = "openai"):
    """Retorna el LLM con tools bindeadas según el proveedor."""
    match provider:
        case "openai":
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
        case "anthropic":
            llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
        case "google":
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
        case _:
            raise ValueError(f"Provider no soportado: {provider}")
    return llm.bind_tools(TOOLS)


def agent_node(state: ReActState) -> dict:
    """
    Nodo principal: el LLM razona y decide la siguiente acción.
    El 'pensamiento' está implícito en el reasoning del modelo.
    """
    llm_with_tools = get_react_llm()
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response],
        "iterations": state["iterations"] + 1,
    }


def tools_node(state: ReActState) -> dict:
    """Ejecuta las herramientas solicitadas por el agente."""
    last_message = state["messages"][-1]
    tool_results: list[ToolMessage] = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name not in TOOLS_BY_NAME:
            result = f"Error: herramienta '{tool_name}' no encontrada."
        else:
            try:
                result = TOOLS_BY_NAME[tool_name].invoke(tool_args)
            except Exception as e:
                result = f"Error ejecutando {tool_name}: {e}"

        tool_results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
        )

    return {"messages": tool_results}


# ─── Enrutamiento ─────────────────────────────────────────────────────────────

def should_continue(state: ReActState) -> Literal["tools", "end"]:
    """Decide si continuar con herramientas o finalizar."""
    last_message = state["messages"][-1]

    # Límite de seguridad contra loops infinitos
    if state["iterations"] >= state["max_iterations"]:
        return "end"

    # Si hay tool_calls, ejecutar herramientas
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"


# ─── Construcción del Grafo ────────────────────────────────────────────────────

def build_react_graph() -> object:
    """Construye y compila el grafo ReAct."""
    graph = StateGraph(ReActState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")  # Loop: tools → agent

    return graph.compile()


# ─── Uso ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_react_graph()

    result = app.invoke({
        "messages": [
            HumanMessage(content=(
                "¿Cuánto es 15% de 2847? "
                "Además, busca información sobre LangGraph y dime el clima en Madrid."
            ))
        ],
        "iterations": 0,
        "max_iterations": 10,
    })

    # Imprimir trayectoria completa
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n🤖 AI: {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"   🔧 Tool call: {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            print(f"   📋 Tool result [{msg.name}]: {msg.content[:100]}...")
        elif isinstance(msg, HumanMessage):
            print(f"\n👤 Human: {msg.content}")
```

---

## Variante: ReAct con Razonamiento Explícito

Para modelos que NO soportan tool calling nativo, se puede implementar ReAct con parsing de texto:

```python
# Prompt estructurado para razonamiento explícito
REACT_SYSTEM_PROMPT = """Eres un agente que resuelve problemas paso a paso.

Para cada paso, usa EXACTAMENTE este formato:
Thought: [tu razonamiento sobre qué hacer]
Action: [nombre_herramienta]
Action Input: [input en JSON]

Cuando tengas la respuesta final:
Thought: Tengo suficiente información para responder.
Final Answer: [tu respuesta completa]

Herramientas disponibles:
{tools_description}
"""
```

---

## Anti-Patterns a Evitar

| Anti-pattern | Problema | Solución |
|---|---|---|
| Sin límite de iteraciones | Loop infinito en producción | Siempre define `max_iterations` |
| Tool errors silenciados | El agente no sabe que falló | Retorna error explícito como ToolMessage |
| Estado sin mensajes acumulados | Pierde contexto entre pasos | Usa `add_messages` annotation |
| `eval()` en calculadora | Vulnerabilidad de seguridad | Usa `sympy` o expresiones restringidas |
| Sin timeout en tools | Cuelga en herramientas lentas | Wrap con `asyncio.wait_for` o `tenacity` |

---

## Observabilidad

```python
# Agregar tracing con LangSmith
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "react-agent-prod"
# Cada invocación genera un trace completo en LangSmith
```
