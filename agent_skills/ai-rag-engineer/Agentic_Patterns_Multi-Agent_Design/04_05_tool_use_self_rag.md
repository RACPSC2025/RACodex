# Patrón 04: Tool Use Avanzado

## Concepto

Diseño profesional de herramientas: schemas claros, ejecución paralela, manejo de errores robusto, versioning y herramientas anidadas (agentes como tools).

---

## Herramientas de Producción

```python
# patterns/tool_use_advanced.py
from typing import Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool, StructuredTool
from langchain_core.tools.base import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential


# ─── Tool con Schema Pydantic (recomendado en producción) ─────────────────────

class SearchInput(BaseModel):
    query: str = Field(description="Consulta de búsqueda en lenguaje natural.")
    max_results: int = Field(default=5, ge=1, le=20, description="Número máximo de resultados.")
    language: str = Field(default="es", description="Idioma de búsqueda (ISO 639-1).")


class DatabaseQueryInput(BaseModel):
    table: str = Field(description="Nombre de la tabla a consultar.")
    filters: dict[str, Any] = Field(default_factory=dict, description="Filtros como dict clave-valor.")
    limit: int = Field(default=10, ge=1, le=100)


@tool(args_schema=SearchInput)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def search_knowledge_base(query: str, max_results: int = 5, language: str = "es") -> str:
    """
    Busca en la base de conocimiento interna de la empresa.
    Ideal para encontrar documentación, políticas y procedimientos.
    """
    # En producción: integrar con Pinecone, Weaviate, pgvector, etc.
    return f"[KB Search] {max_results} resultados para '{query}' en {language}"


@tool(args_schema=DatabaseQueryInput)
def query_database(table: str, filters: dict, limit: int = 10) -> str:
    """
    Consulta la base de datos de producción con filtros seguros.
    NUNCA ejecuta SQL raw — usa el ORM interno.
    """
    # En producción: usar SQLAlchemy con queries parametrizadas
    filter_str = ", ".join(f"{k}={v}" for k, v in filters.items())
    return f"[DB] {limit} registros de '{table}' con filtros: {filter_str}"


# ─── Ejecución Paralela de Tools ──────────────────────────────────────────────

async def execute_tools_parallel(
    tool_calls: list[dict],
    tools_map: dict[str, BaseTool],
) -> list[dict]:
    """
    Ejecuta múltiples tool calls en paralelo con asyncio.
    Reduce latencia cuando las tools son independientes entre sí.
    """
    async def run_tool(tc: dict) -> dict:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id = tc["id"]

        if tool_name not in tools_map:
            return {"id": tool_id, "output": f"Error: tool '{tool_name}' no encontrada.", "success": False}

        try:
            # Si la tool es async, await; si no, usar run_in_executor
            t = tools_map[tool_name]
            if asyncio.iscoroutinefunction(t.func):
                output = await t.ainvoke(tool_args)
            else:
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(None, lambda: t.invoke(tool_args))
            return {"id": tool_id, "output": str(output), "success": True}
        except Exception as e:
            return {"id": tool_id, "output": f"Error: {e}", "success": False}

    return await asyncio.gather(*[run_tool(tc) for tc in tool_calls])


# ─── Agente como Tool (Subagente) ─────────────────────────────────────────────

def create_specialist_tool(name: str, description: str, system_prompt: str) -> BaseTool:
    """
    Crea una herramienta que internamente usa un agente especializado.
    Permite anidar agentes como tools de un agente supervisor.
    """
    specialist_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def run_specialist(task: str) -> str:
        """Ejecuta el agente especialista con la tarea dada."""
        from langchain_core.messages import HumanMessage, SystemMessage
        response = specialist_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=task),
        ])
        return response.content

    return StructuredTool(
        name=name,
        description=description,
        func=run_specialist,
        args_schema=type("Input", (BaseModel,), {
            "task": (str, Field(description="La tarea específica para el especialista.")),
        }),
    )


# ─── Ejemplo de Uso ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Tool especialista como herramienta
    code_reviewer = create_specialist_tool(
        name="code_reviewer",
        description="Revisa código Python y detecta bugs, mejoras y violaciones de SOLID.",
        system_prompt="Eres un senior Python engineer especialista en code review. Sé específico y constructivo.",
    )

    tools = [search_knowledge_base, query_database, code_reviewer]
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = create_react_agent(llm, tools)

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Busca documentación sobre FastAPI y revisa este código: def get_user(id): return db.query(id)"}]
    })
    print(result["messages"][-1].content)
```

---

# Patrón 05: Self-RAG

## Concepto

El agente **decide dinámicamente** si necesita recuperar contexto externo para responder, en lugar de siempre buscar o nunca buscar.

```
Query → Relevance Check → [Needs retrieval? Yes → Retrieve → Grade → Answer] | [No → Direct Answer]
                                                      ↓ (poor docs)
                                               Rewrite Query → Retrieve again
```

```python
# patterns/self_rag.py
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


class SelfRAGState(TypedDict):
    question: str
    rewritten_query: str | None
    documents: list[str]
    answer: str
    needs_retrieval: bool
    docs_relevant: bool
    answer_grounded: bool
    iterations: int


class RetrievalDecision(BaseModel):
    needs_retrieval: bool = Field(description="True si la pregunta requiere información externa.")
    reasoning: str = Field(description="Por qué se necesita o no retrieval.")


class DocumentGrade(BaseModel):
    relevant: bool = Field(description="True si los documentos son relevantes para la pregunta.")
    feedback: str = Field(description="Por qué son o no son relevantes.")


class AnswerGrade(BaseModel):
    grounded: bool = Field(description="True si la respuesta está soportada por los documentos.")
    hallucination_detected: bool = Field(description="True si hay información no respaldada.")


llm = ChatOpenAI(model="gpt-4o", temperature=0)


def decide_retrieval_node(state: SelfRAGState) -> dict:
    """Decide si se necesita recuperar contexto externo."""
    decision: RetrievalDecision = llm.with_structured_output(RetrievalDecision).invoke([
        SystemMessage(content=(
            "Determina si esta pregunta requiere buscar información externa actualizada "
            "o si puedes responderla con conocimiento general. "
            "Necesitas retrieval para: datos específicos, eventos recientes, "
            "información de base de conocimiento privada."
        )),
        HumanMessage(content=f"Pregunta: {state['question']}"),
    ])
    print(f"\n🤔 Decisión retrieval: {'Sí' if decision.needs_retrieval else 'No'} — {decision.reasoning[:60]}")
    return {"needs_retrieval": decision.needs_retrieval}


def retrieve_node(state: SelfRAGState) -> dict:
    """Recupera documentos relevantes."""
    query = state.get("rewritten_query") or state["question"]
    # En producción: usar tu vector store real (pgvector, Pinecone, etc.)
    docs = [
        f"[Doc 1] Información sobre '{query}': contexto relevante A.",
        f"[Doc 2] Datos adicionales sobre '{query}': contexto relevante B.",
    ]
    print(f"   📚 Recuperados {len(docs)} documentos para: '{query}'")
    return {"documents": docs}


def grade_documents_node(state: SelfRAGState) -> dict:
    """Evalúa si los documentos son relevantes para la pregunta."""
    docs_text = "\n".join(state["documents"])
    grade: DocumentGrade = llm.with_structured_output(DocumentGrade).invoke([
        SystemMessage(content="Evalúa si estos documentos son útiles para responder la pregunta."),
        HumanMessage(content=f"Pregunta: {state['question']}\n\nDocumentos:\n{docs_text}"),
    ])
    print(f"   {'✅' if grade.relevant else '⚠️'} Documentos relevantes: {grade.relevant}")
    return {"docs_relevant": grade.relevant}


def rewrite_query_node(state: SelfRAGState) -> dict:
    """Reescribe la query para mejorar el retrieval."""
    rewritten = llm.invoke([
        SystemMessage(content=(
            "Reescribe la siguiente pregunta para optimizar la búsqueda en una base de conocimiento. "
            "Usa términos técnicos precisos. Sé específico."
        )),
        HumanMessage(content=state["question"]),
    ]).content
    print(f"   ✏️  Query reescrita: {rewritten[:60]}...")
    return {"rewritten_query": rewritten, "iterations": state["iterations"] + 1}


def generate_answer_node(state: SelfRAGState) -> dict:
    """Genera la respuesta con o sin contexto."""
    if state["documents"]:
        context = "\n\n".join(state["documents"])
        prompt = f"Contexto:\n{context}\n\nPregunta: {state['question']}"
    else:
        prompt = state["question"]

    answer = llm.invoke([
        SystemMessage(content="Responde de forma precisa y concisa basándote en el contexto disponible."),
        HumanMessage(content=prompt),
    ]).content
    return {"answer": answer}


def grade_answer_node(state: SelfRAGState) -> dict:
    """Verifica que la respuesta esté fundamentada en los documentos."""
    if not state["documents"]:
        return {"answer_grounded": True}  # Sin docs, confiar en conocimiento del LLM

    grade: AnswerGrade = llm.with_structured_output(AnswerGrade).invoke([
        SystemMessage(content="Verifica si la respuesta está respaldada por los documentos."),
        HumanMessage(content=(
            f"Documentos: {' '.join(state['documents'][:2])}\n\n"
            f"Respuesta: {state['answer']}"
        )),
    ])
    print(f"   {'✅' if grade.grounded else '⚠️'} Respuesta fundamentada: {grade.grounded}")
    return {"answer_grounded": grade.grounded}


def route_after_retrieval_decision(state: SelfRAGState) -> Literal["retrieve", "generate"]:
    return "retrieve" if state["needs_retrieval"] else "generate"


def route_after_grading(state: SelfRAGState) -> Literal["rewrite", "generate"]:
    if state["docs_relevant"] or state["iterations"] >= 2:
        return "generate"
    return "rewrite"


def route_after_answer(state: SelfRAGState) -> Literal["end", "generate"]:
    if state["answer_grounded"] or not state["documents"]:
        return "end"
    return "generate"  # Regenerar si hay alucinaciones


def build_self_rag_graph():
    graph = StateGraph(SelfRAGState)
    graph.add_node("decide_retrieval", decide_retrieval_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_docs", grade_documents_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate", generate_answer_node)
    graph.add_node("grade_answer", grade_answer_node)

    graph.add_edge(START, "decide_retrieval")
    graph.add_conditional_edges("decide_retrieval", route_after_retrieval_decision)
    graph.add_edge("retrieve", "grade_docs")
    graph.add_conditional_edges("grade_docs", route_after_grading)
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "grade_answer")
    graph.add_conditional_edges("grade_answer", route_after_answer, {"end": END, "generate": "generate"})

    return graph.compile()


if __name__ == "__main__":
    app = build_self_rag_graph()
    result = app.invoke({
        "question": "¿Cuáles son las mejores prácticas de LangGraph en 2025?",
        "rewritten_query": None, "documents": [], "answer": "",
        "needs_retrieval": False, "docs_relevant": False,
        "answer_grounded": False, "iterations": 0,
    })
    print(f"\n💬 Respuesta: {result['answer']}")
```
