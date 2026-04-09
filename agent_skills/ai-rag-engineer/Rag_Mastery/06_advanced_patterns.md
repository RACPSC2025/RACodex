# RAG 06 — Patrones Avanzados: CRAG, Self-Retrieval, Rethinking, Reflexion, HITL

---

## 1. CRAG — Corrective RAG

```python
# advanced/crag.py
"""
CRAG evalúa la calidad de los documentos recuperados y decide:
- CORRECT: docs buenos → genera respuesta
- AMBIGUOUS: docs mediocres → búsqueda web adicional
- INCORRECT: docs malos → descarta y busca diferente

Ideal para: corpus legales incompletos o con versiones desactualizadas.
"""
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END


class CRAGState(TypedDict):
    query: str
    documents: list[Document]
    doc_quality: Literal["correct", "ambiguous", "incorrect"]
    web_results: list[str]
    final_answer: str
    rewritten_query: str


class DocumentQuality(BaseModel):
    quality: Literal["correct", "ambiguous", "incorrect"]
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    missing_information: str = Field(description="Qué falta en los documentos recuperados.")


llm = ChatOpenAI(model="gpt-4o", temperature=0)


def grade_documents_node(state: CRAGState) -> dict:
    """Evalúa la calidad de los documentos recuperados."""
    docs_text = "\n\n---\n\n".join(
        f"[Doc {i+1}]\n{doc.page_content[:400]}"
        for i, doc in enumerate(state["documents"])
    )
    quality: DocumentQuality = llm.with_structured_output(DocumentQuality).invoke(
        f"""Evalúa si estos documentos legales son suficientes para responder la consulta.

        Consulta: {state['query']}

        Documentos recuperados:
        {docs_text}

        Criterios:
        - correct (>0.7): documentos directamente relevantes y completos
        - ambiguous (0.3-0.7): parcialmente relevantes, información incompleta
        - incorrect (<0.3): no relevantes o contradictorios"""
    )
    print(f"\n   📊 CRAG Quality: {quality.quality} (score={quality.score:.2f})")
    return {"doc_quality": quality.quality}


def rewrite_for_search_node(state: CRAGState) -> dict:
    """Reescribe la query para búsqueda externa cuando docs son insuficientes."""
    rewritten = llm.invoke(
        f"""La búsqueda interna fue insuficiente. Reformula esta consulta legal
        para búsqueda en fuentes externas (BOE, jurisprudencia, doctrina).
        Sé específico con términos técnicos.
        
        Consulta original: {state['query']}
        
        Query para búsqueda externa:"""
    ).content
    return {"rewritten_query": rewritten}


def web_search_node(state: CRAGState) -> dict:
    """Búsqueda web de complemento (integrar con Tavily, SerpAPI, etc.)."""
    # En producción: usar TavilySearchAPIRetriever o similar
    query = state.get("rewritten_query") or state["query"]
    print(f"   🌐 Web search: {query[:60]}")
    mock_results = [f"[Web] Resultado externo relevante para: {query}"]
    return {"web_results": mock_results}


def generate_answer_node(state: CRAGState) -> dict:
    """Genera respuesta combinando docs internos y/o web."""
    # Combinar fuentes según calidad
    context_parts: list[str] = []

    if state["doc_quality"] != "incorrect":
        context_parts.append("=== Documentos internos ===\n" + "\n".join(
            doc.page_content for doc in state["documents"]
        ))

    if state["web_results"]:
        context_parts.append("=== Fuentes externas ===\n" + "\n".join(state["web_results"]))

    context = "\n\n".join(context_parts)
    answer = llm.invoke(
        f"""Responde esta consulta legal basándote en el contexto disponible.
        Cita los artículos o fuentes específicas que uses.
        Si hay incertidumbre, indícalo explícitamente.
        
        Consulta: {state['query']}
        
        Contexto:
        {context}"""
    ).content
    return {"final_answer": answer}


def route_after_grading(state: CRAGState) -> str:
    match state["doc_quality"]:
        case "correct":   return "generate"
        case "ambiguous": return "web_search"   # Complementar con web
        case "incorrect": return "rewrite"       # Buscar diferente


def build_crag_graph():
    graph = StateGraph(CRAGState)
    graph.add_node("grade_docs", grade_documents_node)
    graph.add_node("rewrite", rewrite_for_search_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_answer_node)

    graph.add_edge(START, "grade_docs")
    graph.add_conditional_edges(
        "grade_docs", route_after_grading,
        {"generate": "generate", "web_search": "web_search", "rewrite": "rewrite"},
    )
    graph.add_edge("rewrite", "web_search")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
```

---

## 2. Rethinking / Re-Reading (Re2)

```python
# advanced/rethinking.py
"""
Re-Reading hace que el LLM lea el contexto DOS veces:
1ra lectura: identifica información relevante
2da lectura: responde con foco en lo identificado

Mejora significativa en preguntas complejas multi-hop.
Muy útil para artículos legales con referencias cruzadas.
"""
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

llm = ChatOpenAI(model="gpt-4o", temperature=0)


def rethinking_generate(
    query: str,
    documents: list[Document],
) -> str:
    """
    Genera respuesta con Re-Reading del contexto legal.
    """
    context = "\n\n---\n\n".join(
        f"[Artículo {doc.metadata.get('article_number', i+1)}]\n{doc.page_content}"
        for i, doc in enumerate(documents)
    )

    # Primera lectura: extrae pasajes clave
    key_passages = llm.invoke(
        f"""Lee estos artículos legales cuidadosamente (primera lectura).
        Identifica los pasajes ESPECÍFICOS que son relevantes para: "{query}"
        Cita los números de artículo e incisos exactos.
        
        Documentos:
        {context}
        
        Pasajes clave identificados:"""
    ).content

    # Segunda lectura: responde con los pasajes identificados
    final_answer = llm.invoke(
        f"""Ahora responde la consulta basándote en los pasajes que identificaste.
        
        Consulta: {query}
        
        Pasajes clave (primera lectura):
        {key_passages}
        
        Contexto completo de referencia:
        {context[:2000]}
        
        Respuesta final (cita artículos específicos):"""
    ).content

    return final_answer
```

---

## 3. Reflexion en RAG

```python
# advanced/reflexion_rag.py
"""
Combina Reflexion con RAG:
1. Genera respuesta inicial
2. Evalúa si la respuesta está bien fundamentada
3. Si no, identifica qué información falta y hace retrieval adicional
4. Itera hasta calidad suficiente
"""
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END


class ReflexionRAGState(TypedDict):
    query: str
    documents: list[Document]
    answer: str
    reflection: str
    missing_info: list[str]
    iteration: int
    max_iterations: int
    approved: bool


class ReflexionOutput(BaseModel):
    approved: bool
    missing_information: list[str] = Field(
        description="Información que falta para una respuesta completa."
    )
    retrieval_queries: list[str] = Field(
        description="Queries adicionales de retrieval para obtener lo que falta."
    )
    improved_answer: str = Field(description="Respuesta mejorada si es posible.")


llm = ChatOpenAI(model="gpt-4o", temperature=0)


def generate_node(state: ReflexionRAGState) -> dict:
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    answer = llm.invoke(
        f"Consulta: {state['query']}\n\nContexto legal:\n{context}\n\nRespuesta:"
    ).content
    return {"answer": answer, "iteration": state["iteration"] + 1}


def reflect_node(state: ReflexionRAGState) -> dict:
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    reflexion: ReflexionOutput = llm.with_structured_output(ReflexionOutput).invoke(
        f"""Evalúa esta respuesta legal críticamente.
        
        Consulta: {state['query']}
        Respuesta actual: {state['answer']}
        Contexto disponible: {context[:1500]}
        
        ¿La respuesta está completamente fundamentada en los documentos?
        ¿Falta alguna referencia a artículos específicos?
        ¿Hay afirmaciones sin respaldo?"""
    )
    print(f"   🔍 Reflexion iter {state['iteration']}: approved={reflexion.approved}")
    return {
        "approved": reflexion.approved or state["iteration"] >= state["max_iterations"],
        "missing_info": reflexion.missing_information,
        "answer": reflexion.improved_answer or state["answer"],
    }


def retrieve_missing_node(state: ReflexionRAGState) -> dict:
    """Retrieval adicional para información faltante identificada en reflexión."""
    # En implementación real: usar el retriever con las queries adicionales
    additional_queries = state["missing_info"]
    print(f"   📚 Buscando información faltante: {additional_queries[:2]}")
    # ... retrieval adicional
    return {}


def build_reflexion_rag_graph():
    graph = StateGraph(ReflexionRAGState)
    graph.add_node("generate", generate_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("retrieve_missing", retrieve_missing_node)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "reflect")
    graph.add_conditional_edges(
        "reflect",
        lambda s: "end" if s["approved"] else "retrieve",
        {"end": END, "retrieve": "retrieve_missing"},
    )
    graph.add_edge("retrieve_missing", "generate")
    return graph.compile()
```

---

## 4. Human-on-the-Loop en RAG

```python
# advanced/human_on_the_loop.py
"""
Puntos de validación humana en el pipeline RAG:
1. Aprobar/rechazar el plan de retrieval
2. Validar chunks recuperados antes de generar
3. Aprobar la respuesta antes de enviarla

Fundamental para documentos legales donde los errores tienen consecuencias.
"""
from typing import TypedDict, Literal
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class HumanLoopRAGState(TypedDict):
    query: str
    retrieval_plan: str
    documents: list[Document]
    answer: str
    human_feedback: str
    approved: bool
    checkpoint: str


def plan_retrieval_node(state: HumanLoopRAGState) -> dict:
    """Genera plan de retrieval para revisión humana."""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    plan = llm.invoke(
        f"Describe el plan de retrieval para responder: {state['query']}"
    ).content
    return {"retrieval_plan": plan}


def human_approval_node(state: HumanLoopRAGState) -> dict:
    """
    Punto de interrupción para revisión humana.
    En LangGraph: usar interrupt() o interrupt_before en compile().
    """
    # Este nodo se pausa automáticamente con interrupt_before=["human_approval"]
    print(f"\n⏸️  PAUSA — Revisión humana requerida")
    print(f"Plan de retrieval:\n{state['retrieval_plan']}")
    print("Continúa con: app.invoke(None, config=thread_config)")
    return {}


def process_feedback_node(state: HumanLoopRAGState) -> dict:
    """Procesa el feedback humano y ajusta el plan."""
    feedback = state.get("human_feedback", "")
    approved = "aprobar" in feedback.lower() or "ok" in feedback.lower() or not feedback
    return {"approved": approved}


def build_human_loop_rag():
    """
    Construye el grafo con Human-in-the-Loop.
    Usa MemorySaver para persistir estado entre interrupciones.
    """
    memory = MemorySaver()
    graph = StateGraph(HumanLoopRAGState)

    graph.add_node("plan", plan_retrieval_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("process_feedback", process_feedback_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "human_approval")
    graph.add_edge("human_approval", "process_feedback")
    graph.add_conditional_edges(
        "process_feedback",
        lambda s: "end" if s["approved"] else "plan",
        {"end": END, "plan": "plan"},
    )

    # interrupt_before pausa ANTES de ejecutar ese nodo
    return graph.compile(
        checkpointer=memory,
        interrupt_before=["human_approval"],
    )


# Flujo de uso:
# thread_config = {"configurable": {"thread_id": "legal-session-001"}}
# result = app.invoke({"query": "¿Qué dice el Art. 45?"}, config=thread_config)
# → pausa automáticamente antes de human_approval
# → el humano revisa el plan
# result = app.invoke({"human_feedback": "aprobar"}, config=thread_config)
# → continúa desde donde se pausó
```

---

## 5. Self-Discover para RAG

```python
# advanced/self_discover.py
"""
Self-Discover analiza la estructura de la pregunta ANTES de recuperar.
Para documentos legales: identifica qué tipo de artículo se necesita
(obligación, prohibición, sanción, definición, procedimiento).
"""
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class LegalQueryStructure(BaseModel):
    query_type: str = Field(description="obligation|prohibition|sanction|definition|procedure|other")
    legal_entities: list[str] = Field(description="Personas, entidades o partes mencionadas.")
    temporal_context: str | None = Field(description="Plazos, fechas o períodos relevantes.")
    jurisdiction: str | None = Field(description="Jurisdicción o ámbito de aplicación.")
    key_concepts: list[str] = Field(description="Conceptos legales clave a buscar.")
    retrieval_strategy: str = Field(description="Estrategia de retrieval recomendada.")


def self_discover_query(query: str) -> LegalQueryStructure:
    """
    Analiza la estructura de una consulta legal antes de hacer retrieval.
    Permite personalizar la estrategia de búsqueda según el tipo de consulta.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm.with_structured_output(LegalQueryStructure).invoke(
        f"""Analiza esta consulta legal y descubre su estructura para optimizar la búsqueda.
        
        Consulta: {query}"""
    )
```
