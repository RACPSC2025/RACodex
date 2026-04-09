# RAG 07 — Memoria & Persistencia: ChromaDB, LangGraph, Memory Systems

## Regla de Oro
> Un agente sin memoria es un agente que empieza de cero en cada consulta.
> Implementa al menos Short-term + Long-term desde el día uno.

---

## 1. Short-Term Memory (Conversacional)

```python
# memory/short_term.py
"""
Memoria de conversación en la sesión actual.
El historial es el contexto de la conversación con el usuario.
"""
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def build_rag_with_conversation_memory(retriever):
    """RAG con memoria conversacional completa."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    def rag_chain(inputs: dict) -> str:
        query = inputs["question"]
        history = inputs.get("history", [])

        # Reformular query considerando historial
        if history:
            context_query = llm.invoke([
                SystemMessage(content="Reformula la pregunta considerando el historial. Si es independiente, devuélvela igual."),
                *history[-4:],  # Últimos 2 turnos
                HumanMessage(content=query),
            ]).content
        else:
            context_query = query

        # Retrieve
        docs = retriever.invoke(context_query)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Generate
        return llm.invoke([
            SystemMessage(content=(
                "Eres un asistente legal experto. Responde basándote SOLO en el contexto proporcionado. "
                "Cita los artículos específicos. Si no está en el contexto, dilo explícitamente."
            )),
            *history[-6:],  # Historial reciente
            HumanMessage(content=f"Contexto legal:\n{context}\n\nPregunta: {query}"),
        ]).content

    return rag_chain, get_session_history
```

---

## 2. Long-Term Memory con ChromaDB

```python
# memory/long_term_chroma.py
"""
Memoria persistente entre sesiones.
Almacena hechos importantes, preferencias del usuario,
y conclusiones previas en ChromaDB.
"""
from datetime import datetime
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import uuid


class MemoryEntry(BaseModel):
    content: str = Field(description="El hecho o conclusión a recordar.")
    importance: float = Field(ge=0.0, le=1.0, description="Importancia (0-1).")
    memory_type: str = Field(description="fact|conclusion|preference|entity")
    entities: list[str] = Field(description="Entidades involucradas (personas, artículos, leyes).")


class LongTermMemory:
    """
    Memoria persistente en ChromaDB.
    Extrae y almacena automáticamente hechos importantes de la conversación.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "agent_memory",
    ) -> None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def remember(self, conversation_turn: str, session_id: str) -> list[str]:
        """
        Extrae y persiste memorias de un turno de conversación.
        Retorna lista de memorias guardadas.
        """
        extraction = self._llm.with_structured_output(
            type("MemoryList", (), {"__annotations__": {"memories": list[MemoryEntry]}}),
            method="function_calling",
        )
        # Versión simplificada con structured output directo
        memories_text = self._llm.invoke(
            f"""Extrae hechos importantes de esta conversación legal para recordar.
            Formato JSON: [{{"content": "...", "importance": 0.8, "type": "fact"}}]
            
            Conversación: {conversation_turn}
            
            Solo hechos relevantes de importancia > 0.5. JSON puro:"""
        ).content

        saved: list[str] = []
        try:
            import json
            raw = json.loads(memories_text)
            for item in raw if isinstance(raw, list) else []:
                if isinstance(item, dict) and item.get("importance", 0) > 0.5:
                    doc = Document(
                        page_content=item.get("content", ""),
                        metadata={
                            "session_id": session_id,
                            "memory_type": item.get("type", "fact"),
                            "importance": item.get("importance", 0.5),
                            "timestamp": datetime.now().isoformat(),
                            "memory_id": str(uuid.uuid4()),
                        },
                    )
                    self._store.add_documents([doc])
                    saved.append(doc.page_content)
        except Exception as e:
            print(f"   ⚠️  Memory extraction error: {e}")

        return saved

    def recall(self, query: str, k: int = 5) -> list[Document]:
        """Recupera memorias relevantes para la query actual."""
        return self._store.similarity_search(query, k=k)

    def recall_formatted(self, query: str, k: int = 5) -> str:
        """Retorna memorias como texto para incluir en el prompt."""
        memories = self.recall(query, k=k)
        if not memories:
            return ""
        return "=== Memorias previas relevantes ===\n" + "\n".join(
            f"- [{m.metadata.get('memory_type', 'fact')}] {m.page_content}"
            for m in memories
        )
```

---

## 3. LangGraph Checkpointer — Estado Persistente de Agentes

```python
# memory/langgraph_persistence.py
"""
LangGraph Checkpointer persiste el estado completo del grafo.
Permite: resumir conversaciones, múltiples threads, replay de estados.
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class PersistentRAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    documents: list[dict]
    session_metadata: dict


def build_persistent_rag_agent(retriever, memory_system: "LongTermMemory"):
    """
    Agente RAG con estado persistente entre sesiones.
    Cada thread_id es una conversación independiente que se puede resumir.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    checkpointer = MemorySaver()

    def rag_node(state: PersistentRAGState) -> dict:
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
        )
        if not last_human:
            return {}

        query = last_human.content

        # Recuperar memorias previas relevantes
        past_memories = memory_system.recall_formatted(query)

        # Retrieval
        docs = retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs)

        # Build prompt con memoria + contexto
        system = (
            "Eres un asistente legal experto. Responde SOLO basándote en el contexto. "
            "Cita artículos específicos. Indica si algo no está en el contexto.\n\n"
        )
        if past_memories:
            system += past_memories + "\n\n"

        response = llm.invoke([
            SystemMessage(content=system),
            *state["messages"][-8:],  # Últimos 4 turnos
            HumanMessage(content=f"Contexto recuperado:\n{context}\n\n{query}"),
        ])

        # Guardar memoria de este turno
        turn_text = f"User: {query}\nAssistant: {response.content}"
        thread_id = state["session_metadata"].get("thread_id", "default")
        memory_system.remember(turn_text, thread_id)

        return {
            "messages": [response],
            "documents": [{"content": d.page_content, **d.metadata} for d in docs],
        }

    graph = StateGraph(PersistentRAGState)
    graph.add_node("rag", rag_node)
    graph.add_edge(START, "rag")
    graph.add_edge("rag", END)

    return graph.compile(checkpointer=checkpointer)


# ─── Uso con múltiples threads ────────────────────────────────────────────────
# config_user_a = {"configurable": {"thread_id": "user-abc-session-1"}}
# config_user_b = {"configurable": {"thread_id": "user-xyz-session-1"}}
#
# # Cada thread mantiene su propia historia y no interfiere con otras
# agent.invoke({"messages": [HumanMessage("¿Qué dice el Art. 45?")], ...}, config=config_user_a)
# agent.invoke({"messages": [HumanMessage("¿Y el Art. 50?")], ...}, config=config_user_a)
# # El agente recuerda que en el turno anterior preguntaste por el Art. 45
```

---

## 4. Entity Memory para Documentos Legales

```python
# memory/entity_memory.py
"""
Entity Memory rastrea entidades específicas mencionadas en la conversación:
- Artículos citados
- Personas o partes involucradas
- Plazos y fechas mencionados
- Decisiones y conclusiones previas

Muy útil en asistentes legales donde el usuario pregunta sobre los mismos
artículos o partes a lo largo de la conversación.
"""
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from collections import defaultdict


class ExtractedEntities(BaseModel):
    articles_referenced: list[str] = Field(description="Números de artículos mencionados.")
    parties_mentioned: list[str] = Field(description="Personas, empresas o entidades mencionadas.")
    deadlines: list[str] = Field(description="Plazos, fechas o períodos mencionados.")
    legal_concepts: list[str] = Field(description="Conceptos legales clave mencionados.")


class EntityMemory:
    """Memoria de entidades legales a lo largo de la conversación."""

    def __init__(self) -> None:
        self._entities: dict[str, list[str]] = defaultdict(list)
        self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def update(self, text: str) -> ExtractedEntities:
        """Extrae y actualiza entidades del texto."""
        entities: ExtractedEntities = self._llm.with_structured_output(
            ExtractedEntities
        ).invoke(f"Extrae entidades legales de este texto:\n{text}")

        for art in entities.articles_referenced:
            if art not in self._entities["articles"]:
                self._entities["articles"].append(art)
        for party in entities.parties_mentioned:
            if party not in self._entities["parties"]:
                self._entities["parties"].append(party)

        return entities

    def get_context_summary(self) -> str:
        """Genera resumen del contexto de entidades para el prompt."""
        if not any(self._entities.values()):
            return ""
        parts = []
        if self._entities["articles"]:
            parts.append(f"Artículos discutidos: {', '.join(self._entities['articles'])}")
        if self._entities["parties"]:
            parts.append(f"Partes mencionadas: {', '.join(self._entities['parties'])}")
        return "Contexto de la sesión:\n" + "\n".join(parts)
```

---

## Setup de ChromaDB para Producción

```python
# memory/chroma_setup.py
"""
Configuración profesional de ChromaDB para RAG legal.
Múltiples colecciones: documentos, memoria, proposiciones.
"""
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def get_chroma_client(persist_dir: str = "./chroma_db") -> chromadb.PersistentClient:
    """Cliente ChromaDB con configuración de producción."""
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        ),
    )


def get_collection(
    name: str,
    persist_dir: str = "./chroma_db",
) -> Chroma:
    """Obtiene o crea una colección ChromaDB con embeddings de OpenAI."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return Chroma(
        client=get_chroma_client(persist_dir),
        collection_name=name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


# Colecciones recomendadas para RAG legal:
# - "legal_articles":     artículos indexados (jerarquía completa)
# - "legal_summaries":    resúmenes de documentos (índice jerárquico)
# - "agent_memory":       memoria long-term del agente
# - "user_sessions":      historial de sesiones por usuario
```
