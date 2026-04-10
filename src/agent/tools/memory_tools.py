"""
Memory tools — Herramientas para gestionar contexto conversacional del agente.

Usan un SessionMemoryStore singleton que persiste durante la vida del proceso.
Los nodos sync este store con el estado del grafo (que persiste via checkpointer).

Uso en el grafo:
    - El nodo llama a save_context/retrieve_context como tools del agente
    - El store mantiene la memoria accesible entre llamadas de tools
    - Al final del grafo, el nodo sync el store con el estado
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from src.config.logging import get_logger

log = get_logger(__name__)


# ─── Session Memory Store — Singleton ────────────────────────────────────────

class SessionMemoryStore:
    """
    Store de memoria por sesión.

    Mantiene un dict de clave-valor por session_id.
    Los datos persisten durante la vida del proceso.
    En producción, reemplazar con PostgreSQL-backed store.
    """

    def __init__(self) -> None:
        self._stores: dict[str, dict[str, str]] = {}

    def save(self, session_id: str, key: str, value: str) -> None:
        """Guarda un par clave-valor para una sesión."""
        if session_id not in self._stores:
            self._stores[session_id] = {}
        self._stores[session_id][key] = value

    def retrieve(self, session_id: str, key: str) -> str | None:
        """Recupera un valor por clave y sesión."""
        return self._stores.get(session_id, {}).get(key)

    def retrieve_all(self, session_id: str) -> dict[str, str]:
        """Recupera toda la memoria de una sesión."""
        return self._stores.get(session_id, {}).copy()

    def sync_from_state(self, session_id: str, state_memory: dict[str, str]) -> None:
        """Sync el store con el estado del grafo (al inicio del grafo)."""
        self._stores[session_id] = state_memory.copy()

    def sync_to_state(self, session_id: str) -> dict[str, str]:
        """Sync el store con el estado del grafo (al final del grafo)."""
        return self._stores.get(session_id, {}).copy()

    def clear(self, session_id: str) -> None:
        """Limpia la memoria de una sesión."""
        self._stores.pop(session_id, None)


# Singleton global
_memory_store = SessionMemoryStore()


def get_memory_store() -> SessionMemoryStore:
    """Retorna el singleton de memoria."""
    return _memory_store


def reset_memory_store() -> None:
    """Resetea el store (para tests)."""
    global _memory_store  # noqa: PLW0603
    _memory_store = SessionMemoryStore()


# ─── Tools ───────────────────────────────────────────────────────────────────

@tool
def save_context(
    key: str,
    value: str,
    session_id: str = "",
) -> dict[str, Any]:
    """
    Guarda un par clave-valor en el contexto de la sesión.

    Útil para almacenar hallazgos intermedios, preferencias del usuario,
    o resultados de análisis que se necesitan en iteraciones posteriores.

    Args:
        key: Identificador del contexto (ej: "tema_principal", "articulos_relevantes").
        value: Valor a guardar (string).
        session_id: ID de sesión para aislamiento.

    Returns:
        Dict confirmando el guardado.
    """
    store = get_memory_store()
    store.save(session_id, key, value)

    log.info(
        "context_saved",
        key=key,
        session=session_id,
        value_length=len(value),
    )

    return {
        "saved": True,
        "key": key,
        "session_id": session_id,
        "value_length": len(value),
    }


@tool
def retrieve_context(
    key: str,
    session_id: str = "",
) -> dict[str, Any]:
    """
    Recupera un valor guardado previamente en el contexto de la sesión.

    Args:
        key: Identificador del contexto a recuperar.
        session_id: ID de sesión para aislamiento.

    Returns:
        Dict con el valor recuperado o indicador de no encontrado.
    """
    store = get_memory_store()
    value = store.retrieve(session_id, key)

    if value is not None:
        log.debug(
            "context_retrieved",
            key=key,
            session=session_id,
            found=True,
        )
        return {
            "found": True,
            "key": key,
            "value": value,
            "session_id": session_id,
        }

    log.debug(
        "context_not_found",
        key=key,
        session=session_id,
    )
    return {
        "found": False,
        "key": key,
        "value": None,
        "session_id": session_id,
    }


@tool
def list_context_keys(
    session_id: str = "",
) -> dict[str, Any]:
    """
    Lista todas las claves guardadas en el contexto de la sesión.

    Args:
        session_id: ID de sesión para aislamiento.

    Returns:
        Dict con lista de claves guardadas.
    """
    store = get_memory_store()
    all_memory = store.retrieve_all(session_id)

    return {
        "keys": list(all_memory.keys()),
        "count": len(all_memory),
        "session_id": session_id,
    }


@tool
def clear_context(
    session_id: str = "",
) -> dict[str, Any]:
    """
    Limpia todo el contexto de la sesión.

    Args:
        session_id: ID de sesión para aislamiento.

    Returns:
        Dict confirmando la limpieza.
    """
    store = get_memory_store()
    store.clear(session_id)

    log.info(
        "context_cleared",
        session=session_id,
    )

    return {
        "cleared": True,
        "session_id": session_id,
    }


__all__ = [
    "save_context",
    "retrieve_context",
    "list_context_keys",
    "clear_context",
    "get_memory_store",
    "reset_memory_store",
]
