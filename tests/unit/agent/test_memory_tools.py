"""Tests de memory_tools — SessionMemoryStore y tools."""

from __future__ import annotations

import pytest

from src.agent.tools.memory_tools import (
    SessionMemoryStore,
    get_memory_store,
    reset_memory_store,
    save_context,
    retrieve_context,
    list_context_keys,
    clear_context,
)


# ─── SessionMemoryStore ─────────────────────────────────────────────────────

class TestSessionMemoryStore:
    """Tests del store de memoria por sesión."""

    def setup_method(self):
        reset_memory_store()

    def test_save_and_retrieve(self):
        store = get_memory_store()
        store.save("session-1", "key1", "value1")

        result = store.retrieve("session-1", "key1")
        assert result == "value1"

    def test_retrieve_nonexistent_key(self):
        store = get_memory_store()
        result = store.retrieve("session-1", "nonexistent")
        assert result is None

    def test_retrieve_nonexistent_session(self):
        store = get_memory_store()
        result = store.retrieve("nonexistent-session", "key1")
        assert result is None

    def test_retrieve_all(self):
        store = get_memory_store()
        store.save("session-1", "key1", "value1")
        store.save("session-1", "key2", "value2")

        result = store.retrieve_all("session-1")
        assert result == {"key1": "value1", "key2": "value2"}

    def test_sessions_are_isolated(self):
        store = get_memory_store()
        store.save("session-1", "key1", "value1")
        store.save("session-2", "key1", "value2")

        assert store.retrieve("session-1", "key1") == "value1"
        assert store.retrieve("session-2", "key1") == "value2"

    def test_sync_from_state(self):
        store = get_memory_store()
        state_memory = {"key1": "value1", "key2": "value2"}

        store.sync_from_state("session-1", state_memory)
        assert store.retrieve_all("session-1") == state_memory

    def test_sync_to_state(self):
        store = get_memory_store()
        store.save("session-1", "key1", "value1")

        result = store.sync_to_state("session-1")
        assert result == {"key1": "value1"}

    def test_clear(self):
        store = get_memory_store()
        store.save("session-1", "key1", "value1")
        store.clear("session-1")

        assert store.retrieve_all("session-1") == {}


# ─── Tools ──────────────────────────────────────────────────────────────────

class TestMemoryTools:
    """Tests de las tools de memoria."""

    def setup_method(self):
        reset_memory_store()

    def test_save_context(self):
        result = save_context.invoke({
            "key": "test_key",
            "value": "test_value",
            "session_id": "session-1",
        })

        assert result["saved"] is True
        assert result["key"] == "test_key"
        assert result["value_length"] == 10

        # Verify it was actually saved
        store = get_memory_store()
        assert store.retrieve("session-1", "test_key") == "test_value"

    def test_retrieve_context_found(self):
        store = get_memory_store()
        store.save("session-1", "test_key", "test_value")

        result = retrieve_context.invoke({
            "key": "test_key",
            "session_id": "session-1",
        })

        assert result["found"] is True
        assert result["value"] == "test_value"

    def test_retrieve_context_not_found(self):
        result = retrieve_context.invoke({
            "key": "nonexistent",
            "session_id": "session-1",
        })

        assert result["found"] is False
        assert result["value"] is None

    def test_list_context_keys(self):
        store = get_memory_store()
        store.save("session-1", "key1", "value1")
        store.save("session-1", "key2", "value2")

        result = list_context_keys.invoke({
            "session_id": "session-1",
        })

        assert result["count"] == 2
        assert set(result["keys"]) == {"key1", "key2"}

    def test_clear_context(self):
        store = get_memory_store()
        store.save("session-1", "key1", "value1")

        result = clear_context.invoke({
            "session_id": "session-1",
        })

        assert result["cleared"] is True
        assert store.retrieve_all("session-1") == {}
