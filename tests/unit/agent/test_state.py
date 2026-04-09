"""
Tests del AgentState — validación de estado del grafo.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.agent.state import AgentState, initial_state


class TestAgentState:
    """Tests de inicialización y estructura del estado."""

    def test_initial_state_has_messages(self) -> None:
        state = initial_state(user_query="¿qué dice el artículo 5?")
        assert "messages" in state
        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], HumanMessage)

    def test_initial_state_has_user_query(self) -> None:
        state = initial_state(user_query="test query")
        assert state["user_query"] == "test query"
        assert state["active_query"] == "test query"

    def test_initial_state_has_session_id(self) -> None:
        state = initial_state(user_query="test", session_id="session-123")
        assert state["session_id"] == "session-123"

    def test_initial_state_has_empty_collections(self) -> None:
        state = initial_state(user_query="test")
        assert state["uploaded_files"] == []
        assert state["ingestion_plans"] == []
        assert state["ingested_documents"] == []
        assert state["retrieval_results"] == []
        assert state["sources"] == []

    def test_initial_state_has_defaults(self) -> None:
        state = initial_state(user_query="test")
        assert state["draft_answer"] == ""
        assert state["final_answer"] == ""
        assert state["reflection"] is None
        assert state["iteration_count"] == 0
        assert state["max_iterations"] == 2
        assert state["error"] is None
        assert state["route"] == ""
        assert state["retrieval_strategy"] == ""

    def test_initial_state_with_uploaded_files(self) -> None:
        state = initial_state(
            user_query="test",
            uploaded_files=["/path/to/file.pdf"],
        )
        assert state["uploaded_files"] == ["/path/to/file.pdf"]

    def test_initial_state_custom_max_iterations(self) -> None:
        state = initial_state(user_query="test", max_iterations=5)
        assert state["max_iterations"] == 5
