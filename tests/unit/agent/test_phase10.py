"""
Tests de Fase 10 — Personalidad del Agente + Skill Packs Dinámicos.

Cubre:
  10.3 UserProfile model + repository
  10.4 SkillRegistry integrado en generation_node
  10.5 Tool load_skill
  10.6 Tool search_skills + list_available_profiles
  Estado: active_profile en AgentState e initial_state

Estrategia de mocking:
  - SkillRegistry siempre mockeado — no depende del filesystem de agent_skills/
  - BD async mockeada con AsyncMock — no requiere PostgreSQL real
  - generation_node usa conftest mock_get_llm (heredado de conftest.py)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_core.messages import SystemMessage


# ═══════════════════════════════════════════════════════════════════════════════
# 10.3 — UserProfile Repository
# ═══════════════════════════════════════════════════════════════════════════════

class TestUserProfileRepository:
    """Tests del CRUD async de UserProfile."""

    @pytest.mark.asyncio
    async def test_get_profile_returns_none_when_not_exists(self):
        from src.persistence.repositories.user_profile_repo import get_profile_for_user

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        result = await get_profile_for_user(mock_db, "user@empresa.com")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_profile_returns_profile_when_exists(self):
        from src.persistence.repositories.user_profile_repo import get_profile_for_user
        from src.persistence.models import UserProfile

        mock_profile = MagicMock(spec=UserProfile)
        mock_profile.user_identifier = "user@empresa.com"
        mock_profile.preferred_profile = "ai-rag-engineer"
        mock_profile.is_active = True

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_profile
        mock_db.execute.return_value = mock_result

        result = await get_profile_for_user(mock_db, "user@empresa.com")
        assert result is not None
        assert result.preferred_profile == "ai-rag-engineer"

    @pytest.mark.asyncio
    async def test_get_preferred_profile_name_returns_default_when_no_profile(self):
        from src.persistence.repositories.user_profile_repo import get_preferred_profile_name

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        result = await get_preferred_profile_name(
            mock_db, "new_user@empresa.com", default="general-dev"
        )
        assert result == "general-dev"

    @pytest.mark.asyncio
    async def test_get_preferred_profile_name_returns_profile_when_exists(self):
        from src.persistence.repositories.user_profile_repo import get_preferred_profile_name
        from src.persistence.models import UserProfile

        mock_profile = MagicMock(spec=UserProfile)
        mock_profile.preferred_profile = "django-ai-engineer"
        mock_profile.is_active = True

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_profile
        mock_db.execute.return_value = mock_result

        result = await get_preferred_profile_name(
            mock_db, "user@empresa.com", default="general-dev"
        )
        assert result == "django-ai-engineer"

    @pytest.mark.asyncio
    async def test_upsert_creates_new_profile_when_not_exists(self):
        from src.persistence.repositories.user_profile_repo import upsert_profile

        mock_db = AsyncMock()

        # get_profile_for_user retorna None (no existe)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result
        mock_db.flush = AsyncMock()

        result = await upsert_profile(mock_db, "new@empresa.com", "ai-rag-engineer")

        mock_db.add.assert_called_once()
        mock_db.flush.assert_called_once()
        added_obj = mock_db.add.call_args[0][0]
        assert added_obj.user_identifier == "new@empresa.com"
        assert added_obj.preferred_profile == "ai-rag-engineer"

    @pytest.mark.asyncio
    async def test_upsert_updates_existing_profile(self):
        from src.persistence.repositories.user_profile_repo import upsert_profile
        from src.persistence.models import UserProfile

        existing = MagicMock(spec=UserProfile)
        existing.preferred_profile = "general-dev"
        existing.is_active = True

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        mock_db.execute.return_value = mock_result
        mock_db.flush = AsyncMock()

        result = await upsert_profile(mock_db, "user@empresa.com", "ai-rag-engineer")

        # El perfil existente debe haberse actualizado
        assert existing.preferred_profile == "ai-rag-engineer"
        assert existing.is_active is True
        mock_db.add.assert_not_called()  # No se agrega uno nuevo

    @pytest.mark.asyncio
    async def test_deactivate_profile_returns_false_when_not_exists(self):
        from src.persistence.repositories.user_profile_repo import deactivate_profile

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        result = await deactivate_profile(mock_db, "ghost@empresa.com")
        assert result is False

    @pytest.mark.asyncio
    async def test_deactivate_profile_sets_is_active_false(self):
        from src.persistence.repositories.user_profile_repo import deactivate_profile
        from src.persistence.models import UserProfile

        existing = MagicMock(spec=UserProfile)
        existing.is_active = True

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        mock_db.execute.return_value = mock_result
        mock_db.flush = AsyncMock()

        result = await deactivate_profile(mock_db, "user@empresa.com")

        assert result is True
        assert existing.is_active is False


# ═══════════════════════════════════════════════════════════════════════════════
# 10.4 — Skill Pack en generation_node
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerationNodeSkillPack:
    """Tests de inyección del skill pack en generation_node."""

    def _make_state(self, grade_score: float = 0.9, active_profile: str = "") -> dict:
        from langchain_core.documents import Document
        return {
            "user_query": "¿Cómo implemento CRAG?",
            "active_query": "¿Cómo implemento CRAG?",
            "retrieval_results": [
                Document(page_content="Contenido relevante sobre CRAG", metadata={"source": "doc.pdf"})
            ],
            "grade_score": grade_score,
            "active_profile": active_profile,
            "pipeline_metrics": {},
        }

    @patch("src.agent.skills.registry.get_skill_registry")
    @patch("src.agent.skills.rethinking.generate_direct")
    def test_skill_pack_injected_when_profile_active(
        self, mock_generate, mock_registry
    ):
        """Cuando hay pack, generate_direct recibe extra_system."""
        from src.agent.nodes.generation_node import generation_node

        mock_reg_instance = MagicMock()
        mock_reg_instance.default_profile = "general-dev"
        mock_reg_instance.load_pack.return_value = "# General Dev Pack\nContenido del pack"
        mock_registry.return_value = mock_reg_instance

        mock_generate.return_value = ("Respuesta generada", [{"source": "doc.pdf"}])

        state = self._make_state(grade_score=0.9, active_profile="general-dev")
        result = generation_node(state)

        # generate_direct debe recibir extra_system con el contenido del pack
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["extra_system"] is not None
        assert isinstance(call_kwargs["extra_system"], SystemMessage)
        assert "General Dev Pack" in call_kwargs["extra_system"].content

    @patch("src.agent.skills.registry.get_skill_registry")
    @patch("src.agent.skills.rethinking.generate_direct")
    def test_graceful_degradation_when_pack_empty(
        self, mock_generate, mock_registry
    ):
        """Si el pack está vacío, generate_direct recibe extra_system=None."""
        from src.agent.nodes.generation_node import generation_node

        mock_reg_instance = MagicMock()
        mock_reg_instance.default_profile = "general-dev"
        mock_reg_instance.load_pack.return_value = ""  # Pack vacío
        mock_registry.return_value = mock_reg_instance

        mock_generate.return_value = ("Respuesta sin pack", [])

        state = self._make_state(grade_score=0.9, active_profile="nonexistent-profile")
        result = generation_node(state)

        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["extra_system"] is None

    @patch("src.agent.skills.registry.get_skill_registry")
    @patch("src.agent.skills.rethinking.generate_direct")
    def test_active_profile_written_to_state(self, mock_generate, mock_registry):
        """El nombre del perfil resuelto debe escribirse al estado."""
        from src.agent.nodes.generation_node import generation_node

        mock_reg_instance = MagicMock()
        mock_reg_instance.default_profile = "general-dev"
        mock_reg_instance.load_pack.return_value = "# Pack content"
        mock_registry.return_value = mock_reg_instance

        mock_generate.return_value = ("Respuesta", [])

        state = self._make_state(grade_score=0.9, active_profile="ai-rag-engineer")
        result = generation_node(state)

        assert result["active_profile"] == "ai-rag-engineer"

    @patch("src.agent.skills.registry.get_skill_registry")
    @patch("src.agent.skills.rethinking.generate_direct")
    def test_uses_registry_default_when_active_profile_empty(
        self, mock_generate, mock_registry
    ):
        """Sin active_profile, usa el default_profile del registry."""
        from src.agent.nodes.generation_node import generation_node

        mock_reg_instance = MagicMock()
        mock_reg_instance.default_profile = "general-dev"
        mock_reg_instance.load_pack.return_value = "# General Dev content"
        mock_registry.return_value = mock_reg_instance

        mock_generate.return_value = ("Respuesta", [])

        state = self._make_state(grade_score=0.9, active_profile="")
        result = generation_node(state)

        mock_reg_instance.load_pack.assert_called_with("general-dev")

    @patch("src.agent.skills.registry.get_skill_registry")
    @patch("src.agent.skills.rethinking.generate_with_rethinking")
    def test_re2_selected_for_mid_grade_score(self, mock_rethinking, mock_registry):
        """grade_score 0.5-0.8 → generate_with_rethinking (Re2)."""
        from src.agent.nodes.generation_node import generation_node

        mock_reg_instance = MagicMock()
        mock_reg_instance.default_profile = "general-dev"
        mock_reg_instance.load_pack.return_value = "# Pack"
        mock_registry.return_value = mock_reg_instance

        mock_rethinking.return_value = ("Respuesta Re2", [])

        state = self._make_state(grade_score=0.65, active_profile="general-dev")
        result = generation_node(state)

        mock_rethinking.assert_called_once()
        assert result["generation_mode"] == "rethinking"

    @patch("src.agent.skills.registry.get_skill_registry")
    def test_no_docs_returns_early_without_llm_call(self, mock_registry):
        """Sin documentos, retorna inmediatamente sin llamar al LLM."""
        from src.agent.nodes.generation_node import generation_node

        mock_reg_instance = MagicMock()
        mock_reg_instance.default_profile = "general-dev"
        mock_reg_instance.load_pack.return_value = "# Pack"
        mock_registry.return_value = mock_reg_instance

        state = {
            "user_query": "test",
            "active_query": "test",
            "retrieval_results": [],
            "grade_score": 0.0,
            "active_profile": "",
            "pipeline_metrics": {},
        }
        result = generation_node(state)

        assert result["generation_mode"] == "no_docs"
        assert result["draft_answer"] != ""
        assert result["sources"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# 10.5 — Tool load_skill
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadSkillTool:

    @patch("src.agent.skills.registry.get_skill_registry")
    def test_load_skill_found(self, mock_registry):
        from src.agent.tools.skill_tools import load_skill

        mock_reg = MagicMock()
        mock_reg.load_skill.return_value = "# Contenido del skill\nPatrones de CRAG"
        mock_registry.return_value = mock_reg

        result = load_skill.invoke({
            "profile": "ai-rag-engineer",
            "skill_file": "crag.md",
        })

        assert result["found"] is True
        assert result["profile"] == "ai-rag-engineer"
        assert result["skill_file"] == "crag.md"
        assert "Patrones de CRAG" in result["content"]
        assert result["size_chars"] > 0

    @patch("src.agent.skills.registry.get_skill_registry")
    def test_load_skill_not_found(self, mock_registry):
        from src.agent.tools.skill_tools import load_skill

        mock_reg = MagicMock()
        mock_reg.load_skill.return_value = ""  # archivo no existe
        mock_registry.return_value = mock_reg

        result = load_skill.invoke({
            "profile": "ai-rag-engineer",
            "skill_file": "nonexistent.md",
        })

        assert result["found"] is False
        assert result["content"] == ""
        assert "hint" in result

    @patch("src.agent.skills.registry.get_skill_registry")
    def test_load_skill_graceful_on_exception(self, mock_registry):
        from src.agent.tools.skill_tools import load_skill

        mock_registry.side_effect = Exception("Registry error")

        result = load_skill.invoke({
            "profile": "ai-rag-engineer",
            "skill_file": "crag.md",
        })

        assert result["found"] is False
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 10.6 — Tool search_skills + list_available_profiles
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchSkillsTool:

    @patch("src.agent.skills.registry.get_skill_registry")
    def test_search_skills_returns_matches(self, mock_registry):
        from src.agent.tools.skill_tools import search_skills

        mock_reg = MagicMock()
        mock_reg.search_skills.return_value = [
            "Rag_Mastery/stage7-memory.md",
            "crag-patterns.md",
        ]
        mock_registry.return_value = mock_reg

        result = search_skills.invoke({
            "profile": "ai-rag-engineer",
            "query": "CRAG",
        })

        assert result["found"] is True
        assert result["total"] == 2
        assert "Rag_Mastery/stage7-memory.md" in result["matches"]
        assert "hint" in result

    @patch("src.agent.skills.registry.get_skill_registry")
    def test_search_skills_no_matches(self, mock_registry):
        from src.agent.tools.skill_tools import search_skills

        mock_reg = MagicMock()
        mock_reg.search_skills.return_value = []
        mock_registry.return_value = mock_reg

        result = search_skills.invoke({
            "profile": "general-dev",
            "query": "nonexistent_pattern_xyz",
        })

        assert result["found"] is False
        assert result["total"] == 0
        assert result["matches"] == []
        assert "hint" in result

    @patch("src.agent.skills.registry.get_skill_registry")
    def test_search_skills_graceful_on_exception(self, mock_registry):
        from src.agent.tools.skill_tools import search_skills

        mock_registry.side_effect = Exception("IO error")

        result = search_skills.invoke({
            "profile": "ai-rag-engineer",
            "query": "CRAG",
        })

        assert result["found"] is False
        assert "error" in result


class TestListAvailableProfilesTool:

    @patch("src.agent.skills.registry.get_skill_registry")
    def test_lists_all_profiles(self, mock_registry):
        from src.agent.tools.skill_tools import list_available_profiles

        mock_reg = MagicMock()
        mock_reg.available_profiles = [
            "general-dev", "ai-rag-engineer", "backend-python", "django-ai-engineer"
        ]
        mock_reg.default_profile = "general-dev"
        mock_registry.return_value = mock_reg

        result = list_available_profiles.invoke({})

        assert result["total"] == 4
        assert "ai-rag-engineer" in result["profiles"]
        assert result["default_profile"] == "general-dev"

    @patch("src.agent.skills.registry.get_skill_registry")
    def test_graceful_on_exception(self, mock_registry):
        from src.agent.tools.skill_tools import list_available_profiles

        mock_registry.side_effect = Exception("Registry unavailable")

        result = list_available_profiles.invoke({})

        assert result["total"] == 0
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Estado: active_profile en AgentState
# ═══════════════════════════════════════════════════════════════════════════════

class TestStateActiveProfile:

    def test_initial_state_has_active_profile(self):
        from src.agent.state import initial_state
        state = initial_state(user_query="test")
        assert "active_profile" in state
        assert state["active_profile"] == ""

    def test_initial_state_accepts_explicit_profile(self):
        from src.agent.state import initial_state
        state = initial_state(user_query="test", active_profile="ai-rag-engineer")
        assert state["active_profile"] == "ai-rag-engineer"

    def test_active_profile_independent_from_other_fields(self):
        from src.agent.state import initial_state
        state = initial_state(user_query="test", active_profile="general-dev")
        # Modificar active_profile no afecta otros campos de routing
        state["active_profile"] = "ai-rag-engineer"
        assert state["route"] == ""
        assert state["crag_route"] == ""
        assert state["reflection_route"] == ""
