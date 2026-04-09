"""
SkillRegistry — Carga y gestiona los packs de skills del agente.

Permite cargar el INDEX.md del perfil activo del usuario,
buscar skills específicas dentro de un pack, y resolver conflictos
entre packs con el mismo archivo.

Uso:
    from src.agent.skills.registry import get_skill_registry

    registry = get_skill_registry()
    index_md = registry.load_pack("ai-rag-engineer")
    skill_content = registry.search_skills("ai-rag-engineer", "CRAG")
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config.logging import get_logger

log = get_logger(__name__)

# Ruta base: agent_skills/ está en la raíz del proyecto
AGENT_SKILLS_ROOT = Path(__file__).resolve().parent.parent.parent / "agent_skills"


class SkillRegistry:
    """
    Registro de perfiles de skills del agente.

    Cada perfil es un directorio en agent_skills/{profile}/ con un INDEX.md
    que actúa como punto de entrada. El resto de archivos .md del directorio
    se cargan on-demand vía search_skills() o load_skill().
    """

    def __init__(self, skills_root: Path | None = None) -> None:
        self._root = skills_root or AGENT_SKILLS_ROOT
        self._config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Carga registry.json."""
        config_path = self._root / "registry.json"
        if config_path.exists():
            return json.loads(config_path.read_text(encoding="utf-8"))
        log.warning("registry_json_not_found", path=str(config_path))
        return {"default_profile": "general-dev", "profiles": {}}

    @property
    def default_profile(self) -> str:
        return self._config.get("default_profile", "general-dev")

    @property
    def available_profiles(self) -> list[str]:
        return list(self._config.get("profiles", {}).keys())

    def get_profile_info(self, profile: str) -> dict[str, Any] | None:
        """Retorna metadata de un perfil."""
        return self._config.get("profiles", {}).get(profile)

    def load_pack(self, profile: str | None = None) -> str:
        """
        Carga el INDEX.md del perfil especificado.

        Si el perfil no existe o no se especifica, usa el default.

        Args:
            profile: Nombre del perfil (ej: "ai-rag-engineer").
                     None = usa el default_profile del registry.

        Returns:
            Contenido del INDEX.md como string. Empty string si no existe.
        """
        profile = profile or self.default_profile
        profile_info = self.get_profile_info(profile)

        if profile_info is None:
            # Fallback: intentar cargar directamente si el directorio existe
            pack_dir = self._root / profile
            if pack_dir.exists():
                index_file = pack_dir / "INDEX.md"
                if index_file.exists():
                    return index_file.read_text(encoding="utf-8")
            log.warning("profile_not_found", profile=profile, fallback=self.default_profile)
            profile = self.default_profile

        pack_dir = self._root / profile
        index_path = pack_dir / (profile_info.get("index_file", "INDEX.md"))

        if index_path.exists():
            content = index_path.read_text(encoding="utf-8")
            log.info("skill_pack_loaded", profile=profile, size=len(content))
            return content

        log.warning("skill_pack_index_missing", profile=profile, path=str(index_path))
        return ""

    def load_skill(self, profile: str, skill_file: str) -> str:
        """
        Carga un archivo .md específico dentro de un pack.

        Args:
            profile: Nombre del perfil.
            skill_file: Nombre del archivo (ej: "crag.md", "langgraph-fundamentals.md").

        Returns:
            Contenido del archivo. Empty string si no existe.
        """
        pack_dir = self._root / profile
        skill_path = pack_dir / skill_file

        # Also search in subdirectories
        if not skill_path.exists():
            for md_file in pack_dir.rglob(skill_file):
                skill_path = md_file
                break

        if skill_path.exists():
            content = skill_path.read_text(encoding="utf-8")
            log.debug("skill_loaded", profile=profile, file=skill_file, size=len(content))
            return content

        log.warning("skill_file_not_found", profile=profile, file=skill_file)
        return ""

    def search_skills(self, profile: str, query: str) -> list[str]:
        """
        Busca skills dentro de un pack por keyword en el nombre o contenido.

        Args:
            profile: Nombre del perfil.
            query: Término de búsqueda (case-insensitive, substring match).

        Returns:
            Lista de nombres de archivos .md que coinciden.
        """
        pack_dir = self._root / profile
        if not pack_dir.exists():
            return []

        query_lower = query.lower()
        results: list[str] = []

        for md_file in pack_dir.rglob("*.md"):
            # Match en nombre de archivo
            if query_lower in md_file.stem.lower():
                results.append(str(md_file.relative_to(pack_dir)))
                continue

            # Match en contenido (primeras 500 chars para performance)
            try:
                content = md_file.read_text(encoding="utf-8")[:500]
                if query_lower in content.lower():
                    results.append(str(md_file.relative_to(pack_dir)))
            except (OSError, UnicodeDecodeError):
                continue

        log.debug(
            "skill_search_complete",
            profile=profile,
            query=query,
            matches=len(results),
        )
        return results

    def list_all_skills(self, profile: str) -> list[str]:
        """Lista todos los archivos .md disponibles en un pack."""
        pack_dir = self._root / profile
        if not pack_dir.exists():
            return []

        return [
            str(f.relative_to(pack_dir))
            for f in pack_dir.rglob("*.md")
            if f.name != "INDEX.md"
        ]


# ─── Singleton ───────────────────────────────────────────────────────────────

_registry: SkillRegistry | None = None


@lru_cache(maxsize=1)
def get_skill_registry() -> SkillRegistry:
    """Retorna la instancia singleton del SkillRegistry."""
    global _registry  # noqa: PLW0603
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


def clear_registry_cache() -> None:
    """Limpia el cache del registry (para tests)."""
    global _registry  # noqa: PLW0603
    get_skill_registry.cache_clear()
    _registry = None
