"""
Repositorio de UserProfile — CRUD async para perfiles de usuario.

Sigue el mismo patrón que session_repo.py, document_repo.py:
  - Funciones puras que reciben AsyncSession como primer argumento
  - Usan `transaction()` o `read_session()` de database.py en el call site
  - Sin lógica de negocio — solo acceso a datos
  - flush (no commit) — el caller controla la transacción

Uso en nodos del grafo y endpoints FastAPI:
    from src.persistence.repositories.user_profile_repo import get_profile_for_user

    async with read_session() as db:
        profile = await get_profile_for_user(db, "user@empresa.com")
        preferred = profile.preferred_profile if profile else "general-dev"

    async with transaction() as db:
        await upsert_profile(db, "user@empresa.com", "ai-rag-engineer")
"""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.logging import get_logger
from src.persistence.models import UserProfile

log = get_logger(__name__)


# ─── Consultas ────────────────────────────────────────────────────────────────

async def get_profile_for_user(
    db: AsyncSession,
    user_identifier: str,
) -> Optional[UserProfile]:
    """
    Retorna el UserProfile activo de un usuario, o None si no existe.

    Args:
        db: AsyncSession inyectada.
        user_identifier: Email o JWT sub del usuario.

    Returns:
        UserProfile activo, o None si el usuario no tiene perfil configurado.
        En ese caso, el caller debe usar el default del SkillRegistry.
    """
    stmt = (
        select(UserProfile)
        .where(
            UserProfile.user_identifier == user_identifier,
            UserProfile.is_active.is_(True),
        )
        .limit(1)
    )
    result = await db.execute(stmt)
    profile = result.scalar_one_or_none()

    log.debug(
        "user_profile_fetched",
        user=user_identifier,
        found=profile is not None,
        preferred_profile=profile.preferred_profile if profile else None,
    )
    return profile


async def get_profile_by_id(
    db: AsyncSession,
    profile_id: uuid.UUID,
) -> Optional[UserProfile]:
    """Retorna un UserProfile por su UUID primario."""
    result = await db.execute(
        select(UserProfile).where(UserProfile.id == profile_id)
    )
    return result.scalar_one_or_none()


async def list_all_profiles(
    db: AsyncSession,
    only_active: bool = True,
) -> list[UserProfile]:
    """
    Lista todos los perfiles de usuario.

    Args:
        db: AsyncSession inyectada.
        only_active: Si True, solo retorna perfiles con is_active=True.

    Returns:
        Lista de UserProfile ordenada por user_identifier.
    """
    stmt = select(UserProfile).order_by(UserProfile.user_identifier)
    if only_active:
        stmt = stmt.where(UserProfile.is_active.is_(True))

    result = await db.execute(stmt)
    return list(result.scalars().all())


# ─── Escritura ────────────────────────────────────────────────────────────────

async def upsert_profile(
    db: AsyncSession,
    user_identifier: str,
    preferred_profile: str,
    custom_system_prompt: Optional[str] = None,
) -> UserProfile:
    """
    Crea o actualiza el UserProfile de un usuario.

    Si ya existe un perfil para ese user_identifier, actualiza
    preferred_profile y custom_system_prompt. Si no existe, crea uno nuevo.

    Args:
        db: AsyncSession inyectada (requiere transaction()).
        user_identifier: Email o JWT sub del usuario.
        preferred_profile: Nombre del pack a activar (debe existir en registry.json).
        custom_system_prompt: Override del system prompt. None = usar INDEX.md.

    Returns:
        UserProfile creado o actualizado.
    """
    existing = await get_profile_for_user(db, user_identifier)

    if existing is not None:
        existing.preferred_profile = preferred_profile
        existing.custom_system_prompt = custom_system_prompt
        existing.is_active = True
        await db.flush()

        log.info(
            "user_profile_updated",
            user=user_identifier,
            profile=preferred_profile,
        )
        return existing

    # Crear nuevo perfil
    new_profile = UserProfile(
        user_identifier=user_identifier,
        preferred_profile=preferred_profile,
        custom_system_prompt=custom_system_prompt,
        is_active=True,
    )
    db.add(new_profile)
    await db.flush()  # flush para obtener el id generado

    log.info(
        "user_profile_created",
        user=user_identifier,
        profile=preferred_profile,
        id=str(new_profile.id),
    )
    return new_profile


async def deactivate_profile(
    db: AsyncSession,
    user_identifier: str,
) -> bool:
    """
    Desactiva el perfil de un usuario (is_active=False).

    El usuario sigue existiendo pero el agente usa el default del registry.
    Usar en lugar de DELETE para preservar el historial de preferencias.

    Args:
        db: AsyncSession inyectada (requiere transaction()).
        user_identifier: Email o JWT sub del usuario.

    Returns:
        True si se desactivó, False si no existía.
    """
    profile = await get_profile_for_user(db, user_identifier)
    if profile is None:
        return False

    profile.is_active = False
    await db.flush()

    log.info("user_profile_deactivated", user=user_identifier)
    return True


async def get_preferred_profile_name(
    db: AsyncSession,
    user_identifier: str,
    default: str = "general-dev",
) -> str:
    """
    Retorna el nombre del perfil preferido del usuario, o el default.

    Convenience function para nodos del grafo — evita manejar None
    y la lógica de fallback en cada nodo.

    Args:
        db: AsyncSession inyectada.
        user_identifier: Email o JWT sub del usuario.
        default: Nombre a retornar si no hay perfil activo.

    Returns:
        Nombre del perfil activo, o `default` si no existe.
    """
    profile = await get_profile_for_user(db, user_identifier)
    if profile is None:
        return default
    return profile.preferred_profile
