"""
Repositorios de acceso a datos para Fénix RAG.

Patrón Repository + Unit of Work:
  - Los repositorios NO hacen commit/rollback
  - La sesión se inyecta desde fuera (transaction() o get_db())
  - Permite coordinar múltiples repos en una sola transacción

Uso típico en un nodo del agente:
    from src.persistence.database import transaction
    from src.persistence.repositories import session_repo, document_repo

    async with transaction() as db:
        session = await session_repo.create_session(db, user_identifier="u1")
        doc = await document_repo.create_document(db, session_id=session.id, ...)
        # commit automático al salir del bloque
"""

from src.persistence.repositories import session_repo, document_repo, query_repo

__all__ = ["session_repo", "document_repo", "query_repo"]
