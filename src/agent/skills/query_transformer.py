"""
Query Transformation — Rewriting + Step-Back + Sub-Query Decomposition.

Pipeline de transformación de queries que mejora el recall del retrieval
aplicando tres técnicas complementarias:

1. **Query Rewriting:** Reformula la query para usar terminología técnica
   del dominio y ser más específica.
   "¿qué dice sobre despidos?" → "¿Cuáles son las disposiciones sobre
   terminación unilateral del contrato de trabajo?"

2. **Step-Back Prompting:** Genera una query más amplia para obtener
   contexto de fondo.
   "¿qué dice la sección 3.2?" → "¿Cuáles son los requisitos del
   sistema de gestión de calidad?"

3. **Sub-Query Decomposition:** Divide queries complejas en 2-4
   sub-queries atómicas que el retriever puede manejar individualmente.

Uso:
    from src.agent.skills.query_transformer import QueryTransformer

    transformer = QueryTransformer()

    # Rewriting solo
    rewritten = transformer.rewrite(query)

    # Step-back solo
    broader = transformer.step_back(query)

    # Descomposición
    sub_queries = transformer.decompose(query)

    # Pipeline completo (rewriting + step-back + original)
    variants = transformer.transform_all(query)
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config.logging import get_logger
from src.config.providers import get_llm

log = get_logger(__name__)


# ─── Prompts ──────────────────────────────────────────────────────────────────

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Eres un asistente experto en reformulación de consultas para sistemas RAG.\n"
        "Tu tarea es reescribir la consulta del usuario para que sea más específica, "
        "use terminología técnica del dominio y sea más probable que recupere "
        "información relevante.\n\n"
        "Reglas:\n"
        "- Mantén la intención original de la consulta\n"
        "- Usa terminología técnica formal del dominio\n"
        "- No agregues información que no se pueda inferir\n"
        "- Retorna SOLO la consulta reformulada, sin explicaciones"
    )),
    ("human", "Consulta original: {query}\n\nConsulta reformulada:"),
])

STEP_BACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Eres un asistente experto en generar consultas más amplias y generales "
        "para obtener contexto de fondo en sistemas RAG.\n\n"
        "Dada una consulta específica, genera una versión más general que capture "
        "el tema de fondo. Esto permite recuperar información contextual relevante.\n\n"
        "Reglas:\n"
        "- La consulta debe ser más general, no más específica\n"
        "- Captura el tema o concepto de fondo\n"
        "- Retorna SOLO la consulta general, sin explicaciones"
    )),
    ("human", "Consulta específica: {query}\n\nConsulta general:"),
])

SUB_QUERY_DECOMPOSITION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Eres un asistente experto en descomponer consultas complejas en "
        "sub-consultas simples para sistemas RAG.\n\n"
        "Dada una consulta compleja, divídela en 2 a 4 sub-consultas más simples "
        "que, respondidas individualmente, proporcionen una respuesta completa.\n\n"
        "Reglas:\n"
        "- Cada sub-consulta debe ser atómica (una sola idea)\n"
        "- Genera entre 2 y 4 sub-consultas\n"
        "- Usa un formato de lista, una sub-consulta por línea\n"
        "- No incluyas numeración ni prefijos como '1.' o '-'\n"
        "- No agregues explicaciones, solo las sub-consultas"
    )),
    ("human", "Consulta compleja: {query}\n\nSub-consultas:"),
])


# ─── QueryTransformer ────────────────────────────────────────────────────────

class QueryTransformer:
    """
    Transforma queries para mejorar el recall del retrieval.

    Usa un LLM cacheado (llm_temperature=0 para respuestas deterministas).
    """

    def __init__(self) -> None:
        self._rewrite_chain = QUERY_REWRITE_PROMPT | get_llm(temperature=0) | StrOutputParser()
        self._step_back_chain = STEP_BACK_PROMPT | get_llm(temperature=0) | StrOutputParser()
        self._decompose_chain = SUB_QUERY_DECOMPOSITION_PROMPT | get_llm(temperature=0.3) | StrOutputParser()

    def rewrite(self, query: str) -> str:
        """
        Reformula la query para ser más específica y técnica.

        Args:
            query: Consulta original del usuario.

        Returns:
            Query reformulada con terminología del dominio.
        """
        try:
            rewritten = self._rewrite_chain.invoke({"query": query})
            result = rewritten.strip()
            log.debug("query_rewritten", original=query[:60], rewritten=result[:60])
            return result
        except Exception as exc:
            log.warning("query_rewrite_failed", query=query[:60], error=str(exc))
            return query  # Fallback: retorna la query original

    def step_back(self, query: str) -> str:
        """
        Genera una query más amplia para obtener contexto de fondo.

        Args:
            query: Consulta específica del usuario.

        Returns:
            Query generalizada para recuperación de contexto.
        """
        try:
            broader = self._step_back_chain.invoke({"query": query})
            result = broader.strip()
            log.debug("query_step_back", specific=query[:60], broader=result[:60])
            return result
        except Exception as exc:
            log.warning("query_step_back_failed", query=query[:60], error=str(exc))
            return query

    def decompose(self, query: str, max_sub_queries: int = 4) -> list[str]:
        """
        Divide una query compleja en sub-queries atómicas.

        Args:
            query: Consulta compleja a descomponer.
            max_sub_queries: Máximo número de sub-queries (2-4).

        Returns:
            Lista de sub-queries simples. Si la query ya es simple,
            retorna [query].
        """
        try:
            response = self._decompose_chain.invoke({"query": query})
            sub_queries = [
                line.strip()
                for line in response.strip().split("\n")
                if line.strip() and len(line.strip()) > 5
            ]

            # Limpiar numeración o prefijos que el LLM pueda haber agregado
            cleaned = []
            for sq in sub_queries:
                # Remove leading numbers, bullets, dashes
                sq = sq.lstrip("0123456789.-*• ").strip()
                if sq:
                    cleaned.append(sq)

            result = cleaned[:max_sub_queries]

            # Si la descomposición produjo a 0-1 sub-queries, la query era simple
            if len(result) <= 1:
                return [query]

            log.debug(
                "query_decomposed",
                original=query[:60],
                sub_queries=len(result),
            )
            return result

        except Exception as exc:
            log.warning("query_decompose_failed", query=query[:60], error=str(exc))
            return [query]

    def transform_all(self, query: str) -> list[str]:
        """
        Pipeline completo: genera todas las variantes de la query.

        Combina:
        - Query original
        - Query reformulada (rewriting)
        - Query generalizada (step-back)

        Ideal para Multi-Query Fusion retrieval.

        Args:
            query: Consulta original del usuario.

        Returns:
            Lista de variantes de la query (siempre incluye la original).
        """
        variants = [query]

        # Rewriting
        rewritten = self.rewrite(query)
        if rewritten != query:
            variants.append(rewritten)

        # Step-back
        broader = self.step_back(query)
        if broader != query and broader not in variants:
            variants.append(broader)

        log.info(
            "query_transform_all_complete",
            original=query[:60],
            variants=len(variants),
        )

        return variants
