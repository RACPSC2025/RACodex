"""
QueryPlannerSkill — descompone preguntas complejas en sub-queries.

Cuándo se usa:
  El agente invoca este skill cuando detecta que la query es compuesta
  (menciona múltiples artículos, hace comparaciones, o tiene más de 20 palabras).
  Para queries simples, el retrieval directo es más eficiente.

Patrón Plan-and-Execute:
  1. PlannerSkill produce un QueryPlan con sub-queries atómicas
  2. El nodo de retrieval ejecuta cada sub-query de forma independiente
  3. Los resultados se consolidan antes de la generación
  4. La respuesta integra todos los hallazgos

Ejemplo:
  Query: "¿Cuáles son las diferencias entre las obligaciones del Art. 2.2.4.6.8
          y el Art. 2.2.4.6.15 en materia de capacitación?"
  Plan:
    sub_queries: [
      "obligaciones capacitación artículo 2.2.4.6.8 Decreto 1072",
      "obligaciones capacitación artículo 2.2.4.6.15 Decreto 1072"
    ]
    strategy: "hybrid"
    complexity: "compound"
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.config.logging import get_logger
from src.config.providers import get_llm

log = get_logger(__name__)

# Número de artículos: detecta "2.2.4.6.1", "Art. 15", "artículo 22"
_ARTICLE_RE = re.compile(
    r"(?:art(?:ículo)?\.?\s*)?(\d+(?:\.\d+)+|\d+)",
    re.IGNORECASE,
)

# Palabras que indican comparación (→ query compuesta)
_COMPARISON_KEYWORDS = {
    "diferencia", "diferencias", "comparar", "comparación", "versus",
    "vs", "entre", "mientras", "a diferencia",
}

# Palabras que indican múltiples acciones (→ query compuesta)
_COMPOUND_KEYWORDS = {
    "y", "además", "también", "asimismo", "igualmente",
    "por otro lado", "adicionalmente",
}


@dataclass
class QueryPlan:
    """Plan de ejecución para una query del usuario."""
    original_query: str
    sub_queries: list[str]
    strategy: str               # "hybrid" | "hierarchical" | "full"
    expected_sources: list[str] # tipos de documentos esperados
    complexity: str             # "simple" | "compound" | "complex"
    article_numbers: list[str]  # artículos detectados en la query
    use_planner: bool = True    # False = query simple, retrieval directo


class QueryPlannerSkill:
    """
    Descompone preguntas complejas en sub-queries para retrieval paralelo.

    Flujo:
      1. Analizar la query con heurísticas (sin LLM, instantáneo)
      2. Si es simple → retornar QueryPlan con la query original
      3. Si es compleja → usar LLM para generar sub-queries optimizadas
    """

    def __init__(self, max_sub_queries: int = 3) -> None:
        self._max_sub_queries = max_sub_queries

    def plan(self, query: str) -> QueryPlan:
        """
        Produce un plan de ejecución para la query dada.

        Args:
            query: Pregunta del usuario.

        Returns:
            QueryPlan con sub-queries y estrategia recomendada.
        """
        log.debug("query_planning", query=query[:80])

        # Análisis heurístico primero (sin LLM)
        complexity = self._assess_complexity(query)
        articles = self._extract_articles(query)

        if complexity == "simple":
            plan = QueryPlan(
                original_query=query,
                sub_queries=[query],
                strategy="hybrid" if articles else "hybrid",
                expected_sources=["decreto", "resolución"],
                complexity="simple",
                article_numbers=articles,
                use_planner=False,
            )
            log.debug("query_plan_simple", articles=articles)
            return plan

        # Complejidad media/alta → LLM para sub-queries optimizadas
        try:
            plan = self._plan_with_llm(query, complexity, articles)
        except Exception as exc:
            log.warning("query_planner_llm_failed", error=str(exc), fallback="single_query")
            # Fallback: usar la query original como única sub-query
            plan = QueryPlan(
                original_query=query,
                sub_queries=[query],
                strategy="full",
                expected_sources=["decreto"],
                complexity=complexity,
                article_numbers=articles,
                use_planner=False,
            )

        log.info(
            "query_plan_ready",
            complexity=plan.complexity,
            sub_queries=len(plan.sub_queries),
            strategy=plan.strategy,
        )

        return plan

    # ── Análisis heurístico ───────────────────────────────────────────────────

    def _assess_complexity(self, query: str) -> str:
        """
        Clasifica la complejidad de la query sin LLM.

        simple:   Un artículo específico, pregunta directa (< 15 palabras)
        compound: Múltiples artículos, comparaciones, "y además" (15-30 palabras)
        complex:  Análisis profundo, múltiples temas, > 30 palabras
        """
        lower = query.lower()
        word_count = len(query.split())
        article_count = len(_ARTICLE_RE.findall(query))

        has_comparison = any(kw in lower for kw in _COMPARISON_KEYWORDS)
        has_compound = any(kw in lower for kw in _COMPOUND_KEYWORDS)

        if word_count <= 15 and article_count <= 1 and not has_comparison:
            return "simple"

        if has_comparison or article_count >= 2 or (has_compound and word_count > 20):
            return "compound"

        if word_count > 30:
            return "complex"

        return "simple"

    def _extract_articles(self, query: str) -> list[str]:
        """Extrae números de artículo de la query."""
        matches = _ARTICLE_RE.findall(query)
        # Filtrar números sueltos que no parezcan artículos legales
        articles = [m for m in matches if "." in m or len(m) >= 2]
        return list(dict.fromkeys(articles))[:5]  # deduplicar, máximo 5

    # ── Planificación con LLM ─────────────────────────────────────────────────

    def _plan_with_llm(
        self,
        query: str,
        complexity: str,
        articles: list[str],
    ) -> QueryPlan:
        from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415
        from src.agent.prompts.system import QUERY_PLANNER_PROMPT  # noqa: PLC0415

        llm = get_llm()
        chain = QUERY_PLANNER_PROMPT | llm | StrOutputParser()
        raw = chain.invoke({"query": query})

        parsed = self._parse_json(raw)
        sub_queries = parsed.get("sub_queries", [query])[:self._max_sub_queries]

        if not sub_queries:
            sub_queries = [query]

        return QueryPlan(
            original_query=query,
            sub_queries=sub_queries,
            strategy=parsed.get("strategy", "hybrid"),
            expected_sources=parsed.get("expected_sources", ["decreto"]),
            complexity=parsed.get("complexity", complexity),
            article_numbers=articles,
            use_planner=True,
        )

    def _parse_json(self, raw: str) -> dict:
        clean = re.sub(r"```json\s?|```", "", raw).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            return {}


def get_query_planner(**kwargs) -> QueryPlannerSkill:
    return QueryPlannerSkill(**kwargs)
