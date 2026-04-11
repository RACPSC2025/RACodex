"""
Métricas del pipeline — Timing estructurado por nodo del grafo.

Cada nodo registra su duración, cantidad de documentos procesados,
y metadata adicional para observabilidad en producción.

Uso en nodos:
    from src.agent.metrics import node_timer

    def retrieval_node(state: AgentState) -> dict:
        with node_timer(state, "retrieval") as timer:
            # ... lógica del nodo ...
            timer.update(docs_count=len(results), extra={"strategy": "multi_query"})
            return {"retrieval_results": results, **timer.to_state()}
"""

from __future__ import annotations

import time
from typing import Any

from src.config.logging import get_logger

log = get_logger(__name__)


class NodeTimer:
    """
    Timer para un nodo individual del grafo.

    Registra start_ms, end_ms, duration_ms, docs_count, y metadata extra.
    Se integra con `pipeline_metrics` en el estado del agente.
    """

    def __init__(self, state: dict, node_name: str) -> None:
        self._state = state
        self._node_name = node_name
        self._start_ms = 0
        self._end_ms = 0
        self._docs_count = 0
        self._extra: dict[str, Any] = {}

    def __enter__(self) -> "NodeTimer":
        self._start_ms = _now_ms()
        return self

    def __exit__(self, *args: Any) -> None:
        self._end_ms = _now_ms()
        self._record()

    def update(self, docs_count: int = 0, extra: dict[str, Any] | None = None) -> None:
        """Actualiza conteo de documentos y metadata extra."""
        if docs_count:
            self._docs_count = docs_count
        if extra:
            self._extra.update(extra)

    def to_state(self) -> dict[str, Any]:
        """Retorna las métricas para incluir en el retorno del nodo."""
        return {
            "pipeline_metrics": {
                **self._state.get("pipeline_metrics", {}),
                self._node_name: {
                    "start_ms": self._start_ms,
                    "end_ms": self._end_ms or _now_ms(),
                    "duration_ms": (self._end_ms or _now_ms()) - self._start_ms,
                    "docs_count": self._docs_count,
                    "extra": self._extra,
                },
            }
        }

    def _record(self) -> None:
        """Registra las métricas en el estado y loggea."""
        metrics = self._state.get("pipeline_metrics", {})
        metrics[self._node_name] = {
            "start_ms": self._start_ms,
            "end_ms": self._end_ms,
            "duration_ms": self._end_ms - self._start_ms,
            "docs_count": self._docs_count,
            "extra": self._extra,
        }

        log.info(
            "node_metric",
            node=self._node_name,
            duration_ms=self._end_ms - self._start_ms,
            docs_count=self._docs_count,
            extra=self._extra if self._extra else None,
        )


def node_timer(state: dict, node_name: str) -> NodeTimer:
    """Factory para crear un NodeTimer."""
    return NodeTimer(state, node_name)


def _now_ms() -> int:
    """Retorna el timestamp actual en milisegundos."""
    return int(time.time() * 1000)


def format_pipeline_summary(metrics: dict[str, dict]) -> str:
    """
    Formatea un resumen legible de las métricas del pipeline.

    Útil para logging final y debugging.
    """
    if not metrics:
        return "No hay métricas registradas."

    lines = ["=== Pipeline Metrics ==="]
    total_ms = 0
    for node, data in metrics.items():
        duration = data.get("duration_ms", 0)
        total_ms += duration
        docs = data.get("docs_count", 0)
        extra = data.get("extra", {})
        extra_str = f" | {extra}" if extra else ""
        lines.append(f"  {node}: {duration}ms | docs: {docs}{extra_str}")

    lines.append(f"  TOTAL: {total_ms}ms")
    return "\n".join(lines)
