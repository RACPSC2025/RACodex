"""Tests del módulo de métricas del pipeline."""

from __future__ import annotations

import time

import pytest

from src.agent.metrics import (
    NodeTimer,
    node_timer,
    format_pipeline_summary,
)


# ─── NodeTimer ────────────────────────────────────────────────────────────────

class TestNodeTimer:
    """Tests del NodeTimer para métricas por nodo."""

    def test_timer_records_duration(self):
        state: dict = {"pipeline_metrics": {}}

        with node_timer(state, "test_node") as timer:
            time.sleep(0.01)  # 10ms

        metrics = timer.to_state()["pipeline_metrics"]["test_node"]
        assert metrics["duration_ms"] >= 10
        assert metrics["docs_count"] == 0

    def test_timer_updates_docs_count(self):
        state: dict = {"pipeline_metrics": {}}

        with node_timer(state, "test_node") as timer:
            timer.update(docs_count=5)

        metrics = timer.to_state()["pipeline_metrics"]["test_node"]
        assert metrics["docs_count"] == 5

    def test_timer_updates_extra_metadata(self):
        state: dict = {"pipeline_metrics": {}}

        with node_timer(state, "retrieval") as timer:
            timer.update(docs_count=10, extra={
                "variants": 3,
                "strategy": "multi_query",
            })

        metrics = timer.to_state()["pipeline_metrics"]["retrieval"]
        assert metrics["extra"]["variants"] == 3
        assert metrics["extra"]["strategy"] == "multi_query"
        assert metrics["docs_count"] == 10

    def test_timer_accumulates_metrics(self):
        state: dict = {"pipeline_metrics": {}}

        with node_timer(state, "node_a") as timer_a:
            timer_a.update(docs_count=3)

        with node_timer(state, "node_b") as timer_b:
            timer_b.update(docs_count=7)

        metrics = timer_b.to_state()["pipeline_metrics"]
        assert "node_a" in metrics
        assert "node_b" in metrics
        assert metrics["node_a"]["docs_count"] == 3
        assert metrics["node_b"]["docs_count"] == 7

    def test_timer_preserves_existing_metrics(self):
        state: dict = {
            "pipeline_metrics": {
                "previous_node": {"duration_ms": 100, "docs_count": 2},
            }
        }

        with node_timer(state, "current_node") as timer:
            timer.update(docs_count=5)

        metrics = timer.to_state()["pipeline_metrics"]
        assert "previous_node" in metrics
        assert "current_node" in metrics
        assert metrics["previous_node"]["docs_count"] == 2
        assert metrics["current_node"]["docs_count"] == 5

    def test_format_pipeline_summary(self):
        metrics = {
            "retrieval": {
                "duration_ms": 250,
                "docs_count": 15,
                "extra": {"strategy": "multi_query"},
            },
            "generation": {
                "duration_ms": 1200,
                "docs_count": 1,
                "extra": {"mode": "direct"},
            },
        }

        summary = format_pipeline_summary(metrics)
        assert "retrieval: 250ms" in summary
        assert "generation: 1200ms" in summary
        assert "TOTAL: 1450ms" in summary

    def test_format_empty_metrics(self):
        assert format_pipeline_summary({}) == "No hay métricas registradas."

    def test_format_no_metrics(self):
        assert format_pipeline_summary(None) == "No hay métricas registradas."  # type: ignore
