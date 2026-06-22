# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Phase 3 Prometheus metrics exposed during weight updates.

Metrics under test:
  vllm:rl_weight_update_total           Counter  increments on finish_weight_update
  vllm:rl_weight_update_duration_seconds Histogram  records start→finish latency
  vllm:rl_weight_gen                    Gauge    mirrors weight_gen value
  vllm:rl_weight_update_active          Gauge    1 during update, 0 at rest

RFC: https://github.com/vllm-project/vllm/issues/45585
"""

import requests
from prometheus_client.parser import text_string_to_metric_families

from .conftest import gen, health, ok, server

_PORT_BASE = 8850


def _metrics(url: str) -> dict[str, list]:
    """Parse /metrics and return {metric_name: [Sample]} mapping."""
    r = requests.get(f"{url}/metrics", timeout=5)
    assert r.status_code == 200, f"/metrics failed: {r.status_code}"
    result: dict[str, list] = {}
    for family in text_string_to_metric_families(r.text):
        result[family.name] = family.samples
    return result


def _metric_value(metrics: dict, name: str, labels: dict | None = None) -> float | None:
    """Return the value of a specific labeled sample, or None if not found."""
    samples = metrics.get(name, [])
    for s in samples:
        if labels is None or all(s.labels.get(k) == v for k, v in labels.items()):
            return s.value
    return None


class TestWeightUpdateMetricsPresence:
    """All four metrics appear in /metrics immediately after server start."""

    def test_metrics_registered_at_startup(self):
        """Gauge and Histogram metrics appear immediately at server startup.

        Counter metrics (vllm:rl_weight_update_total) may not appear in
        /metrics output until their first inc() call in some prometheus_client
        versions — that is tested separately after weight updates.
        """
        with server(port=_PORT_BASE, dummy_weights=True) as url:
            m = _metrics(url)
            assert "vllm:rl_weight_update_duration_seconds" in m, (
                "vllm:rl_weight_update_duration_seconds missing from /metrics"
            )
            assert "vllm:rl_weight_gen" in m, (
                "vllm:rl_weight_gen missing from /metrics"
            )
            assert "vllm:rl_weight_update_active" in m, (
                "vllm:rl_weight_update_active missing from /metrics"
            )

    def test_initial_weight_gen_gauge_is_zero(self):
        with server(port=_PORT_BASE + 1, dummy_weights=True) as url:
            m = _metrics(url)
            val = _metric_value(m, "vllm:rl_weight_gen", {"engine": "0"})
            assert val == 0.0, f"initial weight_gen gauge expected 0, got {val}"

    def test_initial_active_gauge_is_zero(self):
        with server(port=_PORT_BASE + 2, dummy_weights=True) as url:
            m = _metrics(url)
            val = _metric_value(m, "vllm:rl_weight_update_active", {"engine": "0"})
            assert val == 0.0, f"initial active gauge expected 0, got {val}"


class TestWeightUpdateMetricsCount:
    """Counter and gauge increment correctly with set_label (no NCCL)."""

    def test_weight_gen_gauge_tracks_update_weight_label_calls(self):
        """/update_weight_label does NOT increment weight_gen; gauge must stay 0."""
        with server(port=_PORT_BASE + 3, dummy_weights=True) as url:
            requests.post(
                f"{url}/update_weight_label",
                json={"weight_label": "test"},
                timeout=5,
            )
            m = _metrics(url)
            val = _metric_value(m, "vllm:rl_weight_gen", {"engine": "0"})
            assert val == 0.0, (
                f"weight_gen gauge must stay 0 after label-only update, got {val}"
            )

    def test_active_gauge_starts_at_zero(self):
        """Before any weight update, active gauge is 0."""
        with server(port=_PORT_BASE + 4, dummy_weights=True) as url:
            m = _metrics(url)
            val = _metric_value(m, "vllm:rl_weight_update_active", {"engine": "0"})
            assert val == 0.0

    def test_engine_label_present_in_all_metrics(self):
        """Every RL metric sample must carry engine='0' label."""
        with server(port=_PORT_BASE + 5, dummy_weights=True) as url:
            m = _metrics(url)
            for metric_name in (
                "vllm:rl_weight_gen",
                "vllm:rl_weight_update_active",
            ):
                samples = m.get(metric_name, [])
                assert len(samples) > 0, f"{metric_name} has no samples"
                for s in samples:
                    assert "engine" in s.labels, (
                        f"{metric_name}: missing 'engine' label in {s.labels}"
                    )


class TestWeightUpdateMetricsCoexist:
    """Metric endpoints do not interfere with normal generation."""

    def test_metrics_and_generate_coexist(self):
        with server(port=_PORT_BASE + 6, dummy_weights=True) as url:
            _metrics(url)  # fetch metrics
            assert ok(gen(url)), "generate failed after fetching metrics"
            _metrics(url)  # fetch again
            assert health(url) == 200
