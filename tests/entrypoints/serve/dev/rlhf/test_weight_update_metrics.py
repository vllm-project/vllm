# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics tests for the RL weight-update lifecycle."""

import os
from unittest.mock import patch

import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families

from .conftest import gen, health, ok, resume, server


@pytest.fixture(scope="module", params=[False, True], ids=["MRV1", "MRV2"])
def use_v2(request):
    """Run the metrics tests with both model-runner implementations."""
    return request.param


@pytest.fixture(scope="module")
def server_url(use_v2):
    """Start one real server per model-runner implementation."""
    env_vars = {
        "VLLM_USE_V2_MODEL_RUNNER": "1" if use_v2 else "0",
    }

    # The shared fixture owns the server process and health polling.
    with (
        patch.dict(os.environ, env_vars),
        server(
            extra_args=[
                "--enable-prefix-caching",
                "--enable-prompt-tokens-details",
            ]
        ) as url,
    ):
        yield url


@pytest.fixture(autouse=True)
def restore_unpaused_state(server_url):
    """Ensure that each metrics test starts and ends in the resumed state."""
    assert resume(server_url) == 200
    yield
    assert resume(server_url) == 200


def _metrics(url: str) -> dict[str, list]:
    """Parse ``/metrics`` into a metric-family-to-samples mapping."""
    response = requests.get(f"{url}/metrics", timeout=5)
    assert response.status_code == 200, f"/metrics failed: {response.text}"

    result: dict[str, list] = {}
    for family in text_string_to_metric_families(response.text):
        result[family.name] = family.samples
    return result


def _metric_value(
    metrics: dict,
    name: str,
    labels: dict | None = None,
) -> float | None:
    """Return one sample value selected by metric name and labels."""
    samples = metrics.get(name, [])
    for sample in samples:
        if labels is None or all(
            sample.labels.get(key) == value for key, value in labels.items()
        ):
            return sample.value
    return None


class TestWeightUpdateMetricsPresence:
    """Check that the core RL metrics are registered after startup."""

    def test_metrics_registered_at_startup(self, server_url):
        metrics = _metrics(server_url)
        assert "vllm:rl_weight_update_duration_seconds" in metrics
        assert "vllm:rl_weight_gen" in metrics
        assert "vllm:rl_weight_update_active" in metrics

    def test_initial_weight_gen_gauge_is_zero(self, server_url):
        metrics = _metrics(server_url)
        value = _metric_value(
            metrics,
            "vllm:rl_weight_gen",
            {"engine": "0"},
        )
        assert value == 0.0

    def test_initial_active_gauge_is_zero(self, server_url):
        metrics = _metrics(server_url)
        value = _metric_value(
            metrics,
            "vllm:rl_weight_update_active",
            {"engine": "0"},
        )
        assert value == 0.0


class TestWeightUpdateMetricsCount:
    """Check label behavior and the engine label on RL metrics."""

    def test_weight_gen_gauge_tracks_update_weight_label_calls(self, server_url):
        """Updating only the label must not increment ``weight_gen``."""
        requests.post(
            f"{server_url}/update_weight_label",
            json={"weight_label": "test"},
            timeout=5,
        )
        metrics = _metrics(server_url)
        value = _metric_value(
            metrics,
            "vllm:rl_weight_gen",
            {"engine": "0"},
        )
        assert value == 0.0

    def test_active_gauge_starts_at_zero(self, server_url):
        metrics = _metrics(server_url)
        value = _metric_value(
            metrics,
            "vllm:rl_weight_update_active",
            {"engine": "0"},
        )
        assert value == 0.0

    def test_engine_label_present_in_all_metrics(self, server_url):
        metrics = _metrics(server_url)
        for metric_name in (
            "vllm:rl_weight_gen",
            "vllm:rl_weight_update_active",
        ):
            samples = metrics.get(metric_name, [])
            assert samples, f"{metric_name} has no samples"
            for sample in samples:
                assert "engine" in sample.labels


class TestWeightUpdateMetricsCoexist:
    """Ensure metrics collection does not interfere with generation."""

    def test_metrics_and_generate_coexist(self, server_url):
        _metrics(server_url)
        assert ok(gen(server_url)), "generation failed after fetching metrics"
        _metrics(server_url)
        assert health(server_url) == 200

