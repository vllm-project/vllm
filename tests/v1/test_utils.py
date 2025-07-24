# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re

import pytest
import requests
import torch

from tests.utils import RemoteOpenAIServer
from vllm.v1.worker.utils import bind_kv_cache


def test_bind_kv_cache():
    from vllm.attention import Attention

    ctx = {
        'layers.0.self_attn': Attention(32, 128, 0.1),
        'layers.1.self_attn': Attention(32, 128, 0.1),
        'layers.2.self_attn': Attention(32, 128, 0.1),
        'layers.3.self_attn': Attention(32, 128, 0.1),
    }
    kv_cache = {
        'layers.0.self_attn': torch.zeros((1, )),
        'layers.1.self_attn': torch.zeros((1, )),
        'layers.2.self_attn': torch.zeros((1, )),
        'layers.3.self_attn': torch.zeros((1, )),
    }
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)
    assert ctx['layers.0.self_attn'].kv_cache[0] is kv_cache[
        'layers.0.self_attn']
    assert ctx['layers.1.self_attn'].kv_cache[0] is kv_cache[
        'layers.1.self_attn']
    assert ctx['layers.2.self_attn'].kv_cache[0] is kv_cache[
        'layers.2.self_attn']
    assert ctx['layers.3.self_attn'].kv_cache[0] is kv_cache[
        'layers.3.self_attn']

    assert runner_kv_caches[0] is kv_cache['layers.0.self_attn']
    assert runner_kv_caches[1] is kv_cache['layers.1.self_attn']
    assert runner_kv_caches[2] is kv_cache['layers.2.self_attn']
    assert runner_kv_caches[3] is kv_cache['layers.3.self_attn']


def test_bind_kv_cache_non_attention():
    from vllm.attention import Attention

    # example from Jamba PP=2
    ctx = {
        'model.layers.20.attn': Attention(32, 128, 0.1),
        'model.layers.28.attn': Attention(32, 128, 0.1),
    }
    kv_cache = {
        'model.layers.20.attn': torch.zeros((1, )),
        'model.layers.28.attn': torch.zeros((1, )),
    }

    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx['model.layers.20.attn'].kv_cache[0] is kv_cache[
        'model.layers.20.attn']
    assert ctx['model.layers.28.attn'].kv_cache[0] is kv_cache[
        'model.layers.28.attn']

    assert runner_kv_caches[0] is kv_cache['model.layers.20.attn']
    assert runner_kv_caches[1] is kv_cache['model.layers.28.attn']


# Prometheus metrics utilities for testing


def get_prometheus_metrics(
        server: RemoteOpenAIServer) -> dict[str, dict[str, float]]:
    """Fetch and parse Prometheus metrics from the /metrics endpoint.
    
    Returns:
        Dict mapping metric names to their values grouped by labels.
        For example: {"vllm:request_success": {
            "engine=0": 5.0, "engine=1": 3.0}
        }
    """
    try:
        response = requests.get(server.url_for("metrics"), timeout=10)
        response.raise_for_status()

        metrics: dict[str, dict[str, float]] = {}

        # Regex patterns for Prometheus metrics
        metric_with_labels = re.compile(
            r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}\s+([\d\.\-\+e]+)$')
        metric_simple = re.compile(
            r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([\d\.\-\+e]+)$')

        for line in response.text.split('\n'):
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Try to match metric with labels first
            match = metric_with_labels.match(line)
            if match:
                metric_name, labels_part, value_str = match.groups()
                try:
                    value = float(value_str)
                    if metric_name not in metrics:
                        metrics[metric_name] = {}
                    metrics[metric_name][f'{{{labels_part}}}'] = value
                except ValueError:
                    continue
            else:
                # Try simple metric without labels
                match = metric_simple.match(line)
                if match:
                    metric_name, value_str = match.groups()
                    try:
                        value = float(value_str)
                        if metric_name not in metrics:
                            metrics[metric_name] = {}
                        metrics[metric_name][''] = value
                    except ValueError:
                        continue

        return metrics
    except Exception as e:
        pytest.fail(f"Failed to fetch Prometheus metrics: {e}")
        return {}


def get_engine_request_counts(
        metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    """Extract request counts per engine from Prometheus metrics.
    
    Returns:
        Dict mapping engine indices to request counts.
        For example: {"0": 15.0, "1": 12.0}
    """
    engine_counts = {}

    # Look for request success metrics with engine labels
    success_metrics = metrics.get("vllm:request_success_total", {})
    engine_pattern = re.compile(r'engine="([^"]*)"')

    for labels, count in success_metrics.items():
        # Extract engine ID from labels using regex
        match = engine_pattern.search(labels)
        if match:
            engine_id = match.group(1)
            if engine_id not in engine_counts:
                engine_counts[engine_id] = 0.0
            engine_counts[engine_id] += count

    return engine_counts


def check_request_balancing(server: RemoteOpenAIServer, dp_size: int):
    """Check request balancing via Prometheus metrics if dp_size > 1.
    
    Args:
        server: The RemoteOpenAIServer instance
        dp_size: Number of data parallel ranks
    """
    if dp_size <= 1:
        return

    # Get metrics after all requests are completed
    metrics = get_prometheus_metrics(server)
    engine_counts = get_engine_request_counts(metrics)

    # Check that multiple engines received requests
    engines_with_requests = [
        engine for engine, count in engine_counts.items() if count > 0
    ]
    assert len(engines_with_requests) == dp_size, (
        f"Expected requests to be distributed across multiple engines,"
        f" but only engine(s) {engines_with_requests} received "
        f"requests. Engine counts: {engine_counts}")

    # Verify that the load is reasonably balanced
    # (no engine should handle all requests)
    total_requests = sum(engine_counts.values())

    for count in engine_counts.values():
        assert count > total_requests // (dp_size + 1), (
            f"requests are imbalanced: {engine_counts}")
