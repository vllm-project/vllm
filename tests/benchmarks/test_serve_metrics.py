# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.benchmarks.serve import PrefixCacheMetrics, _parse_prefix_cache_metrics


def test_parse_prefix_cache_metrics() -> None:
    metrics = """
# HELP vllm:prefix_cache_queries_total Prefix cache queries
vllm:prefix_cache_queries_total{engine="0"} 100
vllm:prefix_cache_queries_total{engine="1"} 50
vllm:prefix_cache_hits_total{engine="0"} 40
vllm:prefix_cache_hits_total{engine="1"} 20
vllm:external_prefix_cache_hits_total{engine="0"} 1000
"""

    assert _parse_prefix_cache_metrics(metrics) == PrefixCacheMetrics(
        queries=150, hits=60
    )


def test_parse_prefix_cache_metrics_unavailable() -> None:
    assert _parse_prefix_cache_metrics("# no prefix cache metrics\n") is None
