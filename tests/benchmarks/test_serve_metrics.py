# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.benchmarks.serve import (
    PrefixCacheMetrics,
    _calculate_prefix_cache_stats,
    _parse_prefix_cache_metrics,
)


def test_parse_prefix_cache_metrics() -> None:
    metrics = """
# HELP vllm:prefix_cache_queries_total Prefix cache queries
vllm:prefix_cache_queries_total{engine="0",model_name="my model"} 100
vllm:prefix_cache_queries_total{engine="1",model_name="my model"} 50
vllm:prefix_cache_hits_total{engine="0",model_name="my model"} 40
vllm:prefix_cache_hits_total{engine="1",model_name="my model"} 20
vllm:external_prefix_cache_hits_total{engine="0"} 1000
"""

    assert _parse_prefix_cache_metrics(metrics) == PrefixCacheMetrics(
        queries=150, hits=60
    )


def test_parse_prefix_cache_metrics_unavailable() -> None:
    assert _parse_prefix_cache_metrics("# no prefix cache metrics\n") is None


def test_calculate_prefix_cache_stats_uses_counter_deltas() -> None:
    before = PrefixCacheMetrics(queries=100, hits=80)
    after = PrefixCacheMetrics(queries=150, hits=100)

    assert _calculate_prefix_cache_stats(before, after) == {
        "prefix_cache_queries": 50,
        "prefix_cache_hits": 20,
        "prefix_cache_hit_rate": 40.0,
    }


def test_calculate_prefix_cache_stats_without_new_queries() -> None:
    snapshot = PrefixCacheMetrics(queries=100, hits=80)

    assert _calculate_prefix_cache_stats(snapshot, snapshot) is None
