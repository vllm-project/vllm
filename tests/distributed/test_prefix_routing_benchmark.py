# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.distributed.run_prefix_routing_benchmark import (
    BenefitMetric,
    _cache_delta,
    _parse_prometheus_counters,
    _validate_workload_equivalence,
    compare_benefit,
)

pytestmark = pytest.mark.skip_global_cleanup


def _result(value: float) -> dict:
    return {
        "metric": value,
        "total_input_tokens": 100,
        "total_output_tokens": 20,
        "completed": 10,
    }


def test_compare_benefit_reports_both_metric_directions():
    results = {
        "baseline": [_result(100.0), _result(102.0), _result(98.0)],
        "candidate": [_result(120.0), _result(118.0), _result(122.0)],
    }

    higher = compare_benefit(results, [BenefitMetric("metric", "higher")])
    lower = compare_benefit(results, [BenefitMetric("metric", "lower")])

    assert higher[0]["improvement_pct"] == pytest.approx(20.0)
    assert lower[0]["improvement_pct"] == pytest.approx(-20.0)


def test_compare_benefit_rejects_missing_metrics():
    results = {
        "baseline": [_result(100.0)] * 3,
        "candidate": [_result(90.0)] * 3,
    }

    with pytest.raises(RuntimeError, match="missing metric 'p95_ttft_ms'"):
        compare_benefit(results, [BenefitMetric("p95_ttft_ms", "lower")])


def test_parse_prometheus_counters_sums_engines():
    metrics = """
# HELP vllm:prefix_cache_queries Prefix cache queries
vllm:prefix_cache_queries_total{engine="0"} 100
vllm:prefix_cache_queries_total{engine="1"} 20
vllm:prefix_cache_hits_total{engine="0"} 70
vllm:prefix_cache_hits_total{engine="1"} 10
"""

    assert _parse_prometheus_counters(metrics) == {
        "prefix_cache_queries": 120.0,
        "prefix_cache_hits": 80.0,
    }


def test_cache_delta_aggregates_nodes():
    before = [
        {"prefix_cache_queries": 10.0, "prefix_cache_hits": 2.0},
        {"prefix_cache_queries": 5.0, "prefix_cache_hits": 1.0},
    ]
    after = [
        {"prefix_cache_queries": 30.0, "prefix_cache_hits": 12.0},
        {"prefix_cache_queries": 25.0, "prefix_cache_hits": 11.0},
    ]

    assert _cache_delta(before, after) == {
        "prefix_cache_queries": 40.0,
        "prefix_cache_hits": 20.0,
        "prefix_cache_hit_rate": 0.5,
    }


def test_validate_workload_equivalence_rejects_different_tokens():
    baseline = _result(1.0)
    candidate = _result(1.0)
    candidate["total_input_tokens"] = 101

    with pytest.raises(RuntimeError, match="workload differs"):
        _validate_workload_equivalence(
            {"baseline": [baseline], "candidate": [candidate]}
        )
