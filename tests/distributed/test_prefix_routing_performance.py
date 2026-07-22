# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.distributed.run_prefix_routing_performance import (
    BenchmarkScenario,
    MetricRule,
    _validate_workload_equivalence,
    compare_performance,
)

pytestmark = pytest.mark.skip_global_cleanup


def _result(value: float) -> dict:
    return {
        "metric": value,
        "total_input_tokens": 100,
        "total_output_tokens": 20,
        "completed": 10,
    }


def test_compare_performance_accepts_metrics_within_thresholds():
    results = {
        "baseline": {
            "throughput": [_result(100.0), _result(101.0), _result(99.0)],
            "latency": [_result(10.0), _result(11.0), _result(9.0)],
        },
        "candidate": {
            "throughput": [_result(96.0), _result(97.0), _result(95.0)],
            "latency": [_result(10.5), _result(10.7), _result(10.3)],
        },
    }
    rules = [
        MetricRule("throughput", "metric", "higher", 5.0),
        MetricRule("latency", "metric", "lower", 10.0),
    ]

    passed, comparisons = compare_performance(results, rules)

    assert passed
    assert comparisons[0]["regression_pct"] == pytest.approx(4.0)
    assert comparisons[1]["regression_pct"] == pytest.approx(5.0)


def test_compare_performance_rejects_regressions_in_both_directions():
    results = {
        "baseline": {
            "throughput": [_result(100.0)] * 3,
            "latency": [_result(10.0)] * 3,
        },
        "candidate": {
            "throughput": [_result(90.0)] * 3,
            "latency": [_result(12.0)] * 3,
        },
    }
    rules = [
        MetricRule("throughput", "metric", "higher", 5.0),
        MetricRule("latency", "metric", "lower", 10.0),
    ]

    passed, comparisons = compare_performance(results, rules)

    assert not passed
    assert [item["regression_pct"] for item in comparisons] == pytest.approx(
        [10.0, 20.0]
    )


def test_compare_performance_rejects_missing_metrics():
    results = {
        "baseline": {"latency": [_result(10.0)] * 3},
        "candidate": {"latency": [_result(10.0)] * 3},
    }
    rules = [MetricRule("latency", "p95_ttft_ms", "lower", 10.0)]

    with pytest.raises(RuntimeError, match="missing metric 'p95_ttft_ms'"):
        compare_performance(results, rules)


def test_validate_workload_equivalence_rejects_different_token_counts():
    baseline = _result(1.0)
    candidate = _result(1.0)
    candidate["total_input_tokens"] = 101
    results = {
        "baseline": {"latency": [baseline]},
        "candidate": {"latency": [candidate]},
    }
    scenarios = [BenchmarkScenario("latency", 10, "1.0", 1, 0)]

    with pytest.raises(RuntimeError, match="workload differs"):
        _validate_workload_equivalence(results, scenarios)
