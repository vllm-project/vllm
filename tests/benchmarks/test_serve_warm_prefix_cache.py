# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings

import pytest

from vllm.benchmarks.serve import (
    PrefixCacheCounters,
    _parse_prefix_cache_counters,
    _warn_if_prefix_cache_hits,
)


def test_parse_prefix_cache_counters_across_engines() -> None:
    metrics = """
# HELP vllm:prefix_cache_queries_total Prefix cache queries
vllm:prefix_cache_queries_total{engine="0"} 100
vllm:prefix_cache_queries_total{engine="1"} 50
vllm:prefix_cache_hits_total{engine="0"} 40
vllm:prefix_cache_hits_total{engine="1"} 20
vllm:external_prefix_cache_hits_total{engine="0"} 1000
"""

    assert _parse_prefix_cache_counters(metrics) == PrefixCacheCounters(
        queries=150, hits=60
    )


@pytest.mark.parametrize(
    "metrics",
    [
        "# no prefix cache metrics\n",
        "vllm:prefix_cache_queries_total 100\n",
        "vllm:prefix_cache_hits_total 50\n",
    ],
)
def test_parse_prefix_cache_counters_requires_both_metrics(metrics: str) -> None:
    assert _parse_prefix_cache_counters(metrics) is None


def test_warns_for_prefix_cache_hits() -> None:
    before = PrefixCacheCounters(queries=100, hits=10)
    after = PrefixCacheCounters(queries=200, hits=108)

    with pytest.warns(UserWarning, match="98.0% prefix cache hit rate"):
        _warn_if_prefix_cache_hits(before, after)


@pytest.mark.parametrize(
    "after",
    [
        PrefixCacheCounters(queries=1100, hits=19),
        PrefixCacheCounters(queries=50, hits=5),
        PrefixCacheCounters(queries=200, hits=210),
    ],
)
def test_does_not_warn_for_invalid_or_low_hit_delta(
    after: PrefixCacheCounters,
) -> None:
    before = PrefixCacheCounters(queries=100, hits=10)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_prefix_cache_hits(before, after)
