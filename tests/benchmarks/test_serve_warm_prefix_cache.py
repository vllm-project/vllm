# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings

import pytest

from vllm.benchmarks.serve import (
    PrefixCacheCounters,
    _parse_prefix_cache_counters,
    _warn_if_warm_prefix_cache,
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


def test_warns_for_warm_random_dataset_cache() -> None:
    before = PrefixCacheCounters(queries=100, hits=10)
    after = PrefixCacheCounters(queries=200, hits=108)

    with pytest.warns(UserWarning, match="98.0% prefix cache hit rate"):
        _warn_if_warm_prefix_cache("random", before, after)


@pytest.mark.parametrize(
    ("dataset_name", "after"),
    [
        ("random", PrefixCacheCounters(queries=1100, hits=19)),
        ("sharegpt", PrefixCacheCounters(queries=200, hits=108)),
        ("random", PrefixCacheCounters(queries=50, hits=5)),
    ],
)
def test_does_not_warn_without_warm_random_cache(
    dataset_name: str, after: PrefixCacheCounters
) -> None:
    before = PrefixCacheCounters(queries=100, hits=10)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_warm_prefix_cache(dataset_name, before, after)
