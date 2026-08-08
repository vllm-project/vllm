# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Snapshot tests locking the default Prometheus histogram buckets.

The expected lists are deliberately hard-coded literals rather than imports:
any change to a default bucket boundary must show up as a failing test here.
"""

import prometheus_client
import pytest

from vllm.config import ModelConfig, VllmConfig
from vllm.config.observability import ObservabilityConfig
from vllm.v1.metrics.buckets import (
    BUCKET_FAMILY_KEYS,
    build_1_2_5_buckets,
    histogram_buckets,
)
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.prometheus import unregister_vllm_metrics

pytestmark = pytest.mark.cpu_test

DEFAULT_BUCKET_SNAPSHOTS: dict[str, list[float]] = {
    "request_latency": [
        0.3,
        0.5,
        0.8,
        1.0,
        1.5,
        2.0,
        2.5,
        5.0,
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        120.0,
        240.0,
        480.0,
        960.0,
        1920.0,
        7680.0,
    ],
    "time_to_first_token": [
        0.001,
        0.005,
        0.01,
        0.02,
        0.04,
        0.06,
        0.08,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
        20.0,
        40.0,
        80.0,
        160.0,
        640.0,
        2560.0,
    ],
    "inter_token_latency": [
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.3,
        0.4,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
        20.0,
        40.0,
        80.0,
    ],
    "iteration_tokens": [
        1,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
    ],
    "request_params_n": [1, 2, 5, 10, 20],
    "kv_cache_residency": [
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
        1,
        2,
        5,
        10,
        20,
        30,
        60,
        120,
        300,
        600,
        1200,
        1800,
    ],
}

METRIC_FAMILIES: dict[str, str] = {
    "vllm:e2e_request_latency_seconds": "request_latency",
    "vllm:request_queue_time_seconds": "request_latency",
    "vllm:request_inference_time_seconds": "request_latency",
    "vllm:request_prefill_time_seconds": "request_latency",
    "vllm:request_decode_time_seconds": "request_latency",
    "vllm:time_to_first_token_seconds": "time_to_first_token",
    "vllm:inter_token_latency_seconds": "inter_token_latency",
    "vllm:request_time_per_output_token_seconds": "inter_token_latency",
    "vllm:iteration_tokens_total": "iteration_tokens",
    "vllm:request_params_n": "request_params_n",
    "vllm:request_prompt_tokens": "request_tokens",
    "vllm:request_generation_tokens": "request_tokens",
    "vllm:request_max_num_generation_tokens": "request_tokens",
    "vllm:request_params_max_tokens": "request_tokens",
    "vllm:request_prefill_kv_computed_tokens": "request_tokens",
    "vllm:kv_block_lifetime_seconds": "kv_cache_residency",
    "vllm:kv_block_idle_before_evict_seconds": "kv_cache_residency",
    "vllm:kv_block_reuse_gap_seconds": "kv_cache_residency",
}

TEST_MODEL = "distilbert/distilgpt2"
TEST_MODEL_MAX_LEN = 1024
REQUEST_TOKENS_SNAPSHOT_FOR_TEST_MODEL: list[float] = [
    1,
    2,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
]


@pytest.mark.parametrize("family", sorted(DEFAULT_BUCKET_SNAPSHOTS))
def test_default_bucket_snapshots(family):
    """Static families must expose exactly today's default boundaries."""
    assert histogram_buckets(family) == DEFAULT_BUCKET_SNAPSHOTS[family]


def test_request_tokens_bucket_snapshot():
    """The 1-2-5 token-count series must be stable for a given model len."""
    assert histogram_buckets("request_tokens", max_model_len=32768) == [
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        200,
        500,
        1000,
        2000,
        5000,
        10000,
        20000,
    ]
    assert build_1_2_5_buckets(100) == [1, 2, 5, 10, 20, 50, 100]


def test_request_tokens_requires_max_model_len():
    with pytest.raises(ValueError, match="max_model_len is required"):
        histogram_buckets("request_tokens")


def test_buckets_return_fresh_copies():
    """Mutating a returned list must not corrupt the shared defaults."""
    first = histogram_buckets("request_latency")
    first.append(1e9)
    second = histogram_buckets("request_latency")
    assert second == DEFAULT_BUCKET_SNAPSHOTS["request_latency"]
    assert second is not first


def test_bucket_family_keys():
    """The override vocabulary is exactly the seven known families."""
    assert {
        "request_latency",
        "time_to_first_token",
        "inter_token_latency",
        "iteration_tokens",
        "request_params_n",
        "request_tokens",
        "kv_cache_residency",
    } == BUCKET_FAMILY_KEYS


def collect_histogram_buckets() -> dict[str, list[float]]:
    """Read back each vllm:* histogram's finite bucket bounds by metric name."""
    found: dict[str, list[float]] = {}
    for metric in prometheus_client.REGISTRY.collect():
        if metric.name.startswith("vllm:") and metric.type == "histogram":
            found[metric.name] = sorted(
                {
                    float(sample.labels["le"])
                    for sample in metric.samples
                    if sample.name.endswith("_bucket") and sample.labels["le"] != "+Inf"
                }
            )
    return found


def build_logger_config(
    observability_config: ObservabilityConfig,
) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(model=TEST_MODEL),
        observability_config=observability_config,
    )


def test_prometheus_logger_default_buckets():
    """Every histogram the logger registers must use its family's defaults."""
    config = build_logger_config(ObservabilityConfig(kv_cache_metrics=True))
    assert config.model_config.max_model_len == TEST_MODEL_MAX_LEN
    try:
        PrometheusStatLogger(config)
        found = collect_histogram_buckets()
        assert set(found) == set(METRIC_FAMILIES)
        for metric_name, family in METRIC_FAMILIES.items():
            if family == "request_tokens":
                expected = REQUEST_TOKENS_SNAPSHOT_FOR_TEST_MODEL
            else:
                expected = DEFAULT_BUCKET_SNAPSHOTS[family]
            assert found[metric_name] == [float(b) for b in expected], metric_name
    finally:
        unregister_vllm_metrics()


def test_histogram_buckets_override_precedence():
    """A present family key wins verbatim; absent keys keep defaults."""
    overrides = {
        "request_latency": [0.5, 1.0],
        "request_tokens": [16.0, 4096.0],
    }
    got = histogram_buckets("request_latency", overrides=overrides)
    assert got == [0.5, 1.0]
    assert got is not overrides["request_latency"]
    # An override replaces the max_model_len-derived series entirely.
    assert histogram_buckets("request_tokens", overrides=overrides) == [
        16.0,
        4096.0,
    ]
    assert (
        histogram_buckets("time_to_first_token", overrides=overrides)
        == DEFAULT_BUCKET_SNAPSHOTS["time_to_first_token"]
    )


def test_prometheus_logger_applies_overrides():
    """Overridden families change every histogram in the family; others
    keep their default boundaries."""
    overridden: dict[str, list[float]] = {
        "request_latency": [0.5, 1.0, 2.0],
        "request_tokens": [16.0, 256.0, 4096.0],
        "kv_cache_residency": [0.1, 10.0],
    }
    config = build_logger_config(
        ObservabilityConfig(
            kv_cache_metrics=True,
            custom_histogram_buckets=overridden,
        )
    )
    try:
        PrometheusStatLogger(config)
        found = collect_histogram_buckets()
        assert set(found) == set(METRIC_FAMILIES)
        for metric_name, family in METRIC_FAMILIES.items():
            if family in overridden:
                expected = overridden[family]
            else:
                expected = [float(b) for b in DEFAULT_BUCKET_SNAPSHOTS[family]]
            assert found[metric_name] == expected, metric_name
    finally:
        unregister_vllm_metrics()
