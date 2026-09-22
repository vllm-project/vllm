# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for priority-scheduling metric labels in PrometheusStatLogger."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from prometheus_client import REGISTRY

from vllm.v1.engine import FinishReason
from vllm.v1.metrics.loggers import PrometheusStatLogger, bucket_priority
from vllm.v1.metrics.stats import IterationStats, RequestStateStats


def _make_vllm_config(policy: str = "fcfs"):
    """Build a lightweight config stand-in for PrometheusStatLogger."""
    return SimpleNamespace(
        model_config=SimpleNamespace(
            served_model_name="test-model",
            max_model_len=4096,
            is_diffusion=False,
        ),
        scheduler_config=SimpleNamespace(policy=policy),
        cache_config=SimpleNamespace(num_gpu_blocks=0),
        compilation_config=SimpleNamespace(
            cudagraph_mode=False,
            cudagraph_capture_sizes=[],
        ),
        observability_config=SimpleNamespace(
            show_hidden_metrics=False,
            kv_cache_metrics=False,
            cudagraph_metrics=False,
            enable_mfu_metrics=False,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        lora_config=None,
        ec_transfer_config=None,
    )


def _make_finished_request_stats(priority: int = 0):
    """Build a FinishedRequestStats via IterationStats."""
    iteration_stats = IterationStats()
    # Override iteration_timestamp so _time_since is deterministic.
    iteration_stats.iteration_timestamp = 10.0
    req_stats = RequestStateStats(arrival_time=0.0)
    req_stats.scheduled_ts = 0.1
    req_stats.first_token_ts = 0.5
    req_stats.last_token_ts = 5.0
    req_stats.num_generation_tokens = 10
    req_stats.first_token_latency = 0.5
    iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="test-req",
        num_prompt_tokens=100,
        max_tokens_param=10,
        req_stats=req_stats,
        priority=priority,
    )
    return iteration_stats


def _collect_samples(metric_name: str):
    """Collect all samples for a given metric name from the global registry."""
    samples = []
    for collector in REGISTRY.collect():
        if collector.name == metric_name:
            for sample in collector.samples:
                samples.append(sample)
    return samples


def _cleanup_registry():
    """Unregister all vllm metrics from the global registry."""
    for collector in list(REGISTRY._collector_to_names):
        if hasattr(collector, "_name") and "vllm" in collector._name:
            REGISTRY.unregister(collector)


@pytest.fixture
def logger_fcfs():
    _cleanup_registry()
    with patch("vllm.v1.metrics.loggers.unregister_vllm_metrics", lambda: None):
        logger = PrometheusStatLogger(_make_vllm_config("fcfs"))
    yield logger
    _cleanup_registry()


@pytest.fixture
def logger_priority():
    _cleanup_registry()
    with patch("vllm.v1.metrics.loggers.unregister_vllm_metrics", lambda: None):
        logger = PrometheusStatLogger(_make_vllm_config("priority"))
    yield logger
    _cleanup_registry()


def test_fcfs_metrics_have_no_priority_label(logger_fcfs):
    """FCFS mode: metrics should NOT have a 'priority' label."""
    iteration_stats = _make_finished_request_stats(priority=3)
    logger_fcfs.record(
        scheduler_stats=None,
        iteration_stats=iteration_stats,
        engine_idx=0,
    )

    e2e_samples = _collect_samples("vllm:e2e_request_latency_seconds")
    assert len(e2e_samples) > 0
    for sample in e2e_samples:
        assert "priority" not in sample.labels


def test_priority_metrics_have_priority_label(logger_priority):
    """Priority mode: metrics should have a bucketed 'priority' label."""
    # priority=2 buckets into ">1" (less urgent than the conventional "low").
    iteration_stats = _make_finished_request_stats(priority=2)
    logger_priority.record(
        scheduler_stats=None,
        iteration_stats=iteration_stats,
        engine_idx=0,
    )

    e2e_samples = _collect_samples("vllm:e2e_request_latency_seconds")
    assert len(e2e_samples) > 0
    for sample in e2e_samples:
        assert "priority" in sample.labels
        assert sample.labels["priority"] == ">1"


def test_priority_metrics_distinct_per_priority(logger_priority):
    """Priority mode: requests with different priority buckets produce
    distinct label series."""
    # First request with priority 0 -> bucket "0"
    iter_stats_0 = _make_finished_request_stats(priority=0)
    logger_priority.record(
        scheduler_stats=None, iteration_stats=iter_stats_0, engine_idx=0
    )

    # Second request with priority 5 -> bucket ">1"
    iter_stats_5 = _make_finished_request_stats(priority=5)
    logger_priority.record(
        scheduler_stats=None, iteration_stats=iter_stats_5, engine_idx=0
    )

    e2e_samples = _collect_samples("vllm:e2e_request_latency_seconds")
    priority_values = {s.labels["priority"] for s in e2e_samples}
    assert "0" in priority_values
    assert ">1" in priority_values


def test_priority_request_success_has_priority_label(logger_priority):
    """Priority mode: vllm:request_success counter has priority label."""
    iteration_stats = _make_finished_request_stats(priority=1)
    logger_priority.record(
        scheduler_stats=None,
        iteration_stats=iteration_stats,
        engine_idx=0,
    )

    success_samples = _collect_samples("vllm:request_success")
    assert len(success_samples) > 0
    for sample in success_samples:
        assert "priority" in sample.labels
        assert sample.labels["priority"] == "1"


def test_priority_all_finished_metrics_have_label(logger_priority):
    """Priority mode: all finished-request metrics carry the priority label."""
    # priority=7 buckets into ">1".
    iteration_stats = _make_finished_request_stats(priority=7)
    logger_priority.record(
        scheduler_stats=None,
        iteration_stats=iteration_stats,
        engine_idx=0,
    )

    finished_metric_names = [
        "vllm:e2e_request_latency_seconds",
        "vllm:request_queue_time_seconds",
        "vllm:request_prefill_time_seconds",
        "vllm:request_inference_time_seconds",
        "vllm:request_decode_time_seconds",
        "vllm:request_prompt_tokens",
        "vllm:request_generation_tokens",
        "vllm:request_prefill_kv_computed_tokens",
        "vllm:request_time_per_output_token_seconds",
        "vllm:request_params_max_tokens",
        "vllm:request_num_preemptions",
        "vllm:request_success",
        "vllm:time_to_first_token_seconds",
    ]
    for name in finished_metric_names:
        samples = _collect_samples(name)
        assert len(samples) > 0, f"No samples for {name}"
        for sample in samples:
            assert "priority" in sample.labels, f"{name} missing priority label"
            assert sample.labels["priority"] == ">1", (
                f"{name} has wrong priority: {sample.labels['priority']}"
            )


@pytest.mark.parametrize(
    "priority, expected",
    [
        (-2, "<-1"),
        (-1, "-1"),
        (0, "0"),
        (1, "1"),
        (2, ">1"),
        (1000, ">1"),
        (-1000, "<-1"),
    ],
)
def test_bucket_priority(priority, expected):
    """The conventional -1/0/1 values are preserved; others are bucketed."""
    assert bucket_priority(priority) == expected


def test_priority_special_values_preserved(logger_priority):
    """Priority mode: -1/0/1 stay as distinct, exact buckets."""
    for priority in (-1, 0, 1):
        iter_stats = _make_finished_request_stats(priority=priority)
        logger_priority.record(
            scheduler_stats=None, iteration_stats=iter_stats, engine_idx=0
        )

    e2e_samples = _collect_samples("vllm:e2e_request_latency_seconds")
    priority_values = {s.labels["priority"] for s in e2e_samples}
    assert {"-1", "0", "1"} <= priority_values


def test_request_priority_histogram_observes_raw_value(logger_priority):
    """Priority mode: vllm:request_priority observes the raw priority number."""
    iteration_stats = _make_finished_request_stats(priority=5)
    logger_priority.record(
        scheduler_stats=None, iteration_stats=iteration_stats, engine_idx=0
    )

    samples = _collect_samples("vllm:request_priority")
    assert len(samples) > 0
    # Histogram exposes _count / _bucket samples but no priority label.
    for sample in samples:
        assert "priority" not in sample.labels
    count = next(s.value for s in samples if s.name.endswith("_count"))
    assert count == 1.0
    # Raw priority 5 falls into (1, 10]: cumulative le=1 bucket is empty,
    # le=10 bucket holds the observation.
    le_1 = next(
        s.value
        for s in samples
        if s.name.endswith("_bucket") and s.labels["le"] == "1.0"
    )
    le_10 = next(
        s.value
        for s in samples
        if s.name.endswith("_bucket") and s.labels["le"] == "10.0"
    )
    assert le_1 == 0.0
    assert le_10 == 1.0


def test_request_priority_histogram_absent_in_fcfs(logger_fcfs):
    """FCFS mode: vllm:request_priority is not registered."""
    iteration_stats = _make_finished_request_stats(priority=0)
    logger_fcfs.record(
        scheduler_stats=None, iteration_stats=iteration_stats, engine_idx=0
    )
    assert _collect_samples("vllm:request_priority") == []


def test_scheduler_policy_info(logger_priority):
    """Scheduler policy info gauge marks the active policy with value 1."""
    samples = _collect_samples("vllm:scheduler_policy_info")
    by_policy = {s.labels["policy"]: s.value for s in samples}
    assert "fcfs" in by_policy
    assert "priority" in by_policy
    assert by_policy["priority"] == 1.0
    assert by_policy["fcfs"] == 0.0


@pytest.fixture
def clean_registry():
    """Ensure a clean global registry before and after a test."""
    _cleanup_registry()
    yield
    _cleanup_registry()


def test_fcfs_metrics_have_correct_engine_label(logger_fcfs):
    """FCFS mode: finished-request metrics still carry the model/engine labels."""
    iteration_stats = _make_finished_request_stats(priority=0)
    logger_fcfs.record(
        scheduler_stats=None, iteration_stats=iteration_stats, engine_idx=0
    )

    samples = _collect_samples("vllm:e2e_request_latency_seconds")
    assert len(samples) > 0
    for sample in samples:
        assert sample.labels["engine"] == "0"
        assert sample.labels["model_name"] == "test-model"


def test_priority_multi_engine_distinct_series(clean_registry):
    """Priority mode + DP: series are distinguished by both engine and
    priority labels."""
    with patch("vllm.v1.metrics.loggers.unregister_vllm_metrics", lambda: None):
        logger = PrometheusStatLogger(_make_vllm_config("priority"), [0, 1])

    # Engine 0 finishes a high-priority (-1) request.
    logger.record(
        scheduler_stats=None,
        iteration_stats=_make_finished_request_stats(priority=-1),
        engine_idx=0,
    )
    # Engine 1 finishes a low-priority (1) request.
    logger.record(
        scheduler_stats=None,
        iteration_stats=_make_finished_request_stats(priority=1),
        engine_idx=1,
    )

    success_samples = _collect_samples("vllm:request_success")
    series = {(s.labels["engine"], s.labels["priority"]) for s in success_samples}
    assert ("0", "-1") in series
    assert ("1", "1") in series

    # The per-engine priority histogram observed one request on each engine.
    prio_samples = _collect_samples("vllm:request_priority")
    counts = {
        s.labels["engine"]: s.value for s in prio_samples if s.name.endswith("_count")
    }
    assert counts["0"] == 1.0
    assert counts["1"] == 1.0


def test_priority_finished_histogram_observed_value(logger_priority):
    """Priority mode: a finished-request histogram records the observed value
    into the correct bucket (not just the label)."""
    # _make_finished_request_stats uses num_prompt_tokens=100, which falls in
    # the (50, 100] bucket of build_1_2_5_buckets(4096).
    logger_priority.record(
        scheduler_stats=None,
        iteration_stats=_make_finished_request_stats(priority=0),
        engine_idx=0,
    )

    samples = _collect_samples("vllm:request_prompt_tokens")
    count = next(s.value for s in samples if s.name.endswith("_count"))
    assert count == 1.0
    le_50 = next(
        s.value
        for s in samples
        if s.name.endswith("_bucket") and s.labels["le"] == "50.0"
    )
    le_100 = next(
        s.value
        for s in samples
        if s.name.endswith("_bucket") and s.labels["le"] == "100.0"
    )
    assert le_50 == 0.0
    assert le_100 == 1.0
