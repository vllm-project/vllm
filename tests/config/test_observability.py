# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ObservabilityConfig validation."""

from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter, ValidationError

from vllm.config import DeviceConfig, VllmConfig
from vllm.config.observability import ObservabilityConfig


def test_iteration_timing_is_not_a_constructor_option():
    with pytest.raises(ValidationError, match="iteration_timing_enabled"):
        TypeAdapter(ObservabilityConfig).validate_python(
            {"iteration_timing_enabled": True}
        )


def test_iteration_workload_enables_gpu_timing_without_logging():
    assert not ObservabilityConfig().enable_iteration_workload_metrics
    config = ObservabilityConfig(enable_iteration_workload_metrics=True)
    assert config.iteration_timing_enabled
    assert not config.enable_logging_iteration_details
    with pytest.raises(ValidationError, match="Iteration GPU timing requires a CUDA"):
        VllmConfig(
            device_config=DeviceConfig(device="cpu"), observability_config=config
        )

    runtime_config = VllmConfig()
    runtime_config.model_config = SimpleNamespace(runner_type="generate")
    runtime_config.device_config.device_type = "cuda"
    runtime_config.observability_config = config
    runtime_config.resolve_iteration_timing(log_stats=False, multiprocess_mode=True)
    assert config.iteration_timing_enabled
    config.enable_iteration_workload_metrics = False
    runtime_config.resolve_iteration_timing(log_stats=False, multiprocess_mode=True)
    assert not config.iteration_timing_enabled


def test_custom_histogram_buckets_default_none():
    assert ObservabilityConfig().custom_histogram_buckets is None


def test_custom_histogram_buckets_valid():
    config = ObservabilityConfig(
        custom_histogram_buckets={
            "request_latency": [0.01, 0.05, 0.1, 0.5],
            "time_to_first_token": [1, 2, 5],
        }
    )
    assert config.custom_histogram_buckets == {
        "request_latency": [0.01, 0.05, 0.1, 0.5],
        "time_to_first_token": [1.0, 2.0, 5.0],
    }


@pytest.mark.parametrize(
    ("buckets", "match"),
    [
        ({"bogus": [1.0, 2.0]}, "unknown bucket family 'bogus'"),
        ({"request_latency": []}, "must not be empty"),
        ({"request_latency": [0.0, 1.0]}, "must be finite and greater than 0"),
        ({"request_latency": [-1.0, 1.0]}, "must be finite and greater than 0"),
        (
            {"request_latency": [1.0, float("inf")]},
            "must be finite and greater than 0",
        ),
        ({"request_latency": [float("nan")]}, "must be finite and greater than 0"),
        ({"request_latency": [1.0, 1.0]}, "must be strictly increasing"),
        ({"request_latency": [2.0, 1.0]}, "must be strictly increasing"),
        ({"request_latency": [True, 2.0]}, "must be a number, not a boolean"),
        ({"request_latency": [False]}, "must be a number, not a boolean"),
        (
            {"request_latency": [1.0, 2.0], "request_tokens": [16, 8]},
            r"request_tokens.*must be strictly increasing",
        ),
    ],
)
def test_custom_histogram_buckets_rejects_invalid(buckets, match):
    """Each validation rule must name the field, the value, and the rule."""
    with pytest.raises(ValidationError, match=match):
        ObservabilityConfig(custom_histogram_buckets=buckets)
