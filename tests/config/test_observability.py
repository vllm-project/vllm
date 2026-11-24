# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ObservabilityConfig validation."""

import logging

import pytest
from pydantic import ValidationError

from vllm.config.observability import ObservabilityConfig


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


def test_enable_layerwise_nvtx_tracing_is_deprecated(caplog_vllm):
    msg = "enable_layerwise_nvtx_tracing is deprecated"
    with caplog_vllm.at_level(logging.WARNING, logger="vllm"):
        ObservabilityConfig()
        assert msg not in caplog_vllm.text
        ObservabilityConfig(enable_layerwise_nvtx_tracing=True)
    assert msg in caplog_vllm.text


def test_prometheus_request_labels_default_none():
    assert ObservabilityConfig().prometheus_request_labels is None


def test_prometheus_request_labels_valid():
    mapping = {"x-fairness-id": "fairness_id", "x-client-id": "client_id"}
    config = ObservabilityConfig(prometheus_request_labels=mapping)
    assert config.prometheus_request_labels == mapping


@pytest.mark.parametrize(
    ("labels", "match"),
    [
        ({"x-a-b": "1bad"}, "not a valid Prometheus label name"),
        ({"x-a-b": "has-dash"}, "not a valid Prometheus label name"),
        ({"x-a-b": "__tenant"}, "reserves for internal labels"),
        ({"x-a-b": "model_name"}, "collides with a label"),
        ({"x-a-b": "engine"}, "collides with a label"),
        ({"x-a-b": "finished_reason"}, "collides with a label"),
        ({"x-a-b": "le"}, "collides with a label"),
        ({"bad header": "ok_name"}, "not a valid HTTP header name"),
        ({"x-a": "same", "x-b": "same"}, "label names must be distinct"),
        (
            {f"x-h{i}": f"l{i}" for i in range(6)},
            "at most 5 labels are allowed",
        ),
    ],
)
def test_prometheus_request_labels_rejects_invalid(labels, match):
    """Invalid header/label mappings must fail fast at startup."""
    with pytest.raises(ValidationError, match=match):
        ObservabilityConfig(prometheus_request_labels=labels)
