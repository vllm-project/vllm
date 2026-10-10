# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ObservabilityConfig validation."""

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
