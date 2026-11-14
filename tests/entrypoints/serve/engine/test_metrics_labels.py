# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for BaseServing._get_metrics_labels header extraction."""

from starlette.datastructures import Headers

from vllm.entrypoints.serve.engine.serving import (
    METRICS_LABEL_VALUE_MAX_LEN,
    BaseServing,
)

MAPPING = {"x-fairness-id": "fairness_id", "x-client-id": "client_id"}


def test_extracts_only_allowlisted_headers():
    headers = Headers(
        {
            "x-fairness-id": "premium",
            "x-client-id": "client-a",
            "x-unrelated": "must-not-appear",
        }
    )
    assert BaseServing._get_metrics_labels(headers, MAPPING) == {
        "fairness_id": "premium",
        "client_id": "client-a",
    }


def test_missing_header_records_empty_value():
    """A missing header still gets an entry, so the label set stays fixed."""
    headers = Headers({"x-fairness-id": "premium"})
    assert BaseServing._get_metrics_labels(headers, MAPPING) == {
        "fairness_id": "premium",
        "client_id": "",
    }


def test_header_lookup_is_case_insensitive():
    headers = Headers({"X-FAIRNESS-ID": "premium"})
    labels = BaseServing._get_metrics_labels(headers, MAPPING)
    assert labels is not None
    assert labels["fairness_id"] == "premium"


def test_values_are_truncated():
    headers = Headers({"x-fairness-id": "x" * (METRICS_LABEL_VALUE_MAX_LEN + 50)})
    labels = BaseServing._get_metrics_labels(headers, MAPPING)
    assert labels is not None
    assert len(labels["fairness_id"]) == METRICS_LABEL_VALUE_MAX_LEN


def test_no_mapping_returns_none():
    headers = Headers({"x-fairness-id": "premium"})
    assert BaseServing._get_metrics_labels(headers, None) is None
    assert BaseServing._get_metrics_labels(headers, {}) is None
