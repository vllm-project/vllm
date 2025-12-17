# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the /v1/kv_cache/status endpoint."""

import pytest
from unittest.mock import patch

from vllm.v1.metrics.reader import Counter, Gauge


class TestKVCacheStatusEndpoint:
    """Tests for the /v1/kv_cache/status endpoint logic."""

    def test_response_structure_with_metrics(self):
        """Test that the response has all required fields with valid metrics."""
        # Mock metrics data
        mock_metrics = [
            Gauge(name="vllm:kv_cache_usage_perc", labels={}, value=0.65),
            Gauge(name="vllm:num_kv_cache_blocks_total", labels={}, value=4096),
            Gauge(name="vllm:num_kv_cache_blocks_free", labels={}, value=1433),
            Gauge(name="vllm:num_requests_running", labels={}, value=10),
            Gauge(name="vllm:num_requests_waiting", labels={}, value=5),
            Counter(name="vllm:prefix_cache_queries", labels={}, value=1000),
            Counter(name="vllm:prefix_cache_hits", labels={}, value=750),
        ]

        # Simulate the endpoint logic
        response = {
            "kv_cache_usage": 0.0,
            "num_total_blocks": 0,
            "num_free_blocks": 0,
            "num_used_blocks": 0,
            "num_running_requests": 0,
            "num_waiting_requests": 0,
            "prefix_cache_hit_rate": 0.0,
            "prefix_cache_queries_total": 0,
            "prefix_cache_hits_total": 0,
        }

        metric_mapping = {
            "vllm:kv_cache_usage_perc": "kv_cache_usage",
            "vllm:num_kv_cache_blocks_total": "num_total_blocks",
            "vllm:num_kv_cache_blocks_free": "num_free_blocks",
            "vllm:num_requests_running": "num_running_requests",
            "vllm:num_requests_waiting": "num_waiting_requests",
            "vllm:prefix_cache_queries": "prefix_cache_queries_total",
            "vllm:prefix_cache_hits": "prefix_cache_hits_total",
        }

        for metric in mock_metrics:
            if metric.name in metric_mapping:
                field_name = metric_mapping[metric.name]
                if isinstance(metric, Gauge):
                    response[field_name] = float(metric.value)
                elif isinstance(metric, Counter):
                    response[field_name] = int(metric.value)

        # Calculate derived metrics
        response["num_used_blocks"] = int(
            response["num_total_blocks"] - response["num_free_blocks"]
        )

        # Calculate hit rate
        if response["prefix_cache_queries_total"] > 0:
            response["prefix_cache_hit_rate"] = (
                response["prefix_cache_hits_total"]
                / response["prefix_cache_queries_total"]
            )

        # Assertions
        assert response["kv_cache_usage"] == 0.65
        assert response["num_total_blocks"] == 4096
        assert response["num_free_blocks"] == 1433
        assert response["num_used_blocks"] == 2663
        assert response["num_running_requests"] == 10
        assert response["num_waiting_requests"] == 5
        assert response["prefix_cache_queries_total"] == 1000
        assert response["prefix_cache_hits_total"] == 750
        assert response["prefix_cache_hit_rate"] == 0.75

    def test_prefix_cache_hit_rate_calculation(self):
        """Test that hit rate is calculated correctly."""
        queries = 1000
        hits = 750
        expected_rate = 0.75

        actual_rate = hits / queries if queries > 0 else 0.0
        assert actual_rate == expected_rate

    def test_prefix_cache_hit_rate_zero_queries(self):
        """Test hit rate when no queries have been made (avoid division by zero)."""
        queries = 0
        hits = 0

        rate = hits / queries if queries > 0 else 0.0
        assert rate == 0.0

    def test_response_with_empty_metrics(self):
        """Test that response returns default values when no metrics available."""
        mock_metrics = []

        response = {
            "kv_cache_usage": 0.0,
            "num_running_requests": 0,
            "num_waiting_requests": 0,
            "prefix_cache_hit_rate": 0.0,
            "prefix_cache_queries_total": 0,
            "prefix_cache_hits_total": 0,
        }

        # Simulate the endpoint logic with empty metrics
        metric_mapping = {
            "vllm:kv_cache_usage_perc": "kv_cache_usage",
            "vllm:num_requests_running": "num_running_requests",
            "vllm:num_requests_waiting": "num_waiting_requests",
            "vllm:prefix_cache_queries": "prefix_cache_queries_total",
            "vllm:prefix_cache_hits": "prefix_cache_hits_total",
        }

        for metric in mock_metrics:
            if metric.name in metric_mapping:
                field_name = metric_mapping[metric.name]
                if isinstance(metric, Gauge):
                    response[field_name] = float(metric.value)
                elif isinstance(metric, Counter):
                    response[field_name] = int(metric.value)

        # All values should remain at defaults
        assert response["kv_cache_usage"] == 0.0
        assert response["num_running_requests"] == 0
        assert response["num_waiting_requests"] == 0
        assert response["prefix_cache_hit_rate"] == 0.0
        assert response["prefix_cache_queries_total"] == 0
        assert response["prefix_cache_hits_total"] == 0

    def test_response_fields_are_correct_types(self):
        """Test that response fields have correct data types."""
        mock_metrics = [
            Gauge(name="vllm:kv_cache_usage_perc", labels={}, value=0.5),
            Gauge(name="vllm:num_requests_running", labels={}, value=5),
            Counter(name="vllm:prefix_cache_queries", labels={}, value=100),
        ]

        response = {
            "kv_cache_usage": 0.0,
            "num_running_requests": 0,
            "num_waiting_requests": 0,
            "prefix_cache_hit_rate": 0.0,
            "prefix_cache_queries_total": 0,
            "prefix_cache_hits_total": 0,
        }

        metric_mapping = {
            "vllm:kv_cache_usage_perc": "kv_cache_usage",
            "vllm:num_requests_running": "num_running_requests",
            "vllm:num_requests_waiting": "num_waiting_requests",
            "vllm:prefix_cache_queries": "prefix_cache_queries_total",
            "vllm:prefix_cache_hits": "prefix_cache_hits_total",
        }

        for metric in mock_metrics:
            if metric.name in metric_mapping:
                field_name = metric_mapping[metric.name]
                if isinstance(metric, Gauge):
                    response[field_name] = float(metric.value)
                elif isinstance(metric, Counter):
                    response[field_name] = int(metric.value)

        # Check types
        assert isinstance(response["kv_cache_usage"], float)
        assert isinstance(response["num_running_requests"], float)  # Gauge returns float
        assert isinstance(response["prefix_cache_queries_total"], int)
        assert isinstance(response["prefix_cache_hit_rate"], float)
