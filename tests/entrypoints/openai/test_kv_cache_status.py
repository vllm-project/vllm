# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the /v1/kv_cache/status endpoint."""

import asyncio
import json
from unittest.mock import patch

import pytest

from vllm.v1.metrics.reader import Counter, Gauge


class TestKVCacheStatusEndpoint:
    """Tests for the /v1/kv_cache/status endpoint logic."""

    @patch('vllm.entrypoints.openai.api_server.get_metrics_snapshot')
    def test_response_structure_with_metrics(self, mock_get_metrics_snapshot):
        """Test that the response has all required fields with valid metrics."""
        from vllm.entrypoints.openai.api_server import get_kv_cache_status

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
        mock_get_metrics_snapshot.return_value = mock_metrics

        # Mock request object (not used in current implementation)
        class MockRequest:
            pass

        # Run the async endpoint function
        response_obj = asyncio.run(get_kv_cache_status(MockRequest()))
        response = json.loads(response_obj.body)

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

    @patch('vllm.entrypoints.openai.api_server.get_metrics_snapshot')
    def test_prefix_cache_hit_rate_calculation(self, mock_get_metrics_snapshot):
        """Test that hit rate is calculated correctly."""
        from vllm.entrypoints.openai.api_server import get_kv_cache_status

        mock_metrics = [
            Counter(name="vllm:prefix_cache_queries", labels={}, value=1000),
            Counter(name="vllm:prefix_cache_hits", labels={}, value=750),
        ]
        mock_get_metrics_snapshot.return_value = mock_metrics

        class MockRequest:
            pass

        response_obj = asyncio.run(get_kv_cache_status(MockRequest()))
        response = json.loads(response_obj.body)

        assert response["prefix_cache_hit_rate"] == 0.75

    @patch('vllm.entrypoints.openai.api_server.get_metrics_snapshot')
    def test_prefix_cache_hit_rate_zero_queries(self, mock_get_metrics_snapshot):
        """Test hit rate when no queries have been made (avoid division by zero)."""
        from vllm.entrypoints.openai.api_server import get_kv_cache_status

        mock_metrics = [
            Counter(name="vllm:prefix_cache_queries", labels={}, value=0),
            Counter(name="vllm:prefix_cache_hits", labels={}, value=0),
        ]
        mock_get_metrics_snapshot.return_value = mock_metrics

        class MockRequest:
            pass

        response_obj = asyncio.run(get_kv_cache_status(MockRequest()))
        response = json.loads(response_obj.body)

        assert response["prefix_cache_hit_rate"] == 0.0

    @patch('vllm.entrypoints.openai.api_server.get_metrics_snapshot')
    def test_response_with_empty_metrics(self, mock_get_metrics_snapshot):
        """Test that response returns default values when no metrics available."""
        from vllm.entrypoints.openai.api_server import get_kv_cache_status

        mock_get_metrics_snapshot.return_value = []

        class MockRequest:
            pass

        response_obj = asyncio.run(get_kv_cache_status(MockRequest()))
        response = json.loads(response_obj.body)

        # All values should be at defaults
        assert response["kv_cache_usage"] == 0.0
        assert response["num_total_blocks"] == 0
        assert response["num_free_blocks"] == 0
        assert response["num_used_blocks"] == 0
        assert response["num_running_requests"] == 0
        assert response["num_waiting_requests"] == 0
        assert response["prefix_cache_hit_rate"] == 0.0
        assert response["prefix_cache_queries_total"] == 0
        assert response["prefix_cache_hits_total"] == 0

    @patch('vllm.entrypoints.openai.api_server.get_metrics_snapshot')
    def test_num_used_blocks_calculation(self, mock_get_metrics_snapshot):
        """Test that num_used_blocks is correctly calculated."""
        from vllm.entrypoints.openai.api_server import get_kv_cache_status

        mock_metrics = [
            Gauge(name="vllm:num_kv_cache_blocks_total", labels={}, value=4096),
            Gauge(name="vllm:num_kv_cache_blocks_free", labels={}, value=1000),
        ]
        mock_get_metrics_snapshot.return_value = mock_metrics

        class MockRequest:
            pass

        response_obj = asyncio.run(get_kv_cache_status(MockRequest()))
        response = json.loads(response_obj.body)

        assert response["num_total_blocks"] == 4096
        assert response["num_free_blocks"] == 1000
        assert response["num_used_blocks"] == 3096  # 4096 - 1000
