# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.config.observability import ObservabilityConfig


class TestHistogramBuckets:
    """Tests for the histogram_buckets configuration field."""

    def test_default_none(self):
        config = ObservabilityConfig()
        assert config.histogram_buckets is None
        assert config.parsed_histogram_buckets == {}

    def test_valid_json(self):
        buckets = {"ttft": [0.01, 0.1, 1.0], "e2e": [0.5, 1.0, 5.0]}
        config = ObservabilityConfig(histogram_buckets=json.dumps(buckets))
        assert config.parsed_histogram_buckets == buckets

    def test_get_histogram_buckets_match(self):
        buckets = {"ttft": [0.01, 0.1, 1.0]}
        config = ObservabilityConfig(histogram_buckets=json.dumps(buckets))
        result = config.get_histogram_buckets("time_to_first_token_seconds", [0.5, 1.0])
        # "ttft" is a substring of "time_to_first_token_seconds"? No.
        # Let's test with exact match pattern
        assert result == [0.5, 1.0]

        # Use a matching pattern
        buckets = {"time_to_first_token": [0.01, 0.1, 1.0]}
        config = ObservabilityConfig(histogram_buckets=json.dumps(buckets))
        result = config.get_histogram_buckets("time_to_first_token_seconds", [0.5, 1.0])
        assert result == [0.01, 0.1, 1.0]

    def test_get_histogram_buckets_no_match(self):
        buckets = {"ttft": [0.01, 0.1, 1.0]}
        config = ObservabilityConfig(histogram_buckets=json.dumps(buckets))
        default = [0.5, 1.0, 5.0]
        result = config.get_histogram_buckets("e2e_request_latency", default)
        assert result == default

    def test_get_histogram_buckets_none_config(self):
        config = ObservabilityConfig()
        default = [0.5, 1.0, 5.0]
        result = config.get_histogram_buckets("any_metric", default)
        assert result == default

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="valid JSON"):
            ObservabilityConfig(histogram_buckets="not json")

    def test_non_dict_raises(self):
        with pytest.raises(ValueError, match="JSON object"):
            ObservabilityConfig(histogram_buckets="[1, 2, 3]")

    def test_non_numeric_buckets_raises(self):
        with pytest.raises(ValueError, match="list of numbers"):
            ObservabilityConfig(histogram_buckets=json.dumps({"ttft": ["a", "b"]}))

    def test_unsorted_buckets_raises(self):
        with pytest.raises(ValueError, match="ascending order"):
            ObservabilityConfig(histogram_buckets=json.dumps({"ttft": [1.0, 0.5, 0.1]}))

    def test_multiple_patterns(self):
        buckets = {
            "ttft": [0.01, 0.1],
            "e2e": [1.0, 10.0],
            "latency": [0.5, 5.0],
        }
        config = ObservabilityConfig(histogram_buckets=json.dumps(buckets))
        parsed = config.parsed_histogram_buckets
        assert len(parsed) == 3
        assert parsed["ttft"] == [0.01, 0.1]
        assert parsed["e2e"] == [1.0, 10.0]
        assert parsed["latency"] == [0.5, 5.0]
