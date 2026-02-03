# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ResponsesRequest.to_sampling_params() parameter mapping."""

import pytest

from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class TestResponsesRequestSamplingParams:
    """Test that ResponsesRequest correctly maps parameters to SamplingParams."""

    def test_basic_sampling_params(self):
        """Test basic sampling parameters are correctly mapped."""
        request = ResponsesRequest(
            model="test-model",
            input="test input",
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            max_output_tokens=100,
        )

        sampling_params = request.to_sampling_params(default_max_tokens=1000)

        assert sampling_params.temperature == 0.8
        assert sampling_params.top_p == 0.95
        assert sampling_params.top_k == 50
        assert sampling_params.max_tokens == 100

    def test_extra_sampling_params(self):
        """Test extra sampling parameters are correctly mapped."""
        request = ResponsesRequest(
            model="test-model",
            input="test input",
            repetition_penalty=1.2,
            seed=42,
            stop=["END", "STOP"],
            ignore_eos=True,
            vllm_xargs={"custom": "value"},
        )

        sampling_params = request.to_sampling_params(default_max_tokens=1000)

        assert sampling_params.repetition_penalty == 1.2
        assert sampling_params.seed == 42
        assert sampling_params.stop == ["END", "STOP"]
        assert sampling_params.ignore_eos is True
        assert sampling_params.extra_args == {"custom": "value"}

    def test_stop_string_conversion(self):
        """Test that single stop string is converted to list."""
        request = ResponsesRequest(
            model="test-model",
            input="test input",
            stop="STOP",
        )

        sampling_params = request.to_sampling_params(default_max_tokens=1000)

        assert sampling_params.stop == ["STOP"]

    def test_default_values(self):
        """Test default values for optional parameters."""
        request = ResponsesRequest(
            model="test-model",
            input="test input",
        )

        sampling_params = request.to_sampling_params(default_max_tokens=1000)

        assert sampling_params.repetition_penalty == 1.0  # None → 1.0
        assert sampling_params.stop == []  # Empty list
        assert sampling_params.extra_args == {}  # Empty dict

    def test_seed_bounds_validation(self):
        """Test that seed values outside torch.long bounds are rejected."""
        import torch
        from pydantic import ValidationError

        # Test seed below minimum
        with pytest.raises(ValidationError) as exc_info:
            ResponsesRequest(
                model="test-model",
                input="test input",
                seed=torch.iinfo(torch.long).min - 1,
            )
        assert "greater_than_equal" in str(exc_info.value).lower()

        # Test seed above maximum
        with pytest.raises(ValidationError) as exc_info:
            ResponsesRequest(
                model="test-model",
                input="test input",
                seed=torch.iinfo(torch.long).max + 1,
            )
        assert "less_than_equal" in str(exc_info.value).lower()

        # Test valid seed at boundaries
        request_min = ResponsesRequest(
            model="test-model",
            input="test input",
            seed=torch.iinfo(torch.long).min,
        )
        assert request_min.seed == torch.iinfo(torch.long).min

        request_max = ResponsesRequest(
            model="test-model",
            input="test input",
            seed=torch.iinfo(torch.long).max,
        )
        assert request_max.seed == torch.iinfo(torch.long).max
