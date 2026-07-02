# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ResponsesRequest.to_sampling_params() and
ResponsesRequest.build_tok_params() parameter mapping."""

from types import SimpleNamespace

import pytest
import torch
from openai.types.responses.response_format_text_json_schema_config import (
    ResponseFormatTextJSONSchemaConfig,
)
from pydantic import ValidationError

from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponseTextConfig,
)
from vllm.sampling_params import StructuredOutputsParams


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

    def test_structured_outputs_passed_through(self):
        """Test that structured_outputs field is passed to SamplingParams."""
        structured_outputs = StructuredOutputsParams(grammar="root ::= 'hello'")
        request = ResponsesRequest(
            model="test-model",
            input="test input",
            structured_outputs=structured_outputs,
        )

        sampling_params = request.to_sampling_params(default_max_tokens=1000)

        assert sampling_params.structured_outputs is not None
        assert sampling_params.structured_outputs.grammar == "root ::= 'hello'"

    def test_structured_outputs_and_json_schema_conflict(self):
        """Test that specifying both structured_outputs and json_schema raises."""
        structured_outputs = StructuredOutputsParams(grammar="root ::= 'hello'")
        text_config = ResponseTextConfig()
        text_config.format = ResponseFormatTextJSONSchemaConfig(
            type="json_schema",
            name="test",
            schema={"type": "object"},
        )
        request = ResponsesRequest(
            model="test-model",
            input="test input",
            structured_outputs=structured_outputs,
            text=text_config,
        )

        with pytest.raises(ValueError) as exc_info:
            request.to_sampling_params(default_max_tokens=1000)

        assert "Cannot specify both structured_outputs and text.format" in str(
            exc_info.value
        )


class TestResponsesRequestTokParams:
    """Test that ResponsesRequest correctly builds TokenizeParams."""

    def test_add_special_tokens_false(self):
        """add_special_tokens must be False to avoid double BOS on models
        whose chat template already includes a BOS token."""
        request = ResponsesRequest(
            model="test-model",
            input="test input",
        )
        model_config = SimpleNamespace(max_model_len=4096)
        tok_params = request.build_tok_params(model_config)
        assert tok_params.add_special_tokens is False

    def test_truncation_disabled(self):
        """truncate_prompt_tokens should be None when truncation is disabled."""
        request = ResponsesRequest(
            model="test-model",
            input="test input",
            truncation="disabled",
        )
        model_config = SimpleNamespace(max_model_len=4096)
        tok_params = request.build_tok_params(model_config)
        assert tok_params.truncate_prompt_tokens is None

    def test_truncation_auto(self):
        """truncate_prompt_tokens should be -1 when truncation is auto."""
        request = ResponsesRequest(
            model="test-model",
            input="test input",
            truncation="auto",
        )
        model_config = SimpleNamespace(max_model_len=4096)
        tok_params = request.build_tok_params(model_config)
        assert tok_params.truncate_prompt_tokens == -1

    def test_max_output_tokens(self):
        """max_output_tokens is correctly passed through."""
        request = ResponsesRequest(
            model="test-model",
            input="test input",
            max_output_tokens=512,
        )
        model_config = SimpleNamespace(max_model_len=4096)
        tok_params = request.build_tok_params(model_config)
        assert tok_params.max_total_tokens == 4096
        assert tok_params.max_output_tokens == 512
