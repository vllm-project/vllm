# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ResponsesRequest.to_sampling_params() parameter mapping."""

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
from vllm.exceptions import VLLMValidationError
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

    def test_text_format_json_object_enables_structured_outputs(self):
        """text.format json_object enables structured outputs for sampling."""
        request = ResponsesRequest(
            model="test-model",
            input="test input",
            text=ResponseTextConfig.model_validate({"format": {"type": "json_object"}}),
        )

        sampling_params = request.to_sampling_params(default_max_tokens=1000)

        assert sampling_params.structured_outputs is not None
        assert sampling_params.structured_outputs.json_object is True
        assert sampling_params.structured_outputs.json is None
        assert request.structured_outputs is None

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

        with pytest.raises(VLLMValidationError) as exc_info:
            request.to_sampling_params(default_max_tokens=1000)

        assert "Cannot specify both structured_outputs and text.format" in str(
            exc_info.value
        )


class TestResponsesStopTokenIds:
    """Test stop_token_ids merging in ResponsesRequest.to_sampling_params()."""

    @pytest.fixture
    def minimal_responses_request(self):
        return ResponsesRequest(
            model="test-model",
            input="hello",
        )

    def test_default_stop_token_ids_applied(self, minimal_responses_request):
        """Server-default stop_token_ids are applied when client sends none."""
        default_sampling_params = {
            "stop_token_ids": [200012, 200002],
        }

        sampling_params = minimal_responses_request.to_sampling_params(
            default_max_tokens=100,
            default_sampling_params=default_sampling_params,
        )

        assert set(sampling_params.stop_token_ids) == {200012, 200002}

    def test_client_stop_token_ids_merged_with_defaults(self):
        """Client-specified stop_token_ids are merged with server defaults."""
        request = ResponsesRequest(
            model="test-model",
            input="hello",
            stop_token_ids=[99999],
        )
        default_sampling_params = {
            "stop_token_ids": [200012, 200002],
        }

        sampling_params = request.to_sampling_params(
            default_max_tokens=100,
            default_sampling_params=default_sampling_params,
        )

        assert set(sampling_params.stop_token_ids) == {200012, 200002, 99999}
        assert sampling_params.stop_token_ids == [99999, 200012, 200002]

    def test_no_stop_token_ids_anywhere(self, minimal_responses_request):
        """When neither client nor server specifies stop_token_ids, result is empty."""
        sampling_params = minimal_responses_request.to_sampling_params(
            default_max_tokens=100,
            default_sampling_params={},
        )

        assert not sampling_params.stop_token_ids

    def test_only_client_stop_token_ids(self):
        """Client stop_token_ids work when no server defaults exist."""
        request = ResponsesRequest(
            model="test-model",
            input="hello",
            stop_token_ids=[42, 43],
        )

        sampling_params = request.to_sampling_params(
            default_max_tokens=100,
            default_sampling_params={},
        )

        assert set(sampling_params.stop_token_ids) == {42, 43}

    def test_duplicate_stop_token_ids_deduplicated(self):
        """Overlapping stop_token_ids between client and server are deduplicated."""
        request = ResponsesRequest(
            model="test-model",
            input="hello",
            stop_token_ids=[200012, 55555],
        )
        default_sampling_params = {
            "stop_token_ids": [200012, 200002],
        }

        sampling_params = request.to_sampling_params(
            default_max_tokens=100,
            default_sampling_params=default_sampling_params,
        )

        assert set(sampling_params.stop_token_ids) == {200012, 200002, 55555}
        assert sampling_params.stop_token_ids == [200012, 55555, 200002]
        assert len(sampling_params.stop_token_ids) == 3

    def test_stop_token_ids_field_is_not_ignored(self):
        """Constructing ResponsesRequest with stop_token_ids binds the field."""
        request = ResponsesRequest(
            model="test-model",
            input="hello",
            stop_token_ids=[200012],
        )

        assert request.stop_token_ids == [200012]
        assert "stop_token_ids" not in (request.model_extra or {})
