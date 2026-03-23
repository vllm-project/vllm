# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for response format schema extraction and developer message injection.

These tests verify that structured output schemas are correctly extracted from
ResponsesRequest and injected into the Harmony developer message per the
Harmony cookbook specification.
"""

from openai.types.responses.response_format_text_json_schema_config import (
    ResponseFormatTextJSONSchemaConfig,
)

from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponseTextConfig,
)
from vllm.entrypoints.openai.responses.serving import (
    _extract_response_format_schema,
)
from vllm.sampling_params import StructuredOutputsParams


def _make_json_schema_text_config(schema: dict) -> ResponseTextConfig:
    text_config = ResponseTextConfig()
    text_config.format = ResponseFormatTextJSONSchemaConfig(
        type="json_schema",
        name="test_schema",
        schema=schema,
    )
    return text_config


class TestExtractResponseFormatSchema:
    def test_extracts_from_text_format_json_schema(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        request = ResponsesRequest(
            model="test-model",
            input="test",
            text=_make_json_schema_text_config(schema),
        )
        result = _extract_response_format_schema(request)
        assert result == schema

    def test_extracts_from_structured_outputs_json(self):
        schema = {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
        }
        request = ResponsesRequest(
            model="test-model",
            input="test",
            structured_outputs=StructuredOutputsParams(json=schema),
        )
        result = _extract_response_format_schema(request)
        assert result == schema

    def test_returns_none_for_text_format(self):
        request = ResponsesRequest(
            model="test-model",
            input="test",
            text=ResponseTextConfig(format={"type": "text"}),
        )
        result = _extract_response_format_schema(request)
        assert result is None

    def test_returns_none_for_no_format(self):
        request = ResponsesRequest(
            model="test-model",
            input="test",
        )
        result = _extract_response_format_schema(request)
        assert result is None

    def test_text_format_takes_precedence(self):
        """text.format.json_schema is checked before structured_outputs."""
        text_schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
        }
        so_schema = {
            "type": "object",
            "properties": {"b": {"type": "string"}},
        }
        request = ResponsesRequest(
            model="test-model",
            input="test",
            text=_make_json_schema_text_config(text_schema),
            structured_outputs=StructuredOutputsParams(json=so_schema),
        )
        result = _extract_response_format_schema(request)
        assert result == text_schema
