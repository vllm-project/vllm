# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Timeout around lm-format-enforcer JSON-schema parser construction."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output.backend_lm_format_enforcer import (
    LMFormatEnforcerBackend,
    _json_schema_parser_with_timeout,
    validate_structured_output_request_lm_format_enforcer,
)
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

BENIGN_SCHEMA = {"type": "string", "pattern": "[a-z]{1,8}"}
BENIGN_SCHEMA_TEXT = json.dumps(BENIGN_SCHEMA)
SLOW_SCHEMA = {"type": "string", "pattern": "(a{1000}){1000}"}


def _slow_parser(_spec):
    time.sleep(10)
    return "never"


def test_json_schema_parser_timeout_rejects_slow_build():
    with (
        patch("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 1),
        patch(
            "vllm.v1.structured_output.backend_lm_format_enforcer.lmformatenforcer.JsonSchemaParser",
            _slow_parser,
        ),
        pytest.raises(ValueError, match="timed out"),
    ):
        _json_schema_parser_with_timeout(SLOW_SCHEMA, json.dumps(SLOW_SCHEMA))


def test_json_schema_parser_timeout_allows_benign_schema():
    parser = object()
    with patch(
        "vllm.v1.structured_output.backend_lm_format_enforcer.lmformatenforcer.JsonSchemaParser",
        return_value=parser,
    ) as mock_parser:
        result = _json_schema_parser_with_timeout(BENIGN_SCHEMA, BENIGN_SCHEMA_TEXT)
    assert result is parser
    mock_parser.assert_called_once_with(BENIGN_SCHEMA)


@pytest.mark.parametrize(
    "json_spec",
    [SLOW_SCHEMA, json.dumps(SLOW_SCHEMA)],
)
def test_json_schema_validation_times_out(json_spec):
    params = SamplingParams(structured_outputs=StructuredOutputsParams(json=json_spec))
    with (
        patch("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 1),
        patch(
            "vllm.v1.structured_output.backend_lm_format_enforcer.lmformatenforcer.JsonSchemaParser",
            _slow_parser,
        ),
        pytest.raises(VLLMValidationError, match="timed out"),
    ):
        validate_structured_output_request_lm_format_enforcer(params)


def test_json_schema_validation_still_rejects_invalid_json():
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json='{"type": ')
    )
    with pytest.raises(VLLMValidationError, match="Invalid JSON grammar"):
        validate_structured_output_request_lm_format_enforcer(params)


def test_empty_json_object_schema_is_compiled():
    params = SamplingParams(structured_outputs=StructuredOutputsParams(json={}))
    with patch(
        "vllm.v1.structured_output.backend_lm_format_enforcer.lmformatenforcer.JsonSchemaParser",
        return_value=object(),
    ) as mock_parser:
        validate_structured_output_request_lm_format_enforcer(params)
    mock_parser.assert_called_once_with({})


def test_empty_json_string_is_rejected():
    params = SamplingParams(structured_outputs=StructuredOutputsParams(json=""))
    with pytest.raises(VLLMValidationError, match="Invalid JSON grammar"):
        validate_structured_output_request_lm_format_enforcer(params)


def test_compile_grammar_json_uses_timeout_wrapper():
    vllm_config = MagicMock(speculative_config=None)
    with (
        patch(
            "vllm.v1.structured_output.backend_lm_format_enforcer._cached_build_vllm_token_enforcer_tokenizer_data"
        ),
        patch(
            "vllm.v1.structured_output.backend_lm_format_enforcer._json_schema_parser_with_timeout",
            return_value=MagicMock(),
        ) as mock_wrapper,
        patch(
            "vllm.v1.structured_output.backend_lm_format_enforcer.lmformatenforcer.TokenEnforcer",
            return_value=MagicMock(),
        ),
    ):
        backend = LMFormatEnforcerBackend(
            vllm_config=vllm_config,
            tokenizer=MagicMock(),
            vocab_size=32,
        )
        backend.compile_grammar(StructuredOutputOptions.JSON, BENIGN_SCHEMA_TEXT)

    mock_wrapper.assert_called_once_with(BENIGN_SCHEMA, BENIGN_SCHEMA_TEXT)


def test_compile_grammar_json_object_skips_timeout_wrapper():
    vllm_config = MagicMock(speculative_config=None)
    with (
        patch(
            "vllm.v1.structured_output.backend_lm_format_enforcer._cached_build_vllm_token_enforcer_tokenizer_data"
        ),
        patch(
            "vllm.v1.structured_output.backend_lm_format_enforcer._json_schema_parser_with_timeout"
        ) as mock_wrapper,
        patch(
            "vllm.v1.structured_output.backend_lm_format_enforcer.lmformatenforcer.JsonSchemaParser",
            return_value=MagicMock(),
        ) as mock_parser,
        patch(
            "vllm.v1.structured_output.backend_lm_format_enforcer.lmformatenforcer.TokenEnforcer",
            return_value=MagicMock(),
        ),
    ):
        backend = LMFormatEnforcerBackend(
            vllm_config=vllm_config,
            tokenizer=MagicMock(),
            vocab_size=32,
        )
        backend.compile_grammar(StructuredOutputOptions.JSON_OBJECT, "")

    mock_wrapper.assert_not_called()
    mock_parser.assert_called_once_with(None)
