# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.exceptions import VLLMContentTooLargeError, VLLMValidationError


class TestVLLMContentTooLargeError:
    """Tests for VLLMContentTooLargeError exception class."""

    def test_is_subclass_of_validation_error(self):
        err = VLLMContentTooLargeError("too large")
        assert isinstance(err, VLLMValidationError)
        assert isinstance(err, ValueError)

    def test_message_and_parameter(self):
        err = VLLMContentTooLargeError("too large", parameter="max_tokens", value=65536)
        assert err.parameter == "max_tokens"
        assert err.value == 65536
        assert "too large" in str(err)

    def test_str_with_extras(self):
        err = VLLMContentTooLargeError(
            "content exceeds limit",
            parameter="input_tokens",
            value=200000,
        )
        s = str(err)
        assert "content exceeds limit" in s
        assert "parameter=input_tokens" in s
        assert "value=200000" in s


class TestCreateErrorResponse413:
    """Tests for create_error_response with VLLMContentTooLargeError."""

    @pytest.fixture
    def serving(self):
        mock_engine = MagicMock()
        mock_engine.model_config = MagicMock()
        mock_engine.model_config.max_model_len = 100
        mock_models = MagicMock()
        return OpenAIServing(
            engine_client=mock_engine,
            models=mock_models,
            request_logger=None,
        )

    def test_content_too_large_returns_413(self, serving):
        err = VLLMContentTooLargeError(
            "max_tokens is too large", parameter="max_tokens", value=65536
        )
        response = serving.create_error_response(err)
        assert isinstance(response, ErrorResponse)
        assert response.error.code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
        assert response.error.type == "ContentTooLargeError"
        assert response.error.param == "max_tokens"
        assert "max_tokens is too large" in response.error.message

    def test_content_too_large_input_tokens(self, serving):
        err = VLLMContentTooLargeError(
            "input too long", parameter="input_tokens", value=200000
        )
        response = serving.create_error_response(err)
        assert response.error.code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
        assert response.error.param == "input_tokens"

    def test_regular_validation_error_still_400(self, serving):
        err = VLLMValidationError("bad parameter", parameter="temperature", value=5.0)
        response = serving.create_error_response(err)
        assert response.error.code == HTTPStatus.BAD_REQUEST
        assert response.error.type == "BadRequestError"

    def test_value_error_still_400(self, serving):
        err = ValueError("invalid value")
        response = serving.create_error_response(err)
        assert response.error.code == HTTPStatus.BAD_REQUEST

    def test_streaming_error_response_413(self, serving):
        err = VLLMContentTooLargeError("too large", parameter="max_tokens", value=65536)
        json_str = serving.create_streaming_error_response(err)
        assert "ContentTooLargeError" in json_str
        assert "too large" in json_str
