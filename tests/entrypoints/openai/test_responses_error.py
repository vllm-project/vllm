# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.protocol import ErrorResponse
from vllm.entrypoints.openai.serving_engine import GenerationError, OpenAIServing


@pytest.mark.asyncio
async def test_raise_if_error_raises_generation_error():
    """test _raise_if_error raises GenerationError"""
    # create a minimal OpenAIServing instance
    mock_engine = MagicMock()
    mock_engine.model_config = MagicMock()
    mock_engine.model_config.max_model_len = 100
    mock_models = MagicMock()

    serving = OpenAIServing(
        engine_client=mock_engine,
        models=mock_models,
        request_logger=None,
    )

    # test that error finish_reason raises GenerationError
    with pytest.raises(GenerationError) as exc_info:
        serving._raise_if_error("error", "test-request-id")

    assert str(exc_info.value) == "Internal server error"
    assert exc_info.value.status_code == HTTPStatus.INTERNAL_SERVER_ERROR

    # test that other finish_reasons don't raise
    serving._raise_if_error("stop", "test-request-id")  # should not raise
    serving._raise_if_error("length", "test-request-id")  # should not raise
    serving._raise_if_error(None, "test-request-id")  # should not raise


@pytest.mark.asyncio
async def test_convert_generation_error_to_response():
    """test _convert_generation_error_to_response creates proper ErrorResponse"""
    mock_engine = MagicMock()
    mock_engine.model_config = MagicMock()
    mock_engine.model_config.max_model_len = 100
    mock_models = MagicMock()

    serving = OpenAIServing(
        engine_client=mock_engine,
        models=mock_models,
        request_logger=None,
    )

    # create a GenerationError
    gen_error = GenerationError("Internal server error")

    # convert to ErrorResponse
    error_response = serving._convert_generation_error_to_response(gen_error)

    assert isinstance(error_response, ErrorResponse)
    assert error_response.error.type == "InternalServerError"
    assert error_response.error.message == "Internal server error"
    assert error_response.error.code == HTTPStatus.INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
async def test_convert_generation_error_to_streaming_response():
    """test _convert_generation_error_to_streaming_response output"""
    mock_engine = MagicMock()
    mock_engine.model_config = MagicMock()
    mock_engine.model_config.max_model_len = 100
    mock_models = MagicMock()

    serving = OpenAIServing(
        engine_client=mock_engine,
        models=mock_models,
        request_logger=None,
    )

    # create a GenerationError
    gen_error = GenerationError("Internal server error")

    # convert to streaming error response
    error_json = serving._convert_generation_error_to_streaming_response(gen_error)

    assert isinstance(error_json, str)
    assert "Internal server error" in error_json
    assert "InternalServerError" in error_json
