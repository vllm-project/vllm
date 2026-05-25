# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus
from unittest.mock import MagicMock

import pytest

import vllm.envs as envs
from vllm.entrypoints.openai.engine.serving import GenerationError, OpenAIServing
from vllm.envs import disable_envs_cache


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


def test_is_model_supported_skip_name_validation_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When VLLM_SKIP_MODEL_NAME_VALIDATION is set, accept any model id."""
    disable_envs_cache()
    monkeypatch.delenv("VLLM_SKIP_MODEL_NAME_VALIDATION", raising=False)

    mock_engine = MagicMock()
    mock_engine.model_config = MagicMock()
    mock_engine.model_config.max_model_len = 100
    mock_models = MagicMock()
    mock_models.is_base_model.return_value = False

    serving = OpenAIServing(
        engine_client=mock_engine,
        models=mock_models,
        request_logger=None,
    )

    assert serving._is_model_supported("not-a-registered-model") is False

    monkeypatch.setenv("VLLM_SKIP_MODEL_NAME_VALIDATION", "1")
    disable_envs_cache()
    assert envs.VLLM_SKIP_MODEL_NAME_VALIDATION is True
    assert serving._is_model_supported("not-a-registered-model") is True

    monkeypatch.setenv("VLLM_SKIP_MODEL_NAME_VALIDATION", "true")
    disable_envs_cache()
    assert envs.VLLM_SKIP_MODEL_NAME_VALIDATION is True
    assert serving._is_model_supported("another-alias") is True
