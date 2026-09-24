# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import logging
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request

import vllm.envs as envs
from vllm.entrypoints.generate.base.serving import GenerateBaseServing
from vllm.envs import disable_envs_cache
from vllm.exceptions import GenerationError


@pytest.mark.asyncio
async def test_raise_if_error_raises_generation_error(caplog):
    """Test _raise_if_error raises GenerationError."""
    # create a minimal GenerateBaseServing instance
    mock_engine = MagicMock()
    mock_engine.model_config = MagicMock()
    mock_engine.model_config.max_model_len = 100
    mock_models = MagicMock()

    serving = GenerateBaseServing(
        engine_client=mock_engine,
        models=mock_models,
        request_logger=None,
    )

    # test that error finish_reason raises GenerationError
    with caplog.at_level(logging.ERROR), pytest.raises(GenerationError) as exc_info:
        serving._raise_if_error("error", "test-request-id")

    assert str(exc_info.value) == "Internal server error"
    assert exc_info.value.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert caplog.records[-1].request_id == "test-request-id"

    # test that other finish_reasons don't raise
    serving._raise_if_error("stop", "test-request-id")  # should not raise
    serving._raise_if_error("length", "test-request-id")  # should not raise
    serving._raise_if_error(None, "test-request-id")  # should not raise


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("external_id", "has_raw_request"),
    [("cmpl-external", True), (None, True), (None, False)],
)
async def test_kv_rejection_warning_uses_assigned_external_id(
    caplog, external_id, has_raw_request
):
    serving = object.__new__(GenerateBaseServing)
    serving.has_kv_connector = True
    serving.engine_client = SimpleNamespace(
        notify_kv_transfer_request_rejected=AsyncMock(
            side_effect=RuntimeError("notification failed")
        )
    )
    request = SimpleNamespace(
        request_id="internal-id", kv_transfer_params={"do_remote_prefill": True}
    )
    raw_request = (
        Request({"type": "http", "method": "POST", "headers": []})
        if has_raw_request
        else None
    )
    if external_id is not None:
        assert raw_request is not None
        raw_request.state.request_metadata = SimpleNamespace(request_id=external_id)

    with caplog.at_level(logging.WARNING):
        await serving._with_kv_transfer_rejection_cleanup(
            asyncio.sleep(0, result=serving.create_error_response("rejected")),
            request,
            raw_request,
        )

    record = caplog.records[-1]
    assert "internal-id" in record.getMessage()
    if external_id is None:
        assert not hasattr(record, "request_id")
    else:
        assert record.request_id == external_id


@pytest.mark.asyncio
async def test_convert_generation_error_to_streaming_response():
    """Test _convert_generation_error_to_streaming_response output."""
    mock_engine = MagicMock()
    mock_engine.model_config = MagicMock()
    mock_engine.model_config.max_model_len = 100
    mock_models = MagicMock()

    serving = GenerateBaseServing(
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

    serving = GenerateBaseServing(
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
