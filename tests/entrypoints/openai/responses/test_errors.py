# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import vllm.envs as envs
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.generate.base.serving import GenerateBaseServing
from vllm.entrypoints.openai.responses.context import (
    HarmonyContext,
    ParsableContext,
    SimpleContext,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.envs import disable_envs_cache
from vllm.exceptions import GenerationError, RetryableRequestError
from vllm.outputs import CompletionOutput, RequestError, RequestOutput
from vllm.parser.harmony import HarmonyParser
from vllm.sampling_params import SamplingParams


def _retryable_output() -> RequestOutput:
    return RequestOutput(
        request_id="test-request-id",
        prompt=None,
        prompt_token_ids=[],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="",
                token_ids=(),
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="error",
            )
        ],
        finished=True,
        error=RequestError(
            code="multimodal_cache_miss",
            message="Multi-modal processor cache miss.",
            retryable=True,
        ),
    )


def _retryable_context(
    context_type: str, request: ResponsesRequest
) -> SimpleContext | ParsableContext | HarmonyContext:
    output = _retryable_output()
    if context_type == "simple":
        context = SimpleContext()
    elif context_type == "parsable":
        context = ParsableContext(
            response_messages=[],
            tokenizer=MagicMock(),
            parser_cls=None,
            request=request,
            available_tools=None,
            chat_template=None,
            chat_template_content_format="auto",
        )
    else:
        parser = MagicMock(spec=HarmonyParser)
        parser.process_chunk.return_value = SimpleNamespace(
            segments=[], reasoning_token_count=0
        )
        parser.flush.return_value = []
        context = HarmonyContext(
            messages=[],
            available_tools=[],
            function_tool_names=frozenset(),
            response_parser=parser,
            request=request,
        )
    context.append_output(output)
    return context


async def _empty_context_generator():
    if False:
        yield


async def _single_context_generator(context):
    yield context


def _background_stream_serving(request: ResponsesRequest):
    stored_response = SimpleNamespace(status="queued")
    serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
    serving.parser = None
    serving.use_harmony = False
    serving.event_store = {}
    serving.response_store = {request.request_id: stored_response}
    serving.response_store_lock = asyncio.Lock()
    return serving, stored_response


async def _collect_background_events(serving: OpenAIServingResponses, request_id: str):
    return [
        event
        async for event in serving.responses_background_stream_generator(request_id)
    ]


@pytest.mark.asyncio
async def test_raise_if_error_raises_generation_error():
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
    with pytest.raises(GenerationError) as exc_info:
        serving._raise_if_error("error", "test-request-id")

    assert str(exc_info.value) == "Internal server error"
    assert exc_info.value.status_code == HTTPStatus.INTERNAL_SERVER_ERROR

    # test that other finish_reasons don't raise
    serving._raise_if_error("stop", "test-request-id")  # should not raise
    serving._raise_if_error("length", "test-request-id")  # should not raise
    serving._raise_if_error(None, "test-request-id")  # should not raise


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_raise_if_error_maps_retryable_error_to_503():
    mock_engine = MagicMock()
    mock_engine.model_config = MagicMock()
    mock_engine.model_config.max_model_len = 100
    serving = GenerateBaseServing(
        engine_client=mock_engine,
        models=MagicMock(),
        request_logger=None,
    )
    error = _retryable_output().error
    assert error is not None

    with pytest.raises(RetryableRequestError) as exc_info:
        serving._raise_if_error("error", "test-request-id", error)

    assert exc_info.value.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    response = serving._convert_generation_error_to_streaming_response(exc_info.value)
    assert "ServiceUnavailableError" in response


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("context_type", ["simple", "parsable", "harmony"])
async def test_responses_full_propagates_retryable_error(context_type: str):
    request = ResponsesRequest(input="test", tools=[], stream=False)
    context = _retryable_context(context_type, request)
    serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
    serving.use_harmony = context_type == "harmony"

    with pytest.raises(RetryableRequestError, match="Multi-modal processor cache miss"):
        await serving.responses_full_generator(
            request=request,
            sampling_params=SamplingParams(max_tokens=1),
            result_generator=_empty_context_generator(),
            context=context,
            model_name="test-model",
            tokenizer=MagicMock(),
            request_metadata=RequestResponseMetadata(request_id=request.request_id),
        )


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_background_response_marks_retryable_error_failed():
    request = ResponsesRequest(
        input="test",
        tools=[],
        stream=False,
        store=True,
        background=True,
    )
    stored_response = SimpleNamespace(status="queued")
    serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
    serving.responses_full_generator = AsyncMock(
        side_effect=RetryableRequestError("Multi-modal processor cache miss.")
    )
    serving.response_store = {request.request_id: stored_response}
    serving.response_store_lock = asyncio.Lock()

    await serving._run_background_request(request)

    serving.responses_full_generator.assert_awaited_once_with(request)
    assert stored_response.status == "failed"


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_background_stream_terminates_after_retryable_error():
    request = ResponsesRequest(
        input="test",
        tools=[],
        stream=True,
        store=True,
        background=True,
    )
    context = _retryable_context("simple", request)
    serving, stored_response = _background_stream_serving(request)

    await serving._run_background_request_stream(
        request,
        SamplingParams(max_tokens=1),
        _single_context_generator(context),
        context,
        "test-model",
        MagicMock(),
        RequestResponseMetadata(request_id=request.request_id),
        0,
    )

    events = await asyncio.wait_for(
        _collect_background_events(serving, request.request_id), timeout=1
    )

    assert [event.type for event in events] == [
        "response.created",
        "response.in_progress",
        "error",
    ]
    assert events[-1].code == "ServiceUnavailableError"
    assert stored_response.status == "failed"


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
