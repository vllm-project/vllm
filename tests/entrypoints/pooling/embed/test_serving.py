# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import AsyncMock, MagicMock, call

import pytest
from fastapi.responses import Response

from vllm.entrypoints.pooling.embed.protocol import (
    CohereEmbedRequest,
    EmbeddingCompletionRequest,
)
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding
from vllm.exceptions import RetryableRequestError
from vllm.outputs import RequestError


def _cache_miss_error() -> RequestError:
    return RequestError(
        code="multimodal_cache_miss",
        message="Multi-modal processor cache miss.",
        retryable=True,
    )


def _retryable_error(error: RequestError) -> RetryableRequestError:
    return RetryableRequestError(error.message, code=error.code)


def _mock_serving(monkeypatch, collect_side_effect):
    serving = object.__new__(ServingEmbedding)
    io_processor = MagicMock()
    ctx = MagicMock()
    expected = Response()
    collect_batch = AsyncMock(side_effect=collect_side_effect)

    monkeypatch.setattr(
        serving, "get_io_processor", MagicMock(return_value=io_processor)
    )
    monkeypatch.setattr(serving, "_init_ctx", AsyncMock(return_value=ctx))
    monkeypatch.setattr(serving, "_preprocessing", AsyncMock())
    monkeypatch.setattr(serving, "_prepare_generators", AsyncMock())
    monkeypatch.setattr(serving, "_collect_batch", collect_batch)
    monkeypatch.setattr(
        serving,
        "_postprocessing_async",
        AsyncMock(return_value=expected),
        raising=False,
    )
    return serving, io_processor, ctx, expected, collect_batch


@pytest.mark.asyncio
async def test_embedding_retries_cache_miss_without_mm_cache(monkeypatch):
    cache_miss = _cache_miss_error()
    retryable_error = _retryable_error(cache_miss)
    serving, io_processor, ctx, expected, collect_batch = _mock_serving(
        monkeypatch, [retryable_error, None]
    )
    request = EmbeddingCompletionRequest(model="test", input="hello")

    result = await serving(request)

    assert result is expected
    assert collect_batch.await_count == 2
    assert serving._preprocessing.await_args_list == [
        call(io_processor, ctx, skip_mm_cache=False),
        call(io_processor, ctx, skip_mm_cache=True),
    ]


@pytest.mark.asyncio
async def test_embedding_second_cache_miss_is_retryable_error(monkeypatch):
    cache_miss = _cache_miss_error()
    retryable_error = _retryable_error(cache_miss)
    serving, _, _, _, collect_batch = _mock_serving(
        monkeypatch, [retryable_error, retryable_error]
    )
    request = EmbeddingCompletionRequest(model="test", input="hello")

    with pytest.raises(RetryableRequestError) as exc_info:
        await serving(request)

    assert exc_info.value is retryable_error
    assert collect_batch.await_count == 2


@pytest.mark.asyncio
async def test_cohere_embedding_retries_cache_miss(monkeypatch):
    cache_miss = _cache_miss_error()
    retryable_error = _retryable_error(cache_miss)
    serving, _, _, expected, collect_batch = _mock_serving(
        monkeypatch, [retryable_error, None]
    )
    request = CohereEmbedRequest(model="test", texts=["hello"])

    result = await serving(request)

    assert result is expected
    assert collect_batch.await_count == 2


@pytest.mark.asyncio
async def test_embedding_retries_other_error_with_mm_cache(monkeypatch):
    retryable_error = RetryableRequestError(
        "Temporary request failure.", code="temporary_error"
    )
    serving, io_processor, ctx, expected, collect_batch = _mock_serving(
        monkeypatch, [retryable_error, None]
    )
    request = EmbeddingCompletionRequest(model="test", input="hello")

    result = await serving(request)

    assert result is expected
    assert collect_batch.await_count == 2
    assert serving._preprocessing.await_args_list == [
        call(io_processor, ctx, skip_mm_cache=False),
        call(io_processor, ctx, skip_mm_cache=False),
    ]


@pytest.mark.asyncio
async def test_collect_batch_maps_cache_miss_to_retryable_error_by_default():
    serving = object.__new__(ServingEmbedding)
    cache_miss = _cache_miss_error()
    result = MagicMock()
    result.error = cache_miss

    async def result_generator():
        yield 0, result

    ctx = MagicMock()
    ctx.engine_inputs = [MagicMock()]
    ctx.result_generator = result_generator()

    with pytest.raises(RetryableRequestError, match=cache_miss.message) as exc_info:
        await serving._collect_batch(ctx)

    assert exc_info.value.code == cache_miss.code


@pytest.mark.asyncio
async def test_embedding_retry_render_skips_mm_cache():
    serving = object.__new__(ServingEmbedding)
    render_request = {"skip_mm_cache": False}
    io_processor = MagicMock()
    io_processor.get_request_factory_online.return_value = [render_request]
    io_processor.render_async = AsyncMock(return_value={"prompts": {}})
    io_processor.maybe_pre_process_chunked = MagicMock()
    serving.io_processor = io_processor
    ctx = MagicMock()

    await serving._preprocessing(io_processor, ctx, skip_mm_cache=True)

    assert render_request["skip_mm_cache"] is True
    io_processor.render_async.assert_awaited_once_with(render_request)
    io_processor.maybe_pre_process_chunked.assert_called_once_with(ctx)
