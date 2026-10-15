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
from vllm.multimodal.cache import MultiModalCacheMissError


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
    cache_miss = MultiModalCacheMissError(["missing-hash"])
    serving, io_processor, ctx, expected, collect_batch = _mock_serving(
        monkeypatch, [cache_miss, None]
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
    cache_miss = MultiModalCacheMissError(["missing-hash"])
    serving, _, _, _, collect_batch = _mock_serving(
        monkeypatch, [cache_miss, cache_miss]
    )
    request = EmbeddingCompletionRequest(model="test", input="hello")

    with pytest.raises(RetryableRequestError) as exc_info:
        await serving(request)

    assert isinstance(exc_info.value.__cause__, MultiModalCacheMissError)
    assert collect_batch.await_count == 2


@pytest.mark.asyncio
async def test_cohere_embedding_does_not_retry_cache_miss(monkeypatch):
    cache_miss = MultiModalCacheMissError(["missing-hash"])
    serving, _, _, _, collect_batch = _mock_serving(monkeypatch, [cache_miss])
    request = CohereEmbedRequest(model="test", texts=["hello"])

    with pytest.raises(RetryableRequestError) as exc_info:
        await serving(request)

    assert isinstance(exc_info.value.__cause__, MultiModalCacheMissError)
    assert collect_batch.await_count == 1


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
