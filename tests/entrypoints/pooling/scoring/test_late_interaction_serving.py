# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from vllm.entrypoints.pooling.scoring.serving import ServingScores
from vllm.entrypoints.pooling.typing import PoolingEngineInput, PoolingServeContext
from vllm.exceptions import GenerationError
from vllm.inputs import tokens_input
from vllm.outputs import PoolingOutput, PoolingRequestOutput, RequestError
from vllm.pooling_params import LateInteractionParams, PoolingParams
from vllm.utils.async_utils import merge_async_iterators
from vllm.v1.pool.late_interaction_runner import LateInteractionRunner


def _make_context() -> PoolingServeContext:
    return PoolingServeContext(
        request=Mock(),
        model_name="model",
        request_id="score-caller-controlled-id",
        pooling_params=PoolingParams(task="token_embed"),
        lora_request=None,
        priorities=None,
        prompt_extras=None,
        engine_inputs=[
            PoolingEngineInput(
                prompts=tokens_input(prompt_token_ids=[1]),
                params=PoolingParams(task="token_embed"),
                lora_requests=None,
                priorities=0,
            ),
            PoolingEngineInput(
                prompts=tokens_input(prompt_token_ids=[2]),
                params=PoolingParams(task="token_embed"),
                lora_requests=None,
                priorities=0,
            ),
        ],
        n_queries=1,
    )


def _query_key(context: PoolingServeContext) -> str:
    assert context.engine_inputs is not None
    pooling_params = context.engine_inputs[0]["params"]
    late_interaction_params = pooling_params.late_interaction_params
    assert late_interaction_params is not None
    query_key = late_interaction_params.query_key
    assert query_key is not None
    return query_key


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("failed_stage", ["query", "doc"])
async def test_flash_late_interaction_cleans_up_queries_on_failure(
    failed_stage: str,
    monkeypatch: pytest.MonkeyPatch,
):
    serving = object.__new__(ServingScores)
    serving.io_processor = Mock()
    serving.engine_client = Mock(abort=AsyncMock(), collective_rpc=AsyncMock())

    context = _make_context()
    monkeypatch.setattr(serving, "_init_ctx", AsyncMock(return_value=context))
    monkeypatch.setattr(serving, "_preprocessing", AsyncMock())

    async def encode_queries(ctx):
        ctx.late_interaction_query_keys = ["query-key"]
        if failed_stage == "query":
            raise RuntimeError("query failed")

    async def encode_docs(ctx):
        ctx.late_interaction_doc_keys = ["doc-key"]
        if failed_stage == "doc":
            raise RuntimeError("doc failed")

    monkeypatch.setattr(
        serving,
        "_flash_late_interaction_encode_queries",
        AsyncMock(side_effect=encode_queries),
    )
    monkeypatch.setattr(
        serving,
        "_flash_late_interaction_encode_docs",
        AsyncMock(side_effect=encode_docs),
    )

    with pytest.raises(RuntimeError, match=f"{failed_stage} failed"):
        await serving.flash_late_interaction()

    expected_doc_keys = [] if failed_stage == "query" else ["doc-key"]
    serving.engine_client.abort.assert_awaited_once_with(
        ["query-key", *expected_doc_keys]
    )
    serving.engine_client.collective_rpc.assert_awaited_once_with(
        "release_late_interaction_query_cache",
        args=(["query-key"],),
    )


@pytest.mark.asyncio
async def test_colliding_request_ids_use_distinct_late_interaction_keys(
    monkeypatch: pytest.MonkeyPatch,
):
    serving = object.__new__(ServingScores)
    serving.engine_client = Mock(abort=AsyncMock(), collective_rpc=AsyncMock())
    prepare_generators = AsyncMock(return_value=[])
    monkeypatch.setattr(serving, "_prepare_generators", prepare_generators)
    monkeypatch.setattr(serving, "_collect_batch", AsyncMock())

    first = _make_context()
    second = _make_context()
    for context in (first, second):
        await serving._flash_late_interaction_encode_queries(context)
        await serving._flash_late_interaction_encode_docs(context)

    prepared_contexts = [call.args[0] for call in prepare_generators.await_args_list]
    first_query_key = _query_key(prepared_contexts[0])
    first_doc_key = _query_key(prepared_contexts[1])
    second_query_key = _query_key(prepared_contexts[2])
    second_doc_key = _query_key(prepared_contexts[3])
    first_doc_request_ids = prepared_contexts[1].prompt_request_ids
    second_doc_request_ids = prepared_contexts[3].prompt_request_ids

    assert first_query_key == first_doc_key
    assert second_query_key == second_doc_key
    assert first_query_key != second_query_key
    assert "caller-controlled-id" not in first_query_key
    assert "caller-controlled-id" not in second_query_key
    assert first_doc_request_ids is not None
    assert second_doc_request_ids is not None
    assert first.late_interaction_doc_keys == first_doc_request_ids
    assert second.late_interaction_doc_keys == second_doc_request_ids
    assert first_doc_request_ids != second_doc_request_ids
    assert all("caller-controlled-id" not in key for key in first_doc_request_ids)
    assert all("caller-controlled-id" not in key for key in second_doc_request_ids)

    assert first.late_interaction_query_keys is not None
    await serving._cleanup_flash_late_interaction(first)
    serving.engine_client.abort.assert_awaited_once_with(
        [*first.late_interaction_query_keys, *first_doc_request_ids]
    )


@pytest.mark.asyncio
async def test_late_interaction_output_error_stops_pending_producers(monkeypatch):
    """An error output must stop the batch even while another request is pending."""
    started, stopped = asyncio.Event(), asyncio.Event()

    async def failing():
        await started.wait()
        output = PoolingOutput(torch.empty(0))
        error = RequestError("failed", "scoring failed")
        yield PoolingRequestOutput("failed", output, [1], 0, True, error=error)

    async def pending():
        try:
            started.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover
        finally:
            stopped.set()

    async def prepare(ctx):
        generators = [failing(), pending()]
        ctx.result_generator = merge_async_iterators(*generators)
        return generators

    serving = object.__new__(ServingScores)
    monkeypatch.setattr(serving, "_prepare_generators", prepare)
    ctx = _make_context()
    ctx.n_queries = 2
    assert ctx.engine_inputs is not None
    ctx.engine_inputs = list(ctx.engine_inputs) * 2
    with pytest.raises(GenerationError, match="scoring failed"):
        await asyncio.wait_for(
            serving._flash_late_interaction_encode_queries(ctx), timeout=5
        )
    assert stopped.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_stage", ["abort", "rpc"])
async def test_repeated_cancellation_releases_requests_and_queries(
    cleanup_stage, monkeypatch
):
    runner = LateInteractionRunner(enable_flash=False)
    pending = set()
    started, cleanup_entered, release = (asyncio.Event() for _ in range(3))
    ctx = _make_context()

    async def encode_queries(ctx):
        ctx.late_interaction_query_keys = ["query"]
        params = PoolingParams(
            late_interaction_params=LateInteractionParams("cache_query", "query", 1)
        )
        runner.postprocess_pooler_output([torch.eye(2)], [params], ["query"], [True])

    async def encode_docs(ctx):
        # The document is queued but has not reached the worker yet.
        ctx.late_interaction_doc_keys = ["doc"]
        pending.add("doc")
        started.set()
        await asyncio.Event().wait()

    async def cleanup(request_ids, *, args=None):
        is_abort = args is None
        if is_abort == (cleanup_stage == "abort"):
            cleanup_entered.set()
            await release.wait()
        if is_abort:
            pending.difference_update(request_ids)
        else:
            runner.release_queries(args[0])

    serving = object.__new__(ServingScores)
    serving.io_processor = Mock()
    serving.engine_client = Mock(abort=cleanup, collective_rpc=cleanup)
    monkeypatch.setattr(serving, "_init_ctx", AsyncMock(return_value=ctx))
    monkeypatch.setattr(serving, "_preprocessing", AsyncMock())
    monkeypatch.setattr(
        serving, "_flash_late_interaction_encode_queries", encode_queries
    )
    monkeypatch.setattr(serving, "_flash_late_interaction_encode_docs", encode_docs)
    handler = asyncio.create_task(serving.flash_late_interaction())

    try:
        async with asyncio.timeout(5):
            await started.wait()
            handler.cancel()
            await cleanup_entered.wait()
            handler.cancel()
            asyncio.get_running_loop().call_soon(release.set)
            await asyncio.wait([handler])
            with pytest.raises(asyncio.CancelledError):
                handler.result()
            assert not pending
            assert not runner._query_cache
            assert not runner._query_uses
    finally:
        # Always unblock cleanup, including when an assertion or timeout fails.
        release.set()
        handler.cancel()
        await asyncio.gather(handler, return_exceptions=True)
