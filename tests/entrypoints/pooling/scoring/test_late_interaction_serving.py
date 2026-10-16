# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from vllm.entrypoints.pooling.scoring.serving import ServingScores
from vllm.entrypoints.pooling.typing import PoolingEngineInput, PoolingServeContext
from vllm.exceptions import GenerationError, RetryableRequestError
from vllm.inputs import tokens_input
from vllm.outputs import PoolingOutput, PoolingRequestOutput, RequestError
from vllm.pooling_params import LateInteractionParams, PoolingParams
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
        wait_for_inflight_batches=True,
    )


@pytest.mark.asyncio
async def test_colliding_request_ids_use_distinct_late_interaction_keys(
    monkeypatch: pytest.MonkeyPatch,
):
    serving = object.__new__(ServingScores)
    serving.engine_client = Mock(abort=AsyncMock(), collective_rpc=AsyncMock())
    collect_batch = AsyncMock()
    monkeypatch.setattr(serving, "_collect_late_interaction_batch", collect_batch)

    first = _make_context()
    second = _make_context()
    for context in (first, second):
        await serving._flash_late_interaction_encode_queries(context)
        await serving._flash_late_interaction_encode_docs(context)

    prepared_contexts = [call.args[0] for call in collect_batch.await_args_list]
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
@pytest.mark.parametrize("retryable", [False, True])
async def test_late_interaction_output_error_stops_pending_producers(
    retryable, monkeypatch
):
    """An error output must stop the batch even while another request is pending."""
    started, stopped, blocked = (asyncio.Event() for _ in range(3))

    async def failing():
        await started.wait()
        yield PoolingRequestOutput(
            request_id="failed",
            outputs=PoolingOutput(torch.empty(0)),
            prompt_token_ids=[1],
            num_cached_tokens=0,
            finished=True,
            error=RequestError("failed", "scoring failed", retryable=retryable),
        )

    async def pending():
        try:
            started.set()
            await blocked.wait()
            yield  # pragma: no cover
        finally:
            stopped.set()

    serving = object.__new__(ServingScores)
    monkeypatch.setattr(
        serving, "_prepare_generators", AsyncMock(return_value=[failing(), pending()])
    )
    error_type = RetryableRequestError if retryable else GenerationError
    with pytest.raises(error_type, match="scoring failed"):
        await asyncio.wait_for(
            serving._collect_late_interaction_batch(_make_context()), timeout=5
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
            late_interaction_params=LateInteractionParams(
                mode="cache_query", query_key="query", query_uses=1
            )
        )
        runner.postprocess_pooler_output([torch.eye(2)], [params], ["query"], [True])

    async def encode_docs(ctx):
        # The RPC case stops before any document can consume the cached query.
        if cleanup_stage == "abort":
            ctx.late_interaction_doc_keys = ["doc"]
            pending.add("doc")
            runner.register_request(
                "doc",
                PoolingParams(
                    late_interaction_params=LateInteractionParams(
                        mode="score_doc", query_key="query"
                    )
                ),
            )
        started.set()
        await asyncio.Event().wait()

    async def abort(request_ids):
        if cleanup_stage == "abort":
            cleanup_entered.set()
            await release.wait()
        pending.difference_update(request_ids)
        runner.on_requests_finished(request_ids)

    async def collective_rpc(method, *, args, wait_for_inflight_batches=False):
        if cleanup_stage == "rpc":
            cleanup_entered.set()
            await release.wait()
        runner.release_queries(args[0])

    serving = object.__new__(ServingScores)
    serving.io_processor = Mock()
    serving.engine_client = Mock(abort=abort, collective_rpc=collective_rpc)
    monkeypatch.setattr(serving, "_init_ctx", AsyncMock(return_value=ctx))
    monkeypatch.setattr(serving, "_preprocessing", AsyncMock())
    monkeypatch.setattr(
        serving, "_flash_late_interaction_encode_queries", encode_queries
    )
    monkeypatch.setattr(serving, "_flash_late_interaction_encode_docs", encode_docs)
    handler = asyncio.create_task(serving.flash_late_interaction())

    async def cancel_twice():
        await started.wait()
        handler.cancel()
        await cleanup_entered.wait()
        handler.cancel()
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait([handler])
        with pytest.raises(asyncio.CancelledError):
            handler.result()
        assert not pending
        assert not runner._query_cache
        assert not runner._query_uses
        assert not runner._doc_query_keys

    try:
        await asyncio.wait_for(cancel_twice(), timeout=5)
    finally:
        # Always unblock cleanup, including when an assertion or timeout fails.
        release.set()
        if not handler.done():
            handler.cancel()
        await asyncio.gather(handler, return_exceptions=True)
