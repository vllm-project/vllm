# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from fastapi import Request
from fastapi.responses import Response

from vllm.entrypoints.pooling.scoring.api_router import create_score, do_rerank
from vllm.entrypoints.pooling.scoring.protocol import RerankRequest, ScoreTextRequest
from vllm.entrypoints.pooling.scoring.serving import ServingScores
from vllm.entrypoints.pooling.typing import PoolingEngineInput, PoolingServeContext
from vllm.exceptions import GenerationError, RetryableRequestError, VLLMValidationError
from vllm.inputs import tokens_input
from vllm.outputs import PoolingOutput, PoolingRequestOutput, RequestError
from vllm.pooling_params import PoolingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
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
@pytest.mark.parametrize("n_queries", [1, 2], ids=["rerank", "score"])
@pytest.mark.parametrize(
    "scenario",
    [
        "normal",
        "query_output",
        "query_add",
        "doc_output",
        "doc_error",
        "double_abort",
        "double_rpc",
    ],
)
async def test_late_interaction_releases_queries_after_producers_stop(
    n_queries, scenario, monkeypatch
):
    """Exercise route cancellation, encoding and cache ownership without a model."""
    runner = LateInteractionRunner()
    reached, cleanup_entered, release = (asyncio.Event() for _ in range(3))
    all_docs_added = asyncio.Event()
    blocked = asyncio.Event()
    producers = []
    pending = set()
    external_aborts, rpcs = [], []
    send_interrupted = False
    handler_task = None
    error = VLLMValidationError("document rejected")

    class ControlledEngine:
        encode = AsyncLLM.encode
        log_requests = False
        is_tracing_enabled = AsyncMock(return_value=False)

        def __init__(self):
            self.output_processor = OutputProcessor(None, log_stats=False)
            self.engine_core = SimpleNamespace(abort_requests_async=self.send_abort)

        async def add_request(self, request_id, prompt, params, **kwargs):
            task = asyncio.current_task()
            assert task is not None
            producers.append(task)
            internal_id = f"{request_id}-internal"
            pending.add(internal_id)
            collector = RequestOutputCollector(params.output_kind, internal_id)
            self.output_processor.add_request(
                EngineCoreRequest(
                    request_id=internal_id,
                    external_req_id=request_id,
                    prompt_token_ids=prompt["prompt_token_ids"],
                    mm_features=None,
                    sampling_params=None,
                    pooling_params=params,
                    arrival_time=0,
                    lora_request=None,
                    cache_salt=None,
                    data_parallel_rank=None,
                ),
                prompt=None,
                queue=collector,
            )
            runner.register_request(internal_id, params)
            index = prompt["prompt_token_ids"][0]
            if index == n_queries + 1:
                all_docs_added.set()
            if scenario == "doc_error" and index == n_queries:
                await all_docs_added.wait()
                raise error
            hold_doc = index == n_queries + 1 and scenario in {
                "doc_output",
                "doc_error",
                "double_abort",
            }
            if hold_doc:
                reached.set()
                return collector

            output = runner.postprocess_pooler_output(
                [torch.eye(2)], [params], [internal_id], [True]
            )[0]
            pending.remove(internal_id)
            runner.on_requests_finished([internal_id])
            hold_query = index == n_queries - 1 and scenario in {
                "query_output",
                "query_add",
                "double_rpc",
            }
            if hold_query:
                reached.set()
                if scenario == "query_add":
                    await blocked.wait()
                return collector
            self.output_processor.process_outputs(
                [
                    EngineCoreOutput(
                        request_id=internal_id,
                        new_token_ids=[],
                        pooling_output=output,
                        finish_reason=FinishReason.STOP,
                    )
                ]
            )
            return collector

        async def abort(self, request_ids, internal=False):
            if not internal:
                assert all(task.done() for task in producers)
                external_aborts.append(request_ids)
            await AsyncLLM.abort(self, request_ids, internal)

        async def send_abort(self, request_ids):
            nonlocal send_interrupted
            if request_ids and scenario == "double_abort":
                # AsyncLLM.abort already removed the external ID mapping.
                assert not self.output_processor.external_req_ids
                cleanup_entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    send_interrupted = True
                    raise
            pending.difference_update(request_ids)
            runner.on_requests_finished(request_ids)

        async def collective_rpc(self, method, *, args, wait_for_inflight_batches):
            assert all(task.done() for task in producers)
            assert not pending
            assert method == "release_late_interaction_query_cache"
            assert wait_for_inflight_batches is True
            rpcs.append(args[0])
            if scenario == "double_rpc":
                cleanup_entered.set()
                await release.wait()
            runner.release_queries(args[0])

    request = (
        RerankRequest(model="model", query="q", documents=["d1", "d2"])
        if n_queries == 1
        else ScoreTextRequest(model="model", text_1=["q1", "q2"], text_2=["d1", "d2"])
    )
    ctx = _make_context()
    ctx.request, ctx.n_queries = request, n_queries
    ctx.engine_inputs = [
        {
            "prompts": tokens_input(prompt_token_ids=[i]),
            "params": PoolingParams(task="token_embed"),
            "lora_requests": None,
            "priorities": 0,
        }
        for i in range(n_queries + 2)
    ]
    serving = object.__new__(ServingScores)
    serving.enable_flash_late_interaction = True
    serving.io_processor = Mock()
    serving.model_config = SimpleNamespace(pooler_config=None)
    engine = ControlledEngine()
    serving.engine_client = engine
    monkeypatch.setattr(serving, "_preprocessing", AsyncMock())
    monkeypatch.setattr(
        serving,
        "_postprocessing_async",
        AsyncMock(return_value=Response()),
        raising=False,
    )
    monkeypatch.setattr(serving, "_log_inputs", Mock())

    async def init_context(*args, **kwargs):
        nonlocal handler_task
        handler_task = asyncio.current_task()
        return ctx

    monkeypatch.setattr(serving, "_init_ctx", init_context)

    async def receive():
        if scenario in {"normal", "doc_error"}:
            await blocked.wait()
        await reached.wait()
        return {"type": "http.disconnect"}

    raw_request = Request(
        {
            "type": "http",
            "headers": [],
            "app": SimpleNamespace(state=SimpleNamespace(serving_scores=serving)),
        },
        receive=receive,
    )
    ctx.raw_request = raw_request
    route = do_rerank if n_queries == 1 else create_score

    async def run_scenario():
        if scenario == "doc_error":
            with pytest.raises(VLLMValidationError) as exc:
                await route(request, raw_request)
            assert exc.value is error
        else:
            response = await route(request, raw_request)
            if scenario == "normal":
                assert response.status_code == 200
                assert len(ctx.final_res_batch) == 2
                assert all(
                    result.outputs.data.item() == 2 for result in ctx.final_res_batch
                )
            else:
                assert response is None
                assert handler_task is not None
                if scenario in {"double_abort", "double_rpc"}:
                    await cleanup_entered.wait()
                    handler_task.cancel()
                    await asyncio.sleep(0)
                    release.set()
                with pytest.raises(asyncio.CancelledError):
                    await handler_task

    await asyncio.wait_for(run_scenario(), timeout=5)
    assert not send_interrupted
    assert not pending
    assert not runner._query_cache
    assert not runner._query_uses
    assert not runner._doc_query_keys
    assert not engine.output_processor.external_req_ids
    if scenario == "normal":
        assert not external_aborts and not rpcs
    else:
        assert ctx.late_interaction_query_keys is not None
        assert external_aborts == [
            ctx.late_interaction_query_keys + (ctx.late_interaction_doc_keys or [])
        ]
        assert rpcs == [ctx.late_interaction_query_keys]
