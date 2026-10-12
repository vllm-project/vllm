# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import AsyncMock, Mock

import pytest

from vllm.entrypoints.pooling.scoring.serving import ServingScores
from vllm.entrypoints.pooling.typing import PoolingEngineInput, PoolingServeContext
from vllm.inputs import tokens_input
from vllm.pooling_params import PoolingParams


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
    prepare_generators = AsyncMock()
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
