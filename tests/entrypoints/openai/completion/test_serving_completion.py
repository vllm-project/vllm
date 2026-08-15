# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import AsyncIterator

import pytest

from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import (
    RequestResponseMetadata,
    StreamOptions,
)
from vllm.outputs import CompletionOutput, RequestOutput


def _make_serving() -> OpenAIServingCompletion:
    serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    serving.enable_prompt_tokens_details = True
    serving.enable_per_request_metrics = False
    serving.enable_force_include_usage = False
    serving.system_fingerprint = None
    return serving


def _make_request_output(
    num_cached_tokens: int | None,
    num_local_cached_tokens: int | None,
    num_external_cached_tokens: int | None,
    request_id: str = "req",
) -> RequestOutput:
    output = CompletionOutput(
        index=0,
        text="",
        token_ids=[],
        cumulative_logprob=0.0,
        logprobs=None,
        finish_reason="length",
    )
    return RequestOutput(
        request_id=request_id,
        prompt="hi",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[output],
        finished=True,
        num_cached_tokens=num_cached_tokens,
        num_local_cached_tokens=num_local_cached_tokens,
        num_external_cached_tokens=num_external_cached_tokens,
    )


def test_prompt_tokens_details_aggregates_across_prompts():
    """Cache stats are aggregated across all prompts of a multi-prompt
    Completion request (regression test for the #52199 review comment)."""
    serving = _make_serving()
    request = CompletionRequest(model="test", prompt="hi", max_tokens=1)
    metadata = RequestResponseMetadata(request_id="req", final_usage_info=None)

    batch = [
        _make_request_output(
            num_cached_tokens=3,
            num_local_cached_tokens=1,
            num_external_cached_tokens=2,
        ),
        _make_request_output(
            num_cached_tokens=5,
            num_local_cached_tokens=3,
            num_external_cached_tokens=2,
            request_id="req2",
        ),
        # A prompt without cache details must not break the aggregate.
        _make_request_output(
            num_cached_tokens=None,
            num_local_cached_tokens=None,
            num_external_cached_tokens=None,
            request_id="req3",
        ),
    ]

    response = serving.request_output_to_completion_response(
        batch,
        request,
        request_id="req",
        created_time=0,
        model_name="test",
        request_metadata=metadata,
    )

    assert response.usage.prompt_tokens == 9
    assert response.usage.prompt_tokens_details is not None
    assert response.usage.prompt_tokens_details.cached_tokens == 8
    assert response.usage.prompt_tokens_details.local_cached_tokens == 4
    assert response.usage.prompt_tokens_details.external_cached_tokens == 4


def test_prompt_tokens_details_omitted_without_cache_info():
    """Cache details are omitted when no prompt carries cache info."""
    serving = _make_serving()
    request = CompletionRequest(model="test", prompt="hi", max_tokens=1)
    metadata = RequestResponseMetadata(request_id="req", final_usage_info=None)

    batch = [
        _make_request_output(
            num_cached_tokens=None,
            num_local_cached_tokens=None,
            num_external_cached_tokens=None,
        )
    ]

    response = serving.request_output_to_completion_response(
        batch,
        request,
        request_id="req",
        created_time=0,
        model_name="test",
        request_metadata=metadata,
    )

    assert response.usage.prompt_tokens == 3
    assert response.usage.prompt_tokens_details is None


@pytest.mark.asyncio
async def test_streaming_aggregates_cache_once_per_prompt():
    """The streaming path aggregates cache stats once per prompt (each prompt
    stream yields many chunks carrying the same prefill value)."""
    serving = _make_serving()
    request = CompletionRequest(
        model="test",
        prompt=["hi", "hi"],
        max_tokens=1,
        stream=True,
        stream_options=StreamOptions(include_usage=True),
    )

    async def result_generator() -> AsyncIterator[tuple[int, RequestOutput]]:
        # Each prompt yields multiple chunks; the cache value must be counted
        # only once per prompt.
        for prompt_idx in range(2):
            for _ in range(3):
                yield (
                    prompt_idx,
                    _make_request_output(
                        num_cached_tokens=4 + prompt_idx,
                        num_local_cached_tokens=1 + prompt_idx,
                        num_external_cached_tokens=1,
                        request_id=f"req{prompt_idx}",
                    ),
                )

    metadata = RequestResponseMetadata(request_id="req", final_usage_info=None)
    chunks = []
    async for chunk in serving.completion_stream_generator(
        request,
        [None, None],
        result_generator(),
        request_id="req",
        created_time=0,
        model_name="test",
        num_prompts=2,
        tokenizer=None,
        request_metadata=metadata,
    ):
        if '"usage":' in chunk:
            chunks.append(chunk)

    assert len(chunks) == 1
    assert '"cached_tokens":9' in chunks[0]
    assert '"local_cached_tokens":3' in chunks[0]
    assert '"external_cached_tokens":2' in chunks[0]
