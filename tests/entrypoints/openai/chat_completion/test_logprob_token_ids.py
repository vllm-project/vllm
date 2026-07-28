# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the `logprob_token_ids` field on the OpenAI-compat
chat-completion endpoint.

`logprob_token_ids` lets a caller pin the set of vocab ids whose logprobs
should appear in the response, independent of where those ids would rank in
the natural top-k distribution. This is the primitive that multilabel
scoring postprocessors use to gather logprobs at a fixed small label
vocabulary (e.g. PII detection where each label corresponds to a known
digit-token vocab id).
"""

import math

import pytest
from pydantic import ValidationError

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "8",
        "--enforce-eager",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def _logprob_entries(resp) -> list:
    """Return the top_logprobs list from the first generated position."""
    return resp.choices[0].logprobs.content[0].top_logprobs


def _token_id(token: str) -> int:
    assert token.startswith("token_id:"), (
        "expected return_tokens_as_token_ids=True to yield the "
        f"`token_id:<int>` form, got {token!r}"
    )
    return int(token.removeprefix("token_id:"))


def _top_logprob_token_ids(resp) -> list[int]:
    return [_token_id(entry.token) for entry in _logprob_entries(resp)]


def test_chat_request_decouples_top_k_from_explicit_token_ids():
    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Hello"}],
        logprobs=True,
        top_logprobs=5,
        logprob_token_ids=[5000],
    )

    sampling_params = request.to_sampling_params(
        max_tokens=1, default_sampling_params={}
    )

    assert sampling_params.logprobs is None
    assert sampling_params.logprob_token_ids == [5000]
    assert sampling_params.num_logprobs == 1


def test_completion_request_decouples_top_k_from_explicit_token_ids():
    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="Hello",
        logprobs=5,
        logprob_token_ids=[5000],
    )

    sampling_params = request.to_sampling_params(max_tokens=1)

    assert sampling_params.logprobs is None
    assert sampling_params.logprob_token_ids == [5000]
    assert sampling_params.num_logprobs == 1


def test_completion_rejects_explicit_token_ids_without_generated_tokens():
    with pytest.raises(ValidationError, match="no output tokens are generated"):
        CompletionRequest(
            model=MODEL_NAME,
            prompt="Hello",
            echo=True,
            max_tokens=0,
            logprobs=5,
            logprob_token_ids=[5000],
        )


def test_requests_reject_explicit_token_ids_with_beam_search():
    with pytest.raises(ValidationError, match="not supported with beam search"):
        ChatCompletionRequest(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            logprobs=True,
            logprob_token_ids=[5000],
            use_beam_search=True,
        )

    with pytest.raises(ValidationError, match="not supported with beam search"):
        CompletionRequest(
            model=MODEL_NAME,
            prompt="Hello",
            logprobs=5,
            logprob_token_ids=[5000],
            use_beam_search=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested_ids", "top_logprobs", "sampled_id"),
    [
        pytest.param([5000], 5, 42, id="original-issue-shape"),
        pytest.param([100, 1000, 5000], 3, 42, id="sampled-outside-set"),
        pytest.param([100, 1000, 5000], 3, 5000, id="sampled-inside-set"),
    ],
)
async def test_logprob_token_ids_returns_requested_ids(
    server, requested_ids, top_logprobs, sampled_id
):
    async with server.get_async_client() as client:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=top_logprobs,
            extra_body={
                "logprob_token_ids": requested_ids,
                "allowed_token_ids": [sampled_id],
                "return_tokens_as_token_ids": True,
            },
        )
        entries = _logprob_entries(resp)
        returned_token_ids = _top_logprob_token_ids(resp)

        assert _token_id(resp.choices[0].logprobs.content[0].token) == sampled_id
        assert set(returned_token_ids) == {*requested_ids, sampled_id}
        assert len(returned_token_ids) == len(set(returned_token_ids))

        for e in entries:
            assert isinstance(e.logprob, float)
            assert not math.isnan(e.logprob)
            assert not math.isinf(e.logprob)


@pytest.mark.asyncio
async def test_logprob_token_ids_stream_with_no_top_k(server):
    async with server.get_async_client() as client:
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=None,
            stream=True,
            extra_body={
                "logprob_token_ids": [5000],
                "allowed_token_ids": [42],
                "return_tokens_as_token_ids": True,
            },
        )

        returned_token_ids: list[int] = []
        async for chunk in stream:
            if not chunk.choices or chunk.choices[0].logprobs is None:
                continue
            for content in chunk.choices[0].logprobs.content:
                returned_token_ids.extend(
                    _token_id(entry.token) for entry in content.top_logprobs
                )

    assert set(returned_token_ids) == {42, 5000}


@pytest.mark.asyncio
async def test_logprob_token_ids_default_behavior_unchanged(server):
    """Without `logprob_token_ids`, the response carries the natural top-k
    most-likely tokens. This guards against the new field accidentally
    changing the default-path output."""
    async with server.get_async_client() as client:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=5,
        )
        entries = _logprob_entries(resp)
        assert len(entries) == 5
        # Default mode emits log_softmax, so all values are <= 0.
        for e in entries:
            assert e.logprob <= 0.0
