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

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "8",
        "--enforce-eager",
        "--max-logprobs",
        "-1",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


def _logprob_entries(resp) -> list:
    """Return the top_logprobs list from the first generated position."""
    return resp.choices[0].logprobs.content[0].top_logprobs


@pytest.mark.asyncio
async def test_logprob_token_ids_returns_requested_ids(server):
    """With `logprob_token_ids=[a, b, c]`, the response surfaces the
    requested vocab ids in the per-position top_logprobs list, independent
    of where they rank in the model's natural top-k distribution.

    Internally the sampler reserves the first slot for the sampled token
    and fills the remaining slots with the requested ids, so when the
    sampled token isn't in the requested set, the last requested id may
    be displaced. We assert at least len(requested) - 1 requested ids
    appear when the sampled token differs from all requested ids.
    """
    # Pick three vocab ids that are unlikely to be the sampled token on a
    # plain "Hello" prompt. They should still appear in the response only
    # because we asked for them.
    requested_ids = [100, 1000, 5000]
    async with server.get_async_client() as client:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=len(requested_ids),
            extra_body={
                "logprob_token_ids": requested_ids,
                "return_tokens_as_token_ids": True,
            },
        )
        entries = _logprob_entries(resp)

        returned_token_ids: set[int] = set()
        for e in entries:
            assert e.token.startswith("token_id:"), (
                "expected return_tokens_as_token_ids=True to yield the "
                f"`token_id:<int>` form, got {e.token!r}"
            )
            returned_token_ids.add(int(e.token.removeprefix("token_id:")))

        requested_present = returned_token_ids.intersection(requested_ids)
        # If the sampled token also happens to be one of the requested ids,
        # all of them must appear. Otherwise the sampled token consumes one
        # slot and at least len-1 of the requested ids must appear.
        sampled_in_requested = bool(returned_token_ids - set(requested_ids)) is False
        expected_min = (
            len(requested_ids) if sampled_in_requested else len(requested_ids) - 1
        )
        assert len(requested_present) >= expected_min, (
            f"requested {requested_ids}, returned {sorted(returned_token_ids)}, "
            f"matched {sorted(requested_present)} (expected at least {expected_min})"
        )

        # All returned values should be finite floats. The exact numeric
        # range depends on the server-wide `--logprobs-mode`; default is
        # `raw_logprobs` (log_softmax), so values are <= 0.
        for e in entries:
            assert isinstance(e.logprob, float)
            assert not math.isnan(e.logprob)
            assert not math.isinf(e.logprob)


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
