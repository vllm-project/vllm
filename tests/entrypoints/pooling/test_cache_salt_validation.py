# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`cache_salt` must reject the empty string on the pooling endpoints too.

Every other protocol that accepts `cache_salt` pins ``min_length=1`` on it:
completions, chat completions, responses, the Anthropic messages API and the
scale-out protocol. The pooling mixin did not, and
``generate_block_hash_extra_keys`` tests the salt for truthiness rather than
for ``None``, so an empty salt produced no extra key at all and shared the
unsalted namespace instead of being rejected.

A client that emits `""` by mistake would therefore be silently sharing
prefix-cache blocks with every unsalted request, which is the failure this
parameter exists to prevent.
"""

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.pooling.embed.protocol import EmbeddingCompletionRequest

MODEL = "any-model"
PROMPT = "hello"


def test_pooling_rejects_empty_cache_salt():
    with pytest.raises(ValidationError):
        EmbeddingCompletionRequest(model=MODEL, input=PROMPT, cache_salt="")


def test_pooling_accepts_a_real_cache_salt():
    request = EmbeddingCompletionRequest(
        model=MODEL, input=PROMPT, cache_salt="tenant-a"
    )
    assert request.cache_salt == "tenant-a"


def test_pooling_allows_cache_salt_to_be_omitted():
    """Omission stays valid; only the empty string is rejected."""
    assert EmbeddingCompletionRequest(model=MODEL, input=PROMPT).cache_salt is None


def test_pooling_matches_completions_on_empty_cache_salt():
    """The two endpoints should agree, which is the point of the change."""
    with pytest.raises(ValidationError):
        CompletionRequest(model=MODEL, prompt=PROMPT, cache_salt="")
    with pytest.raises(ValidationError):
        EmbeddingCompletionRequest(model=MODEL, input=PROMPT, cache_salt="")
