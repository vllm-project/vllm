# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Regression tests for https://github.com/vllm-project/vllm/issues/42119:
# two prompt_embeds-only requests with the same length and same cache_salt
# must not share LMCache external KV entries. The connector achieves this
# by mixing a digest of the prompt_embeds tensor into tracker.cache_salt,
# which then propagates into IPCCacheEngineKey on every store/retrieve.

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request

pytest.importorskip("lmcache")

# Imported after the importorskip so collection succeeds without lmcache.
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPRequestTracker,
)


def _make_request(
    request_id: str,
    *,
    prompt_embeds: torch.Tensor | None,
    prompt_token_ids: list[int] | None,
    cache_salt: str | None,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        prompt_embeds=prompt_embeds,
        cache_salt=cache_salt,
    )


def test_prompt_embeds_distinct_content_yields_distinct_salts():
    """Different prompt_embeds tensors of the same shape must produce
    different effective cache_salts even when the user-provided salt and
    sequence length are identical."""
    torch.manual_seed(0)
    embeds_a = torch.randn(32, 16, dtype=torch.float32)
    embeds_b = torch.randn(32, 16, dtype=torch.float32)

    req_a = _make_request(
        "req-a",
        prompt_embeds=embeds_a,
        prompt_token_ids=None,
        cache_salt="same",
    )
    req_b = _make_request(
        "req-b",
        prompt_embeds=embeds_b,
        prompt_token_ids=None,
        cache_salt="same",
    )

    tracker_a = LMCacheMPRequestTracker(req_a)
    tracker_b = LMCacheMPRequestTracker(req_b)

    # Both salts carry the user-provided prefix...
    assert tracker_a.cache_salt.startswith("same|pe:")
    assert tracker_b.cache_salt.startswith("same|pe:")
    # ...but the embed-derived suffix diverges, so external keys won't collide.
    assert tracker_a.cache_salt != tracker_b.cache_salt


def test_prompt_embeds_same_content_yields_same_salt():
    """Identical prompt_embeds with the same user salt must hash identically,
    so retrieval after a vLLM restart still works for the same input."""
    torch.manual_seed(1)
    embeds = torch.randn(8, 4, dtype=torch.float32)

    tracker_first = LMCacheMPRequestTracker(
        _make_request(
            "req-1", prompt_embeds=embeds, prompt_token_ids=None, cache_salt="s"
        )
    )
    tracker_second = LMCacheMPRequestTracker(
        _make_request(
            "req-2",
            prompt_embeds=embeds.clone(),
            prompt_token_ids=None,
            cache_salt="s",
        )
    )

    assert tracker_first.cache_salt == tracker_second.cache_salt


def test_token_only_request_salt_is_unchanged():
    """Token-only requests must keep the original cache_salt verbatim, so
    the existing token-id key path is unaffected."""
    req = _make_request(
        "req-tokens",
        prompt_embeds=None,
        prompt_token_ids=[1, 2, 3, 4],
        cache_salt="user-salt",
    )

    tracker = LMCacheMPRequestTracker(req)

    assert tracker.cache_salt == "user-salt"


def test_token_only_request_without_salt_is_empty_string():
    """Backwards compat: a request with no cache_salt and no prompt_embeds
    keeps the historical empty-string salt."""
    req = _make_request(
        "req-no-salt",
        prompt_embeds=None,
        prompt_token_ids=[1, 2, 3, 4],
        cache_salt=None,
    )

    tracker = LMCacheMPRequestTracker(req)

    assert tracker.cache_salt == ""


def test_prompt_embeds_with_no_user_salt_still_isolates():
    """Even without a user-provided salt, distinct embeddings must produce
    distinct effective salts."""
    torch.manual_seed(2)
    embeds_a = torch.randn(16, 8, dtype=torch.float32)
    embeds_b = torch.randn(16, 8, dtype=torch.float32)

    tracker_a = LMCacheMPRequestTracker(
        _make_request(
            "a", prompt_embeds=embeds_a, prompt_token_ids=None, cache_salt=None
        )
    )
    tracker_b = LMCacheMPRequestTracker(
        _make_request(
            "b", prompt_embeds=embeds_b, prompt_token_ids=None, cache_salt=None
        )
    )

    assert tracker_a.cache_salt != tracker_b.cache_salt
    assert tracker_a.cache_salt.startswith("|pe:")
    assert tracker_b.cache_salt.startswith("|pe:")
