# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Regression tests for https://github.com/vllm-project/vllm/issues/42119:
# two prompt_embeds-only requests with the same length and same cache_salt
# must not share LMCache external KV entries. The connector achieves this
# by mixing a digest of the prompt_embeds tensor into tracker.cache_salt,
# which then propagates into IPCCacheEngineKey on every store/retrieve.

import pytest
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
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
    prompt_embeds: torch.Tensor | None = None,
    prompt_token_ids: list[int] | None = None,
    cache_salt: str | None = None,
    mm_features: list[MultiModalFeatureSpec] | None = None,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        prompt_embeds=prompt_embeds,
        cache_salt=cache_salt,
        mm_features=mm_features,
    )


def _mm_feature(identifier: str, offset: int, length: int) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length),
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


def test_prompt_embeds_digest_is_memoized_across_trackers():
    """Recreating a tracker for the same Request (e.g. on preempt/resume)
    must reuse the cached digest rather than rehashing the tensor."""
    torch.manual_seed(3)
    embeds = torch.randn(16, 8, dtype=torch.float32)
    req = _make_request("req-memo", prompt_embeds=embeds, cache_salt="s")

    assert req._prompt_embeds_digest is None
    tracker_first = LMCacheMPRequestTracker(req)
    digest_after_first = req._prompt_embeds_digest
    assert digest_after_first is not None
    # Mutating the underlying tensor in-place after the first construction
    # must NOT change the cached digest. This is the load-bearing property
    # of the memoization: subsequent tracker constructions for the same
    # request reuse the original hash without touching the tensor again.
    embeds.fill_(0)
    tracker_second = LMCacheMPRequestTracker(req)
    assert req._prompt_embeds_digest == digest_after_first
    assert tracker_first.cache_salt == tracker_second.cache_salt


def test_mm_features_distinct_identifiers_yield_distinct_salts():
    """Two requests with the same text (same placeholder token IDs) but
    different image identifiers must produce distinct effective cache_salts.
    Reviewer-flagged gap: same length + same cache_salt + different MM
    inputs would otherwise collide in the external KV store."""
    feature_a = _mm_feature("img-hash-A", offset=2, length=336)
    feature_b = _mm_feature("img-hash-B", offset=2, length=336)

    tracker_a = LMCacheMPRequestTracker(
        _make_request(
            "a",
            prompt_token_ids=[0] * 512,
            cache_salt="same",
            mm_features=[feature_a],
        )
    )
    tracker_b = LMCacheMPRequestTracker(
        _make_request(
            "b",
            prompt_token_ids=[0] * 512,
            cache_salt="same",
            mm_features=[feature_b],
        )
    )

    assert tracker_a.cache_salt.startswith("same|mm:")
    assert tracker_b.cache_salt.startswith("same|mm:")
    assert tracker_a.cache_salt != tracker_b.cache_salt


def test_mm_features_identical_inputs_yield_identical_salts():
    """Same text + same images must collapse to the same salt so retrieval
    after restart still works."""
    feature = _mm_feature("img-hash", offset=2, length=336)

    tracker_first = LMCacheMPRequestTracker(
        _make_request(
            "1", prompt_token_ids=[0] * 512, cache_salt="s", mm_features=[feature]
        )
    )
    tracker_second = LMCacheMPRequestTracker(
        _make_request(
            "2",
            prompt_token_ids=[0] * 512,
            cache_salt="s",
            mm_features=[_mm_feature("img-hash", offset=2, length=336)],
        )
    )

    assert tracker_first.cache_salt == tracker_second.cache_salt


def test_mm_features_position_change_yields_distinct_salts():
    """Same MM identifier but different placeholder offsets must still
    produce distinct salts, mirroring local APC's per-block extra keys
    that include `(identifier, offset)`."""
    tracker_a = LMCacheMPRequestTracker(
        _make_request(
            "a",
            prompt_token_ids=[0] * 512,
            cache_salt="s",
            mm_features=[_mm_feature("img", offset=2, length=336)],
        )
    )
    tracker_b = LMCacheMPRequestTracker(
        _make_request(
            "b",
            prompt_token_ids=[0] * 512,
            cache_salt="s",
            mm_features=[_mm_feature("img", offset=64, length=336)],
        )
    )

    assert tracker_a.cache_salt != tracker_b.cache_salt


def test_prompt_embeds_and_mm_features_compose():
    """A request with both prompt_embeds and mm_features should fold both
    digests into the salt, ordered as `<base>|pe:<...>|mm:<...>`."""
    torch.manual_seed(4)
    embeds = torch.randn(8, 4, dtype=torch.float32)
    feature = _mm_feature("img", offset=0, length=8)

    tracker = LMCacheMPRequestTracker(
        _make_request(
            "both",
            prompt_embeds=embeds,
            cache_salt="user",
            mm_features=[feature],
        )
    )

    assert tracker.cache_salt.startswith("user|pe:")
    assert "|mm:" in tracker.cache_salt
