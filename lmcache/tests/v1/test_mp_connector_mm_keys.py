# SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal-aware cache-key token ids in the MP connector.

vLLM emits identical placeholder token ids for every image, so the MP
connector must overwrite placeholder spans with mm_hash-derived values
before token ids are used for key derivation (lookup, store, retrieve,
lock release). These tests exercise the public tracker/metadata
interfaces of ``lmcache_mp_connector``.
"""

# Standard
from dataclasses import dataclass

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.v1.utils import ConstantList  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)
from lmcache.integration.vllm.utils import hex_hash_to_int16  # noqa: E402

IMAGE_PLACEHOLDER_ID = 99


@dataclass
class _FakePlaceholder:
    offset: int
    length: int


@dataclass
class _FakeMMFeature:
    identifier: str
    mm_position: _FakePlaceholder


class _FakeRequest:
    """Duck-typed vLLM Request carrying only what the tracker reads."""

    def __init__(
        self,
        prompt_token_ids: list[int],
        mm_features: list[_FakeMMFeature] | None = None,
        cache_salt: str = "",
    ):
        self.request_id = "req-0"
        self.cache_salt = cache_salt
        self.prompt_token_ids = list(prompt_token_ids)
        self._live_token_ids = list(prompt_token_ids)
        self.all_token_ids = ConstantList(self._live_token_ids)
        self.mm_features = mm_features or []

    def append_decode_token(self, token_id: int):
        """Simulate vLLM appending a decode token to the live token list."""
        self._live_token_ids.append(token_id)


def _make_mm_request(
    prompt_token_ids: list[int],
    identifier: str,
    offset: int,
    length: int,
) -> _FakeRequest:
    mm_features = [_FakeMMFeature(identifier, _FakePlaceholder(offset, length))]
    return _FakeRequest(prompt_token_ids, mm_features=mm_features)


def test_text_only_request_uses_raw_token_ids():
    prompt = list(range(100, 108))
    tracker = LMCacheMPRequestTracker(_FakeRequest(prompt))
    assert tracker.get_token_ids() == prompt


def test_text_only_request_returns_mutable_copy():
    prompt = list(range(100, 108))
    tracker = LMCacheMPRequestTracker(_FakeRequest(prompt))
    token_ids = tracker.get_token_ids()
    token_ids[0] = -1
    assert tracker.get_token_ids() == prompt


def test_mm_request_overwrites_placeholder_span():
    prompt = [1, 2] + [IMAGE_PLACEHOLDER_ID] * 3 + [3, 4, 5]
    tracker = LMCacheMPRequestTracker(
        _make_mm_request(prompt, identifier="0xabcd", offset=2, length=3)
    )
    fill = hex_hash_to_int16("0xabcd")
    assert tracker.get_token_ids() == [1, 2, fill, fill, fill, 3, 4, 5]


def test_different_images_produce_different_key_tokens():
    prompt = [1, 2] + [IMAGE_PLACEHOLDER_ID] * 3 + [3, 4, 5]
    tracker_a = LMCacheMPRequestTracker(
        _make_mm_request(prompt, identifier="0xaaaa", offset=2, length=3)
    )
    tracker_b = LMCacheMPRequestTracker(
        _make_mm_request(prompt, identifier="0xbbbb", offset=2, length=3)
    )
    assert tracker_a.get_token_ids() != tracker_b.get_token_ids()


def test_decode_tokens_appended_unchanged():
    prompt = [1, 2] + [IMAGE_PLACEHOLDER_ID] * 2 + [3]
    request = _make_mm_request(prompt, identifier="0xabcd", offset=2, length=2)
    tracker = LMCacheMPRequestTracker(request)
    request.append_decode_token(500)
    request.append_decode_token(501)
    fill = hex_hash_to_int16("0xabcd")
    assert tracker.get_token_ids() == [1, 2, fill, fill, 3, 500, 501]


def _prepare_storable_tracker(request: _FakeRequest) -> LMCacheMPRequestTracker:
    """Give the tracker enough scheduled tokens and blocks to emit one
    store op covering the whole 8-token prompt (chunk size 4)."""
    tracker = LMCacheMPRequestTracker(request)
    tracker.allocated_block_ids = {0: [0, 1]}
    tracker.num_scheduled_tokens = 8
    return tracker


def test_store_metadata_uses_mm_adjusted_token_ids():
    prompt = [1, 2] + [IMAGE_PLACEHOLDER_ID] * 2 + [3, 4, 5, 6]
    request = _make_mm_request(prompt, identifier="0xabcd", offset=2, length=2)
    tracker = _prepare_storable_tracker(request)

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, lmcache_tokens_per_chunk=4, group_tokens_per_block=[4]
    )

    assert metadata is not None
    fill = hex_hash_to_int16("0xabcd")
    assert metadata.op.token_ids == [1, 2, fill, fill, 3, 4, 5, 6]
    assert metadata.op.start == 0
    assert metadata.op.end == 8


def test_retrieve_metadata_uses_mm_adjusted_token_ids():
    prompt = [1, 2] + [IMAGE_PLACEHOLDER_ID] * 2 + [3, 4, 5, 6]
    request = _make_mm_request(prompt, identifier="0xabcd", offset=2, length=2)
    tracker = LMCacheMPRequestTracker(request)
    tracker.allocated_block_ids = {0: [0, 1]}
    tracker.num_lmcache_hit_tokens = 8
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD

    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, lmcache_tokens_per_chunk=4, group_tokens_per_block=[4]
    )

    assert metadata is not None
    fill = hex_hash_to_int16("0xabcd")
    assert metadata.op.token_ids == [1, 2, fill, fill, 3, 4, 5, 6]
    assert metadata.op.start == 0
    assert metadata.op.end == 8
