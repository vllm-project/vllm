# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for deferred, reference-pinned eviction in EncoderCache.

The scheduler requests eviction of encoder outputs based on its own view of
request progress, which can be stale under async scheduling and speculative
decoding (rollback), and entries can be shared by concurrent requests with
identical content (same mm_hash). The cache must defer such evictions until
the last in-flight request referencing the entry is removed.

See: https://github.com/vllm-project/vllm/issues/38551
"""

import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

pytestmark = pytest.mark.cpu_test


def _make_feature(mm_hash: str, offset: int = 0, length: int = 4):
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        mm_position=PlaceholderRange(offset=offset, length=length),
        identifier=mm_hash,
        modality="image",
    )


def _make_cache(**kwargs) -> EncoderCache:
    return EncoderCache(**kwargs)


def _put(cache: EncoderCache, mm_hash: str) -> torch.Tensor:
    tensor = torch.zeros(1)
    cache.encoder_outputs[mm_hash] = tensor
    return tensor


def test_free_unreferenced_entry_is_eager():
    """Entries with no in-flight references are evicted immediately."""
    cache = _make_cache()
    _put(cache, "hash_a")

    cache.free_encoder_cache("hash_a")
    assert "hash_a" not in cache.encoder_outputs


def test_free_referenced_entry_is_deferred_until_request_removed():
    """Spec-decode rollback scenario: the scheduler frees an entry while the
    referencing request is still in flight; the entry must survive until the
    request is removed."""
    cache = _make_cache()
    cache.add_request("req_a", [_make_feature("hash_a")])
    _put(cache, "hash_a")

    cache.free_encoder_cache("hash_a")
    # Still readable by the in-flight request (e.g. drafter's shifted
    # window after rollback).
    assert "hash_a" in cache.encoder_outputs

    cache.remove_request("req_a")
    assert "hash_a" not in cache.encoder_outputs


def test_shared_hash_survives_first_request_completion():
    """Hash-reuse scenario: two concurrent requests share one image. The
    first request's completion must not evict the entry while the second
    still references it."""
    cache = _make_cache()
    cache.add_request("req_a", [_make_feature("hash_shared")])
    cache.add_request("req_b", [_make_feature("hash_shared")])
    _put(cache, "hash_shared")

    # Scheduler frees when req_a's reference is dropped.
    cache.free_encoder_cache("hash_shared")
    cache.remove_request("req_a")
    assert "hash_shared" in cache.encoder_outputs

    cache.remove_request("req_b")
    assert "hash_shared" not in cache.encoder_outputs


def test_readd_cancels_pending_eviction():
    """A new request reusing a pending-free hash revives the entry."""
    cache = _make_cache()
    cache.add_request("req_a", [_make_feature("hash_a")])
    _put(cache, "hash_a")
    cache.free_encoder_cache("hash_a")

    # New request reuses the same content before req_a finishes.
    cache.add_request("req_b", [_make_feature("hash_a")])
    cache.remove_request("req_a")
    # req_b's reference keeps the entry alive; the stale pending-free from
    # req_a's lifetime must not evict it.
    assert "hash_a" in cache.encoder_outputs

    # Entry now lives until the scheduler frees it again AND req_b is done.
    cache.free_encoder_cache("hash_a")
    assert "hash_a" in cache.encoder_outputs
    cache.remove_request("req_b")
    assert "hash_a" not in cache.encoder_outputs


def test_remove_without_pending_free_keeps_entry():
    """Request removal alone does not evict: the scheduler owns the
    eviction decision; the cache only defers it."""
    cache = _make_cache()
    cache.add_request("req_a", [_make_feature("hash_a")])
    _put(cache, "hash_a")

    cache.remove_request("req_a")
    assert "hash_a" in cache.encoder_outputs

    # Eviction happens once the scheduler asks for it.
    cache.free_encoder_cache("hash_a")
    assert "hash_a" not in cache.encoder_outputs


def test_eager_eviction_mode_for_encoder_decoder():
    """Encoder-decoder models (e.g. Whisper) evict immediately even while
    referenced, since the scheduler only frees provably-dead entries."""
    cache = _make_cache(eager_eviction=True)
    cache.add_request("req_a", [_make_feature("hash_a")])
    _put(cache, "hash_a")

    cache.free_encoder_cache("hash_a")
    assert "hash_a" not in cache.encoder_outputs


def test_update_request_keeps_refs_for_overlapping_hashes():
    """Streaming-session updates must not drop references to hashes that
    appear in both the old and new feature lists."""
    cache = _make_cache()
    cache.add_request("req_a", [_make_feature("hash_a"), _make_feature("hash_b")])
    _put(cache, "hash_a")
    _put(cache, "hash_b")

    # hash_a is kept, hash_b is dropped, hash_c is added.
    cache.free_encoder_cache("hash_b")
    cache.update_request("req_a", [_make_feature("hash_a"), _make_feature("hash_c")])

    # hash_b lost its last reference with a pending free -> evicted.
    assert "hash_b" not in cache.encoder_outputs

    # hash_a is still referenced; a scheduler free must defer.
    cache.free_encoder_cache("hash_a")
    assert "hash_a" in cache.encoder_outputs
    cache.remove_request("req_a")
    assert "hash_a" not in cache.encoder_outputs


def test_duplicate_features_in_one_request():
    """The same image appearing twice in one prompt holds a single
    per-request reference and is released on request removal."""
    cache = _make_cache()
    cache.add_request("req_a", [_make_feature("hash_a"), _make_feature("hash_a")])
    _put(cache, "hash_a")

    cache.free_encoder_cache("hash_a")
    assert "hash_a" in cache.encoder_outputs
    cache.remove_request("req_a")
    assert "hash_a" not in cache.encoder_outputs


def test_none_and_empty_mm_features_are_noops():
    cache = _make_cache()
    cache.add_request("req_text", [])
    cache.add_request("req_none", None)
    cache.remove_request("req_text")
    cache.remove_request("req_none")
    cache.remove_request("req_never_added")


def test_reset_encoder_cache_clears_outputs_and_pending():
    cache = _make_cache()
    cache.add_request("req_a", [_make_feature("hash_a")])
    _put(cache, "hash_a")
    cache.free_encoder_cache("hash_a")

    cache.reset_encoder_cache()
    assert not cache.encoder_outputs
    # Request removal after reset must not crash or resurrect state.
    cache.remove_request("req_a")
    assert not cache.encoder_outputs


def test_shared_external_dict_observes_evictions():
    """The legacy GPU model runner shares its encoder_cache dict with the
    tracker; evictions must be visible through the original reference."""
    external: dict[str, torch.Tensor] = {}
    cache = _make_cache(encoder_outputs=external)
    cache.add_request("req_a", [_make_feature("hash_a")])
    external["hash_a"] = torch.zeros(1)

    cache.free_encoder_cache("hash_a")
    assert "hash_a" in external
    cache.remove_request("req_a")
    assert "hash_a" not in external
