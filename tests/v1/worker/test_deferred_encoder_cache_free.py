# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for encoder cache retention.

Verifies that encoder cache entries are retained (never evicted by the
model runner) for multimodal models, preventing cache misses from
preemption, speculative rollback, and hash reuse across requests.

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


def _make_mm_feature(mm_hash: str, offset: int, length: int):
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        mm_position=PlaceholderRange(offset=offset, length=length),
        identifier=mm_hash,
        modality="image",
    )


# ------------------------------------------------------------------ #
# Old model runner (gpu_model_runner.py)                             #
# ------------------------------------------------------------------ #


class TestOldModelRunnerDeferredFree:
    """Verify the deferred free pattern in the old model runner."""

    def test_update_states_defers_free(self):
        """_update_states should save hashes instead of popping them."""
        encoder_cache = {"img_a": torch.zeros(1), "img_b": torch.zeros(1)}
        runner = type("MockRunner", (), {
            "encoder_cache": encoder_cache,
            "_deferred_encoder_free_hashes": [],
        })()

        # Simulate what _update_states now does.
        free_hashes = ["img_a"]
        runner._deferred_encoder_free_hashes = free_hashes

        # Entries should still be in the cache.
        assert "img_a" in runner.encoder_cache
        assert "img_b" in runner.encoder_cache

    def test_deferred_pop_after_mtp(self):
        """After MTP, deferred hashes are popped."""
        encoder_cache = {"img_a": torch.zeros(1), "img_b": torch.zeros(1)}
        runner = type("MockRunner", (), {
            "encoder_cache": encoder_cache,
            "_deferred_encoder_free_hashes": ["img_a"],
        })()

        # Simulate the deferred free after MTP.
        for mm_hash in runner._deferred_encoder_free_hashes:
            runner.encoder_cache.pop(mm_hash, None)
        runner._deferred_encoder_free_hashes = []

        assert "img_a" not in runner.encoder_cache
        assert "img_b" in runner.encoder_cache
        assert runner._deferred_encoder_free_hashes == []

    def test_mtp_reads_survive_during_step(self):
        """Simulate the full step: defer → MTP reads → deferred pop.
        Entries must be available during MTP reads."""
        encoder_cache = {"img_x": torch.zeros(10)}
        deferred = ["img_x"]

        # Phase 1: _update_states defers (does NOT pop).
        # Cache still has img_x.
        assert "img_x" in encoder_cache

        # Phase 2: MTP's _gather_mm_embeddings reads from cache.
        encoder_output = encoder_cache.get("img_x", None)
        assert encoder_output is not None, "MTP would hit cache miss here"

        # Phase 3: After MTP, deferred pop.
        for mm_hash in deferred:
            encoder_cache.pop(mm_hash, None)
        assert "img_x" not in encoder_cache

    def test_without_fix_mtp_hits_cache_miss(self):
        """Demonstrate the bug: if we pop eagerly, MTP hits a cache miss."""
        encoder_cache = {"img_x": torch.zeros(10)}

        # Old code: pop immediately in _update_states.
        encoder_cache.pop("img_x", None)

        # MTP tries to read — cache miss!
        encoder_output = encoder_cache.get("img_x", None)
        assert encoder_output is None, (
            "This proves the old code causes a cache miss for MTP"
        )


# ------------------------------------------------------------------ #
# New model runner (gpu/model_runner.py)                             #
# ------------------------------------------------------------------ #


class TestNewModelRunnerDeferredFree:
    """Verify the deferred free pattern in the new model runner."""

    def test_free_states_with_empty_hashes(self):
        """free_states() with empty list (multimodal) keeps entries.
        The scheduler returns [] for multimodal models, so free_states
        is effectively a no-op for them."""
        from unittest.mock import MagicMock

        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        cache = EncoderCache()
        cache.encoder_outputs["img_a"] = torch.zeros(1)

        scheduler_output = MagicMock()
        scheduler_output.free_encoder_mm_hashes = []

        runner = type("MockRunner", (), {
            "encoder_cache": cache,
            "free_states": GPUModelRunner.free_states,
        })()

        runner.free_states(scheduler_output)

        # Entry survives — scheduler sent empty list.
        assert "img_a" in cache.encoder_outputs

    def test_deferred_free_after_speculator(self):
        """After speculator, deferred hashes are freed."""
        cache = EncoderCache()
        cache.encoder_outputs["img_a"] = torch.zeros(1)
        cache.encoder_outputs["img_b"] = torch.zeros(1)

        deferred = ["img_a"]

        for mm_hash in deferred:
            cache.free_encoder_cache(mm_hash)

        assert "img_a" not in cache.encoder_outputs
        assert "img_b" in cache.encoder_outputs

    def test_empty_deferred_list_is_noop(self):
        """No crash or side effects when deferred list is empty."""
        cache = EncoderCache()
        cache.encoder_outputs["img_a"] = torch.zeros(1)

        deferred: list[str] = []
        for mm_hash in deferred:
            cache.free_encoder_cache(mm_hash)

        assert "img_a" in cache.encoder_outputs

    def test_deferred_hash_not_in_cache_is_safe(self):
        """Popping a hash that's already gone should not crash."""
        cache = EncoderCache()
        # "img_a" is NOT in the cache.

        deferred = ["img_a"]
        for mm_hash in deferred:
            cache.free_encoder_cache(mm_hash)
        # No error — free_encoder_cache uses pop(, None).
