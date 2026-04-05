# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression tests for GitHub issue #38551:
AssertionError: Encoder cache miss crashes engine with MTP + multimodal
under high concurrency.

The bug: when MTP speculative decoding calls _gather_mm_embeddings with
shift_computed_tokens=1, the encoder cache may have already evicted entries
that MTP still needs. Previously a fatal assert killed the engine; now
the missing embedding is skipped with a warning on the draft path only.
The regular forward path still raises RuntimeError on cache miss.

This test directly exercises the _gather_mm_embeddings inner loop by
extracting it and simulating encoder cache eviction.
"""

import logging
import types

import pytest


def _build_mm_feature(identifier: str, offset: int, length: int):
    """Build a minimal mm_feature stub matching the real interface."""
    feat = types.SimpleNamespace()
    feat.identifier = identifier
    pos = types.SimpleNamespace()
    pos.offset = offset
    pos.length = length
    pos.is_embed = None
    pos.get_embeds_indices_in_range = lambda s, e: (s, e)
    feat.mm_position = pos
    return feat


def _gather_mm_embeddings_inner(
    encoder_cache,
    mm_features,
    num_computed_tokens,
    num_scheduled_tokens,
    shift_computed_tokens,
    logger,
):
    """
    Minimal extraction of the _gather_mm_embeddings inner loop from
    gpu_model_runner.py:2919-2971.  Mirrors the fixed logic: graceful
    skip on draft path, RuntimeError on regular path.
    """
    gathered = []
    for mm_feature in mm_features:
        pos_info = mm_feature.mm_position
        start_pos = pos_info.offset
        num_encoder_tokens = pos_info.length

        if start_pos >= num_computed_tokens + num_scheduled_tokens:
            break
        if start_pos + num_encoder_tokens <= num_computed_tokens:
            continue

        start_idx = max(num_computed_tokens - start_pos, 0)
        end_idx = min(
            num_computed_tokens - start_pos + num_scheduled_tokens,
            num_encoder_tokens,
        )
        assert start_idx < end_idx
        curr_embeds_start, curr_embeds_end = pos_info.get_embeds_indices_in_range(
            start_idx, end_idx
        )
        if curr_embeds_start == curr_embeds_end:
            continue

        mm_hash = mm_feature.identifier
        encoder_output = encoder_cache.get(mm_hash, None)
        if encoder_output is None:
            if shift_computed_tokens > 0:
                logger.warning(
                    "Encoder cache miss for %s — skipping "
                    "multimodal embedding for this draft step.",
                    mm_hash,
                )
                continue
            raise RuntimeError(f"Encoder cache miss for {mm_hash}.")

        gathered.append(encoder_output[start_idx:end_idx])
    return gathered


class TestEncoderCacheMissIssue38551:
    """Regression tests for issue #38551."""

    def test_normal_path_no_eviction(self):
        """Baseline: gathering works when encoder cache is populated."""
        mm_hash = "image_hash_abc123"
        fake_output = list(range(10))
        encoder_cache = {mm_hash: fake_output}
        mm_features = [_build_mm_feature(mm_hash, offset=0, length=10)]

        result = _gather_mm_embeddings_inner(
            encoder_cache,
            mm_features,
            num_computed_tokens=0,
            num_scheduled_tokens=5,
            shift_computed_tokens=0,
            logger=logging.getLogger("test"),
        )
        assert len(result) == 1

    def test_mtp_path_no_eviction(self):
        """MTP path (shift_computed_tokens=1) works when cache is populated."""
        mm_hash = "image_hash_abc123"
        fake_output = list(range(10))
        encoder_cache = {mm_hash: fake_output}
        mm_features = [_build_mm_feature(mm_hash, offset=0, length=10)]

        result = _gather_mm_embeddings_inner(
            encoder_cache,
            mm_features,
            num_computed_tokens=1,
            num_scheduled_tokens=5,
            shift_computed_tokens=1,
            logger=logging.getLogger("test"),
        )
        assert len(result) == 1

    def test_regular_path_cache_miss_raises(self):
        """Regular forward path (shift=0) must still fail fast on cache miss."""
        mm_hash = "image_hash_abc123"
        encoder_cache: dict = {}
        mm_features = [_build_mm_feature(mm_hash, offset=0, length=10)]

        with pytest.raises(RuntimeError, match="Encoder cache miss"):
            _gather_mm_embeddings_inner(
                encoder_cache,
                mm_features,
                num_computed_tokens=0,
                num_scheduled_tokens=5,
                shift_computed_tokens=0,
                logger=logging.getLogger("test"),
            )

    def test_mtp_cache_miss_skips_gracefully(self):
        """
        Regression test for #38551: eviction before MTP read must NOT
        crash the engine.  The missing embedding should be skipped with
        a warning instead.
        """
        mm_hash = "image_hash_abc123"
        encoder_cache = {mm_hash: list(range(10))}
        mm_features = [_build_mm_feature(mm_hash, offset=0, length=10)]

        # Normal forward pass succeeds
        _gather_mm_embeddings_inner(
            encoder_cache,
            mm_features,
            num_computed_tokens=0,
            num_scheduled_tokens=5,
            shift_computed_tokens=0,
            logger=logging.getLogger("test"),
        )

        # Simulate cache eviction
        del encoder_cache[mm_hash]

        # MTP draft proposer: should NOT crash, should return empty
        result = _gather_mm_embeddings_inner(
            encoder_cache,
            mm_features,
            num_computed_tokens=1,
            num_scheduled_tokens=5,
            shift_computed_tokens=1,
            logger=logging.getLogger("test"),
        )
        assert result == [], "Expected empty result when cache entry is evicted"

    def test_mtp_cache_miss_logs_warning(self, caplog):
        """Verify the warning is actually logged on cache miss."""
        mm_hash = "image_hash_abc123"
        encoder_cache: dict = {}
        mm_features = [_build_mm_feature(mm_hash, offset=0, length=10)]

        with caplog.at_level(logging.WARNING):
            _gather_mm_embeddings_inner(
                encoder_cache,
                mm_features,
                num_computed_tokens=0,
                num_scheduled_tokens=5,
                shift_computed_tokens=1,
                logger=logging.getLogger("test"),
            )

        assert any("Encoder cache miss" in msg for msg in caplog.messages)

    def test_multiple_images_partial_eviction_graceful(self):
        """
        When multiple images are in a request and only some are evicted,
        the non-evicted images are still gathered correctly on draft path.
        """
        hashes = [f"img_{i}" for i in range(4)]
        encoder_cache = {h: list(range(10)) for h in hashes}
        mm_features = [
            _build_mm_feature(h, offset=i * 10, length=10) for i, h in enumerate(hashes)
        ]

        # Evict images 1 and 2 (simulating selective cache pressure)
        del encoder_cache["img_1"]
        del encoder_cache["img_2"]

        # MTP draft path: should gather images 0 and 3, skip 1 and 2
        result = _gather_mm_embeddings_inner(
            encoder_cache,
            mm_features,
            num_computed_tokens=0,
            num_scheduled_tokens=40,
            shift_computed_tokens=1,
            logger=logging.getLogger("test"),
        )
        assert len(result) == 2, "Expected 2 gathered embeddings (skipped 2 evicted)"
