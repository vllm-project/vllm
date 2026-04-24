# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Scheduler._mamba_block_aligned_split.

Covers the fix that prevents a permanent scheduling deadlock when
Mamba block alignment would truncate num_new_tokens to 0 in a middle
chunk.  Before the fix the request would be stuck forever; after the
fix the original (sub-block) value is kept so that the decoder can
make progress and eventually free encoder-cache space.
"""

import pytest
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_hash():
    init_none_hash(sha256)


def _make_request(
    num_prompt_tokens: int,
    num_computed_tokens: int = 0,
    block_size: int = BLOCK_SIZE,
) -> Request:
    """Create a minimal Request for testing _mamba_block_aligned_split."""
    _init_hash()
    block_hasher = get_request_block_hasher(block_size, sha256)
    sampling_params = SamplingParams(max_tokens=16)
    request = Request(
        request_id="test-req-0",
        prompt_token_ids=[0] * num_prompt_tokens,
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=block_hasher,
    )
    request.num_computed_tokens = num_computed_tokens
    return request


def _make_hybrid_scheduler(
    block_size: int = BLOCK_SIZE,
    max_num_batched_tokens: int = 8192,
    max_model_len: int = 32768,
    use_eagle: bool = False,
    mamba_cache_mode: str = "align",
) -> Scheduler:
    """Create a Scheduler whose kv_cache_config contains a MambaSpec group,
    so that need_mamba_block_aligned_split is True."""
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
        mamba_cache_mode=mamba_cache_mode,
    )
    num_blocks = 10000
    cache_config.num_gpu_blocks = num_blocks

    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attn_layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode=mamba_cache_mode,
                ),
            ),
        ],
    )

    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )

    # Directly set use_eagle flag since SpeculativeConfig("ngram") won't
    # trigger use_eagle() (only "eagle", "eagle3", "mtp" do).
    # This is a unit test — we control the flag directly.
    if use_eagle:
        scheduler.use_eagle = True

    return scheduler


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMambaBlockAlignedSplit:
    """Tests for Scheduler._mamba_block_aligned_split."""

    def test_already_aligned(self):
        """When num_new_tokens is already a multiple of block_size, it
        should be returned unchanged."""
        scheduler = _make_hybrid_scheduler()
        # num_prompt_tokens must be larger than num_new_tokens so that
        # num_computed_tokens_after_sched < last_cache_position (middle chunk).
        request = _make_request(num_prompt_tokens=20000, num_computed_tokens=0)

        result = scheduler._mamba_block_aligned_split(request, 8192)
        assert result == 8192  # 8192 = 512 * 16

    def test_normal_alignment_rounds_down(self):
        """Non-aligned values > block_size should be rounded down to the
        nearest multiple of block_size."""
        scheduler = _make_hybrid_scheduler()
        request = _make_request(num_prompt_tokens=1000, num_computed_tokens=0)

        result = scheduler._mamba_block_aligned_split(request, 100)
        # 100 // 16 * 16 = 96
        assert result == 96

    def test_normal_alignment_exact_block_size(self):
        """Exactly block_size tokens should pass through."""
        scheduler = _make_hybrid_scheduler()
        request = _make_request(num_prompt_tokens=1000, num_computed_tokens=0)

        result = scheduler._mamba_block_aligned_split(request, BLOCK_SIZE)
        assert result == BLOCK_SIZE

    @pytest.mark.parametrize("num_new_tokens", [1, 5, 8, 15])
    def test_sub_block_not_truncated_to_zero(self, num_new_tokens: int):
        """THE FIX: when num_new_tokens < block_size in a *middle* chunk,
        the original value should be kept (not truncated to 0).

        This is the deadlock scenario: encoder cache can't hold two images,
        the gap between two images is < block_size, and the old code would
        round it to 0, causing a permanent deadlock.
        """
        scheduler = _make_hybrid_scheduler()
        # Large prompt so we're definitely in a "middle chunk"
        # (num_computed_tokens_after_sched < last_cache_position).
        request = _make_request(
            num_prompt_tokens=20000,
            num_computed_tokens=8736,
        )

        result = scheduler._mamba_block_aligned_split(request, num_new_tokens)

        # Before the fix: result would be 0 (= num_new_tokens // 16 * 16)
        # After the fix: result should be num_new_tokens (kept as-is)
        assert result == num_new_tokens
        assert result > 0  # The critical invariant: never truncate to 0

    def test_sub_block_boundary_16_still_aligned(self):
        """When num_new_tokens == block_size exactly, it IS aligned (16 // 16 = 1)
        and should follow the normal path."""
        scheduler = _make_hybrid_scheduler()
        request = _make_request(
            num_prompt_tokens=20000,
            num_computed_tokens=8736,
        )

        result = scheduler._mamba_block_aligned_split(request, 16)
        assert result == 16  # 16 is a multiple of 16

    def test_sub_block_boundary_17_rounds_to_16(self):
        """17 tokens in a middle chunk should be rounded to 16 (not 0)."""
        scheduler = _make_hybrid_scheduler()
        request = _make_request(
            num_prompt_tokens=20000,
            num_computed_tokens=8736,
        )

        result = scheduler._mamba_block_aligned_split(request, 17)
        assert result == 16  # 17 // 16 * 16 = 16

    def test_last_chunk_no_alignment(self):
        """When the chunk reaches past last_cache_position, no alignment
        is applied (the 'prefill last few tokens' path)."""
        scheduler = _make_hybrid_scheduler()
        # num_tokens = 105, last_cache_position = 105 - 105%16 = 96
        # Without eagle: last_cache_position = 96
        request = _make_request(
            num_prompt_tokens=105,
            num_computed_tokens=100,
        )

        # 100 + 5 = 105 >= 96 (last_cache_position) → "last few tokens" → pass
        result = scheduler._mamba_block_aligned_split(request, 5)
        assert result == 5

    def test_force_last_cacheable_chunk(self):
        """When the chunk crosses last_cache_position, it should be forced
        to end exactly at last_cache_position."""
        scheduler = _make_hybrid_scheduler()
        # num_tokens = 105, last_cache_position = 96
        request = _make_request(
            num_prompt_tokens=105,
            num_computed_tokens=80,
        )

        # 80 < 96 < 80+25=105 → force: num_new_tokens = 96 - 80 = 16
        result = scheduler._mamba_block_aligned_split(request, 25)
        assert result == 16

    def test_decode_phase_skips_alignment(self):
        """During normal decoding (not prefill), alignment is skipped entirely."""
        scheduler = _make_hybrid_scheduler()
        request = _make_request(
            num_prompt_tokens=100,
            num_computed_tokens=100,
        )
        # num_computed_tokens (100) >= max(num_prompt_tokens(100), num_tokens-1(99))
        # So the outer 'if' is False → return as-is
        result = scheduler._mamba_block_aligned_split(request, 1)
        assert result == 1


class TestMambaBlockAlignedSplitWithEagle:
    """Tests with Eagle/MTP speculative decoding enabled."""

    def test_eagle_shifts_last_cache_position(self):
        """With Eagle, last_cache_position is reduced by one block_size."""
        scheduler = _make_hybrid_scheduler(use_eagle=True)
        # num_tokens = 105
        # Without eagle: last_cache_position = 96
        # With eagle:    last_cache_position = max(96 - 16, 0) = 80
        request = _make_request(
            num_prompt_tokens=105,
            num_computed_tokens=0,
        )

        # 0 + 100 = 100 > 80 (eagle last_cache_position)
        # 0 < 80 < 100 → force: num_new_tokens = 80 - 0 = 80
        result = scheduler._mamba_block_aligned_split(request, 100)
        assert result == 80

    @pytest.mark.parametrize("num_new_tokens", [1, 7, 15])
    def test_eagle_sub_block_not_truncated(self, num_new_tokens: int):
        """With Eagle, the sub-block fix should still apply."""
        scheduler = _make_hybrid_scheduler(use_eagle=True)
        request = _make_request(
            num_prompt_tokens=20000,
            num_computed_tokens=8736,
        )
        # With eagle: last_cache_position = large value - 16
        # Still a middle chunk, so alignment applies
        result = scheduler._mamba_block_aligned_split(request, num_new_tokens)
        assert result == num_new_tokens
        assert result > 0


class TestMambaBlockAlignedSplitDeadlockScenario:
    """End-to-end simulation of the exact deadlock scenario."""

    def test_deadlock_scenario_two_large_images(self):
        """Simulate the scenario that caused the original deadlock:

        - Two large images with tokens close together
        - Encoder cache can only hold one image
        - Gap between images < block_size

        Before the fix: _mamba_block_aligned_split returns 0 → deadlock
        After the fix: returns the sub-block gap → progress is made
        """
        scheduler = _make_hybrid_scheduler()

        # Simulate: prompt has 2 images, each 8720 tokens
        # Image 1: [30, 8750)
        # Gap: 2 tokens at [8750, 8752)
        # Image 2: [8752, 17472)
        # Total prompt: ~17500 tokens
        total_prompt_tokens = 17500

        # After step 1 (8192 tokens) and step 2 (544 tokens):
        # num_computed_tokens = 8736
        # Image 1 ends at 8750 → still 14 tokens short of freeing it
        # Gap to image 2 starts at 8752

        # The encoder scheduling would set num_new_tokens to:
        # start_pos_image2 - (num_computed_tokens + shift)
        # = 8752 - (8736 + 1) = 15  (with Eagle shift=1)
        # Without Eagle shift: = 8752 - 8736 = 16

        # Without Eagle (shift=0), gap = 16 → aligned to 16 → OK
        # With Eagle (shift=1), gap = 15 → DEADLOCK without the fix

        request = _make_request(
            num_prompt_tokens=total_prompt_tokens,
            num_computed_tokens=8736,
        )

        # Test gap = 15 (the deadlock value)
        result = scheduler._mamba_block_aligned_split(request, 15)
        assert result == 15  # After fix: kept as-is
        assert result > 0  # Not truncated to 0

        # Test gap = 14
        result = scheduler._mamba_block_aligned_split(request, 14)
        assert result == 14
        assert result > 0

        # Test gap = 1 (extreme case)
        result = scheduler._mamba_block_aligned_split(request, 1)
        assert result == 1
        assert result > 0

    def test_progress_chain_escapes_deadlock(self):
        """Verify that the fix allows the scheduler to make progress
        across multiple steps, eventually freeing the first image."""
        scheduler = _make_hybrid_scheduler()

        # Image 1: [30, 8750), Image 2: [8752, ...)
        # After initial chunked prefills: num_computed_tokens = 8736
        num_prompt = 17500
        request = _make_request(
            num_prompt_tokens=num_prompt,
            num_computed_tokens=8736,
        )

        # Step A: gap = 15 (simulating encoder scheduling with Eagle shift)
        result_a = scheduler._mamba_block_aligned_split(request, 15)
        assert result_a == 15  # Progress!

        # After step A: num_computed_tokens = 8736 + 15 = 8751
        request.num_computed_tokens = 8751

        # Image 1 ends at 8750. Now 8750 <= 8751 → Image 1 freed!
        # Image 2 can now be allocated.
        # Step B: normal large chunk
        result_b = scheduler._mamba_block_aligned_split(request, 8192)
        # 8751 + 8192 = 16943 < last_cache_position → align
        # 8192 // 16 * 16 = 8192 (already aligned)
        assert result_b == 8192  # Back to normal!

    def test_values_above_block_size_still_aligned(self):
        """Verify that values > block_size are still properly aligned
        (the fix only affects the aligned == 0 case)."""
        scheduler = _make_hybrid_scheduler()
        request = _make_request(
            num_prompt_tokens=20000,
            num_computed_tokens=8736,
        )

        # These should all be rounded DOWN to nearest block_size multiple
        assert scheduler._mamba_block_aligned_split(request, 31) == 16
        assert scheduler._mamba_block_aligned_split(request, 32) == 32
        assert scheduler._mamba_block_aligned_split(request, 33) == 32
        assert scheduler._mamba_block_aligned_split(request, 47) == 32
        assert scheduler._mamba_block_aligned_split(request, 48) == 48
        assert scheduler._mamba_block_aligned_split(request, 100) == 96
