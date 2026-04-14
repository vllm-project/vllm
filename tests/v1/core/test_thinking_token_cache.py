# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for thinking token prefix cache eviction.

When serving reasoning models with --reasoning-parser, thinking tokens
(<think>...</think>) should be immediately evicted from the prefix cache
on request completion since they are unreachable in multi-turn prompts.

See: https://github.com/vllm-project/vllm/issues/39321
"""

import pytest

from vllm.v1.core.kv_cache_utils import (
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)
from vllm.v1.request import Request


# ---------------------------------------------------------------------------
# FreeKVCacheBlockQueue.prepend_n tests
# ---------------------------------------------------------------------------


class TestPrependN:
    """Test that prepend_n inserts blocks at the head of the free queue."""

    def test_prepend_n_basic(self):
        """Prepended blocks should be popped first by popleft."""
        blocks = [KVCacheBlock(block_id=i) for i in range(5)]
        queue = FreeKVCacheBlockQueue(blocks)

        # Pop all original blocks
        original = [queue.popleft() for _ in range(5)]
        assert [b.block_id for b in original] == [0, 1, 2, 3, 4]

        # Now append some blocks to the tail (normal path)
        tail_blocks = [KVCacheBlock(block_id=10), KVCacheBlock(block_id=11)]
        queue.append_n(tail_blocks)

        # Prepend some blocks to the head
        head_blocks = [KVCacheBlock(block_id=20), KVCacheBlock(block_id=21)]
        queue.prepend_n(head_blocks)

        assert queue.num_free_blocks == 4

        # Head blocks should come out first
        popped = [queue.popleft() for _ in range(4)]
        assert [b.block_id for b in popped] == [20, 21, 10, 11]

    def test_prepend_n_empty_list(self):
        """Prepending empty list should be a no-op."""
        blocks = [KVCacheBlock(block_id=i) for i in range(3)]
        queue = FreeKVCacheBlockQueue(blocks)
        queue.prepend_n([])
        assert queue.num_free_blocks == 3
        # First block should still be block 0
        assert queue.popleft().block_id == 0

    def test_prepend_n_to_empty_queue(self):
        """Prepending to an empty queue should work."""
        queue = FreeKVCacheBlockQueue([])
        assert queue.num_free_blocks == 0

        new_blocks = [KVCacheBlock(block_id=5), KVCacheBlock(block_id=6)]
        queue.prepend_n(new_blocks)
        assert queue.num_free_blocks == 2
        assert queue.popleft().block_id == 5
        assert queue.popleft().block_id == 6

    def test_prepend_then_append_ordering(self):
        """Verify ordering: prepended < existing < appended."""
        queue = FreeKVCacheBlockQueue([KVCacheBlock(block_id=1)])

        queue.prepend_n([KVCacheBlock(block_id=0)])
        queue.append_n([KVCacheBlock(block_id=2)])

        popped = [queue.popleft() for _ in range(3)]
        assert [b.block_id for b in popped] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Request.update_reasoning_token_count tests
# ---------------------------------------------------------------------------


class TestReasoningTokenCount:
    """Test reasoning token counting on the Request object."""

    def _make_request(self, output_token_ids: list[int]) -> Request:
        """Create a minimal Request with the given output tokens."""
        from vllm.sampling_params import SamplingParams

        req = Request(
            request_id="test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=100),
            pooling_params=None,
        )
        req.append_output_token_ids(output_token_ids)
        return req

    def test_simple_thinking_block(self):
        """Count tokens in <think>...</think> and return first index."""
        # <think>=50, </think>=51, normal tokens are anything else
        req = self._make_request([50, 100, 101, 102, 51, 200, 201])
        first_idx = req.update_reasoning_token_count(
            start_token_id=50, end_token_id=51)
        # <think>(50) + 100 + 101 + 102 + </think>(51) = 5 tokens
        assert req.num_reasoning_tokens == 5
        assert first_idx == 0

    def test_no_thinking_tokens(self):
        """No thinking markers -> 0 reasoning tokens, None index."""
        req = self._make_request([200, 201, 202])
        first_idx = req.update_reasoning_token_count(
            start_token_id=50, end_token_id=51)
        assert req.num_reasoning_tokens == 0
        assert first_idx is None

    def test_all_thinking(self):
        """Entire output is thinking."""
        req = self._make_request([50, 100, 101, 51])
        first_idx = req.update_reasoning_token_count(
            start_token_id=50, end_token_id=51)
        assert req.num_reasoning_tokens == 4
        assert first_idx == 0

    def test_nested_thinking(self):
        """Nested thinking blocks should be handled by depth counter."""
        req = self._make_request([50, 100, 50, 101, 51, 102, 51, 200])
        first_idx = req.update_reasoning_token_count(
            start_token_id=50, end_token_id=51)
        # All of [50, 100, 50, 101, 51, 102, 51] = 7 tokens are thinking
        assert req.num_reasoning_tokens == 7
        assert first_idx == 0

    def test_empty_output(self):
        """Empty output -> 0 reasoning tokens, None index."""
        req = self._make_request([])
        first_idx = req.update_reasoning_token_count(
            start_token_id=50, end_token_id=51)
        assert req.num_reasoning_tokens == 0
        assert first_idx is None

    def test_thinking_starts_mid_output(self):
        """Thinking tokens starting after some answer tokens."""
        # Output: [200, 201, <think>, 100, 101, </think>]
        req = self._make_request([200, 201, 50, 100, 101, 51])
        first_idx = req.update_reasoning_token_count(
            start_token_id=50, end_token_id=51)
        # 4 reasoning tokens: <think>(50), 100, 101, </think>(51)
        assert req.num_reasoning_tokens == 4
        # First reasoning token is at output index 2
        assert first_idx == 2

    def test_unmatched_end_token(self):
        """End token without start should be ignored."""
        req = self._make_request([51, 200, 201])
        req.update_reasoning_token_count(start_token_id=50, end_token_id=51)
        assert req.num_reasoning_tokens == 0


# ---------------------------------------------------------------------------
# Integration: thinking blocks evicted before normal blocks
# ---------------------------------------------------------------------------


class TestThinkingBlockEviction:
    """Test that thinking blocks are evicted before normal cached blocks."""

    def _make_block(self, block_id: int, ref_cnt: int = 1,
                    block_hash: int | None = None) -> KVCacheBlock:
        block = KVCacheBlock(block_id=block_id)
        block.ref_cnt = ref_cnt
        if block_hash is not None:
            from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
            block.block_hash = make_block_hash_with_group_id(
                (block_hash, ()), 0
            )
        return block

    def test_free_blocks_immediate_evict_prepends_to_head(self):
        """Blocks freed via immediate_evict should appear before
        blocks freed via normal free_blocks."""
        from vllm.v1.core.block_pool import BlockPool

        num_blocks = 20
        pool = BlockPool(
            num_gpu_blocks=num_blocks,
            enable_caching=True,
            hash_block_size=16,
            enable_kv_cache_events=False,
        )

        # Allocate some blocks
        normal_blocks = pool.get_new_blocks(2)
        thinking_blocks = pool.get_new_blocks(2)

        # Give them hashes so they're "cached"
        for i, blk in enumerate(normal_blocks):
            blk.block_hash = (i, (), 0)
            pool.cached_block_hash_to_block.insert((i, (), 0), blk)
        for i, blk in enumerate(thinking_blocks):
            h = (100 + i, (), 0)
            blk.block_hash = h
            pool.cached_block_hash_to_block.insert(h, blk)

        # Free normal blocks (go to tail)
        pool.free_blocks(normal_blocks)
        # Free thinking blocks with immediate evict (go to head)
        pool.free_blocks_immediate_evict(thinking_blocks)

        # Now allocate 2 blocks - should get the thinking blocks first
        # (they were prepended to head)
        new_blocks = pool.get_new_blocks(2)
        new_ids = {b.block_id for b in new_blocks}
        thinking_ids = {b.block_id for b in thinking_blocks}
        assert new_ids == thinking_ids

    def test_immediate_evict_removes_hash(self):
        """Blocks freed via immediate_evict should have their hash removed."""
        from vllm.v1.core.block_pool import BlockPool

        pool = BlockPool(
            num_gpu_blocks=10,
            enable_caching=True,
            hash_block_size=16,
            enable_kv_cache_events=False,
        )

        blocks = pool.get_new_blocks(1)
        blk = blocks[0]
        block_hash = (42, (), 0)
        blk.block_hash = block_hash
        pool.cached_block_hash_to_block.insert(block_hash, blk)

        pool.free_blocks_immediate_evict(blocks)

        # Hash should be removed
        assert blk.block_hash is None
        # Block should be in the free queue
        assert pool.free_block_queue.num_free_blocks > 0
