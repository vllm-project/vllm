# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for prefix-cache hash collision detection.

Verifies that when verify_content_on_hit=True (automatically enabled for
non-cryptographic hash algorithms like xxhash), the BlockPool detects and
rejects colliding blocks with different token content.
"""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.request import Request


def _make_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int,
    hash_fn=sha256,
):
    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        block_hasher=get_request_block_hasher(block_size, hash_fn),
    )


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


class TestCollisionDetection:
    """Tests for insertion-time hash collision detection in BlockPool."""

    def test_collision_detected_and_rejected(self):
        """Two blocks with different tokens but the same hash must not
        both be inserted when verification is enabled."""
        block_size = 4
        kv_cache_group_id = 0

        pool = BlockPool(
            num_gpu_blocks=4,
            enable_caching=True,
            hash_block_size=block_size,
            verify_content_on_hit=True,
        )

        fake_hash = BlockHash(b"collision-hash-value")
        key = make_block_hash_with_group_id(fake_hash, kv_cache_group_id)

        block_a = pool.get_new_blocks(1)[0]
        block_b = pool.get_new_blocks(1)[0]

        fp_a = (1, 2, 3, 4)
        fp_b = (5, 6, 7, 8)

        pool._insert_block_hash(key, block_a, num_tokens=4, content_fingerprint=fp_a)
        assert pool.cached_block_hash_to_block.get_one_block(key) is not None
        assert key in pool._block_fingerprints
        assert pool._block_fingerprints[key] == fp_a

        # Second insertion with different content should be rejected.
        key2 = make_block_hash_with_group_id(fake_hash, kv_cache_group_id)
        pool._insert_block_hash(key2, block_b, num_tokens=4, content_fingerprint=fp_b)

        # The original block should still be there, not overwritten.
        cached = pool.cached_block_hash_to_block.get_one_block(key)
        assert cached is not None
        assert cached.block_id == block_a.block_id

    def test_same_content_allowed(self):
        """Two blocks with the same hash AND same content are both accepted
        (this is the normal de-dup case)."""
        block_size = 4
        kv_cache_group_id = 0

        pool = BlockPool(
            num_gpu_blocks=4,
            enable_caching=True,
            hash_block_size=block_size,
            verify_content_on_hit=True,
        )

        fake_hash = BlockHash(b"same-hash")
        key = make_block_hash_with_group_id(fake_hash, kv_cache_group_id)
        fp = (10, 20, 30, 40)

        block_a = pool.get_new_blocks(1)[0]
        block_b = pool.get_new_blocks(1)[0]

        pool._insert_block_hash(key, block_a, num_tokens=4, content_fingerprint=fp)
        pool._insert_block_hash(key, block_b, num_tokens=4, content_fingerprint=fp)

        # Both should be in the map (different block_ids, same hash).
        assert pool.cached_block_hash_to_block.contain(key, block_a.block_id)
        assert pool.cached_block_hash_to_block.contain(key, block_b.block_id)

    def test_no_verification_when_disabled(self):
        """When verify_content_on_hit=False (sha256), no fingerprints are
        stored and collisions are not checked."""
        block_size = 4
        kv_cache_group_id = 0

        pool = BlockPool(
            num_gpu_blocks=4,
            enable_caching=True,
            hash_block_size=block_size,
            verify_content_on_hit=False,
        )

        fake_hash = BlockHash(b"noverify-hash")
        key = make_block_hash_with_group_id(fake_hash, kv_cache_group_id)

        block_a = pool.get_new_blocks(1)[0]
        block_b = pool.get_new_blocks(1)[0]

        fp_a = (1, 2, 3, 4)
        fp_b = (5, 6, 7, 8)

        pool._insert_block_hash(key, block_a, num_tokens=4, content_fingerprint=fp_a)
        pool._insert_block_hash(key, block_b, num_tokens=4, content_fingerprint=fp_b)

        # Both inserted (no collision check), no fingerprints stored.
        assert pool.cached_block_hash_to_block.contain(key, block_a.block_id)
        assert pool.cached_block_hash_to_block.contain(key, block_b.block_id)
        assert len(pool._block_fingerprints) == 0

    def test_fingerprints_cleaned_on_eviction(self):
        """Fingerprints are removed when blocks are evicted."""
        block_size = 4
        kv_cache_group_id = 0

        pool = BlockPool(
            num_gpu_blocks=3,
            enable_caching=True,
            hash_block_size=block_size,
            verify_content_on_hit=True,
        )

        fake_hash = BlockHash(b"evict-hash")
        key = make_block_hash_with_group_id(fake_hash, kv_cache_group_id)
        fp = (100, 200, 300, 400)

        block = pool.get_new_blocks(1)[0]
        pool._insert_block_hash(key, block, num_tokens=4, content_fingerprint=fp)
        assert key in pool._block_fingerprints

        pool._remove_cached_block_hashes(block)
        assert key not in pool._block_fingerprints

    def test_fingerprints_cleaned_on_reset(self):
        """Fingerprints are cleared when prefix cache is reset."""
        block_size = 4

        pool = BlockPool(
            num_gpu_blocks=3,
            enable_caching=True,
            hash_block_size=block_size,
            verify_content_on_hit=True,
        )

        fake_hash = BlockHash(b"reset-hash")
        key = make_block_hash_with_group_id(fake_hash, 0)
        fp = (1, 2, 3, 4)

        block = pool.get_new_blocks(1)[0]
        pool._insert_block_hash(key, block, num_tokens=4, content_fingerprint=fp)
        assert len(pool._block_fingerprints) == 1

        pool.free_blocks([block])
        pool.reset_prefix_cache()
        assert len(pool._block_fingerprints) == 0

    def test_cache_full_blocks_stores_fingerprint(self):
        """cache_full_blocks stores fingerprints when verification is on."""
        block_size = 4
        kv_cache_group_id = 0

        pool = BlockPool(
            num_gpu_blocks=4,
            enable_caching=True,
            hash_block_size=block_size,
            verify_content_on_hit=True,
        )

        token_ids = [10, 20, 30, 40, 50, 60, 70, 80]
        req = _make_request("req1", token_ids, block_size)

        blocks = pool.get_new_blocks(2)
        pool.cache_full_blocks(
            request=req,
            blocks=blocks,
            num_cached_blocks=0,
            num_full_blocks=2,
            block_size=block_size,
            kv_cache_group_id=kv_cache_group_id,
        )

        assert len(pool._block_fingerprints) == 2

        fps = list(pool._block_fingerprints.values())
        assert fps[0] == tuple(token_ids[:4])
        assert fps[1] == tuple(token_ids[4:8])
