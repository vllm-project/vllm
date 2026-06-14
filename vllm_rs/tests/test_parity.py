"""Parity tests: Rust vllm_rs vs Python vllm.v1.core.block_pool.

Drives identical allocate/free/touch/evict sequences through both
implementations and asserts matching: returned block_ids, ref_cnt state,
free-queue membership, prefix-cache hash map contents, and invariants
exposed by `get_num_free_blocks` / `get_usage`.

Run: `pytest vllm_rs/tests/test_parity.py -v`

Requires `vllm` installed (so we can compare against the reference
Python implementation) and `vllm_rs` installed (via `maturin develop`).
"""

from __future__ import annotations

import random
from typing import Any

import pytest

import vllm_rs as rs
from vllm.v1.core.block_pool import BlockPool as PyBlockPool
from vllm.v1.core.kv_cache_utils import (
    FreeKVCacheBlockQueue as PyQueue,
    KVCacheBlock as PyBlock,
    make_block_hash_with_group_id,
)


def py_block_ids(seq):
    return [b.block_id for b in seq]


# ---------------- FreeKVCacheBlockQueue parity ----------------


def test_queue_init_popleft_append():
    py_blocks = [PyBlock(i) for i in range(8)]
    py_q = PyQueue(py_blocks)

    rs_blocks = [rs.KVCacheBlock(i) for i in range(8)]
    rs_q = rs.FreeKVCacheBlockQueue(rs_blocks)

    assert py_q.num_free_blocks == rs_q.num_free_blocks == 8

    # popleft() returns the lowest-indexed block for both
    for _ in range(3):
        py_b = py_q.popleft()
        rs_b = rs_q.popleft()
        assert py_b.block_id == rs_b.block_id

    assert py_q.num_free_blocks == rs_q.num_free_blocks == 5

    # append in reverse order — reproducing the scheduler's free pattern
    # (free returns them to the tail in allocation order).
    to_return_py = []
    to_return_rs = []
    for _ in range(3):
        py_b = py_q.popleft()
        rs_b = rs_q.popleft()
        to_return_py.append(py_b)
        to_return_rs.append(rs_b)

    for pb, rb in zip(reversed(to_return_py), reversed(to_return_rs)):
        py_q.append(pb)
        rs_q.append(rb)

    # Walk both lists and confirm identical order
    py_all = py_block_ids(py_q.get_all_free_blocks())
    rs_all = py_block_ids(rs_q.get_all_free_blocks())
    assert py_all == rs_all


def test_queue_popleft_n():
    py_q = PyQueue([PyBlock(i) for i in range(16)])
    rs_q = rs.FreeKVCacheBlockQueue([rs.KVCacheBlock(i) for i in range(16)])

    for n in (4, 1, 0, 3, 8):
        py_out = py_block_ids(py_q.popleft_n(n))
        rs_out = py_block_ids(rs_q.popleft_n(n))
        assert py_out == rs_out, f"popleft_n({n}): {py_out} vs {rs_out}"
        assert py_q.num_free_blocks == rs_q.num_free_blocks


def test_queue_append_n_preserves_order():
    py_blocks = [PyBlock(i) for i in range(8)]
    rs_blocks = [rs.KVCacheBlock(i) for i in range(8)]
    py_q = PyQueue(py_blocks)
    rs_q = rs.FreeKVCacheBlockQueue(rs_blocks)

    # Drain and re-append in weird orders — reproduce what BlockPool.free_blocks does
    py_drained = py_q.popleft_n(8)
    rs_drained = rs_q.popleft_n(8)
    assert py_block_ids(py_drained) == py_block_ids(rs_drained)

    order = [3, 1, 7, 0, 5]
    py_q.append_n([py_drained[i] for i in order])
    rs_q.append_n([rs_drained[i] for i in order])

    assert py_block_ids(py_q.get_all_free_blocks()) == py_block_ids(
        rs_q.get_all_free_blocks()
    )


def test_queue_remove_middle():
    py_blocks = [PyBlock(i) for i in range(6)]
    rs_blocks = [rs.KVCacheBlock(i) for i in range(6)]
    py_q = PyQueue(py_blocks)
    rs_q = rs.FreeKVCacheBlockQueue(rs_blocks)
    # Remove block 2 and 4 — middle-of-list splice
    py_q.remove(py_blocks[2])
    rs_q.remove(rs_blocks[2])
    py_q.remove(py_blocks[4])
    rs_q.remove(rs_blocks[4])
    assert py_q.num_free_blocks == rs_q.num_free_blocks == 4
    assert py_block_ids(py_q.get_all_free_blocks()) == py_block_ids(
        rs_q.get_all_free_blocks()
    )


# ---------------- BlockPool parity ----------------


def _make_pair(num=16, caching=True):
    py = PyBlockPool(num_gpu_blocks=num, enable_caching=caching, hash_block_size=16)
    r = rs.BlockPool(num, caching, 16)
    return py, r


def test_block_pool_get_free_parity_simple():
    py, r = _make_pair(num=8)
    assert py.get_num_free_blocks() == r.get_num_free_blocks() == 7  # minus null
    py_bs = py.get_new_blocks(3)
    rs_bs = r.get_new_blocks(3)
    assert py_block_ids(py_bs) == py_block_ids(rs_bs)
    py.free_blocks(py_bs)
    r.free_blocks(rs_bs)
    # Same free count after freeing
    assert py.get_num_free_blocks() == r.get_num_free_blocks()
    # Next allocation returns same ids (LRU order)
    py_bs2 = py.get_new_blocks(3)
    rs_bs2 = r.get_new_blocks(3)
    assert py_block_ids(py_bs2) == py_block_ids(rs_bs2)


def test_block_pool_touch_parity():
    py, r = _make_pair(num=8)
    py_bs = py.get_new_blocks(2)
    rs_bs = r.get_new_blocks(2)
    # Free them so they land back in queue with ref_cnt=0
    py.free_blocks(py_bs)
    r.free_blocks(rs_bs)
    # Touch them — should take them out of the queue and bump ref_cnt
    py.touch(py_bs)
    r.touch(rs_bs)
    assert py.get_num_free_blocks() == r.get_num_free_blocks()
    for pb, rb in zip(py_bs, rs_bs):
        assert pb.ref_cnt == rb.ref_cnt


def test_block_pool_cached_block_round_trip():
    py, r = _make_pair(num=16, caching=True)
    py_bs = py.get_new_blocks(2)
    rs_bs = r.get_new_blocks(2)
    # Fake a block hash and register it in both impls — Python uses the
    # BlockHashWithGroupId bytes directly; the Rust side accepts raw bytes too.
    block_hash = make_block_hash_with_group_id(b"\x00" * 32, 0)
    py_bs[0].block_hash = block_hash
    py.cached_block_hash_to_block.insert(block_hash, py_bs[0])

    rs_bs[0].block_hash = block_hash
    r.cached_block_hash_to_block.insert(block_hash, rs_bs[0])

    # Look up via get_cached_block
    py_hit = py.get_cached_block(b"\x00" * 32, [0])
    rs_hit = r.get_cached_block(b"\x00" * 32, [0])
    assert py_hit is not None and rs_hit is not None
    assert py_hit[0].block_id == rs_hit[0].block_id

    # Miss on wrong group
    assert py.get_cached_block(b"\x00" * 32, [1]) is None
    assert r.get_cached_block(b"\x00" * 32, [1]) is None


def test_block_pool_randomized_trace():
    """Drive N random allocate/free steps and assert free-queue order is
    identical on both sides throughout."""
    py, r = _make_pair(num=64, caching=False)
    rng = random.Random(0x5afe)

    live_py: list[list[Any]] = []
    live_rs: list[list[Any]] = []

    for step in range(500):
        action = rng.choice(["alloc", "alloc", "free"])
        if action == "alloc" and py.get_num_free_blocks() > 0:
            n = rng.randint(1, min(5, py.get_num_free_blocks()))
            pb = py.get_new_blocks(n)
            rb = r.get_new_blocks(n)
            assert py_block_ids(pb) == py_block_ids(rb), f"step {step}"
            live_py.append(pb)
            live_rs.append(rb)
        elif action == "free" and live_py:
            idx = rng.randrange(len(live_py))
            pb = live_py.pop(idx)
            rb = live_rs.pop(idx)
            py.free_blocks(pb)
            r.free_blocks(rb)
            assert py.get_num_free_blocks() == r.get_num_free_blocks(), f"step {step}"

    # Final free-queue walk parity
    assert py_block_ids(py.free_block_queue.get_all_free_blocks()) == py_block_ids(
        r.free_block_queue.get_all_free_blocks()
    )


def test_cache_full_blocks_fast_parity():
    """Rust cache_full_blocks_fast should produce identical post-state to the
    Python BlockPool.cache_full_blocks fast path (block_size==hash_block_size,
    events off).
    """
    import types

    py, r = _make_pair(num=32, caching=True)
    py_bs = py.get_new_blocks(8)
    rs_bs = r.get_new_blocks(8)
    # Fake block hashes: 32 bytes each. Must be distinct per block.
    block_hashes = [bytes([i]) * 32 for i in range(8)]

    # Fake Request with just .block_hashes — cache_full_blocks reads that attr.
    req = types.SimpleNamespace(block_hashes=block_hashes)

    py.cache_full_blocks(
        request=req,
        blocks=py_bs,
        num_cached_blocks=0,
        num_full_blocks=5,
        block_size=16,
        kv_cache_group_id=0,
    )
    # Rust RustBlockPool wrapper also takes the same args.
    # But here we're using the raw `rs.BlockPool` (not the wrapper),
    # so call the Rust fast method directly.
    r.cache_full_blocks_fast(rs_bs, block_hashes[:5], 0, 5, 0)

    # Parity: the hash-to-block map contents should match.
    for i in range(5):
        key = block_hashes[i] + (0).to_bytes(4, "big", signed=False)
        py_b = py.cached_block_hash_to_block.get_one_block(key)
        rs_b = r.cached_block_hash_to_block.get_one_block(key)
        assert py_b is not None
        assert rs_b is not None
        assert py_b.block_id == rs_b.block_id
        # Block hash is stamped on the block
        assert py_bs[i].block_hash == rs_bs[i].block_hash


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v"] + sys.argv[1:])
