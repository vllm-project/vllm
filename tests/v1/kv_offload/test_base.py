# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the OffloadKey codec in vllm.v1.kv_offload.base."""

from vllm.v1.kv_offload.base import (
    get_offload_block_hash,
    get_offload_chunk_idx,
    get_offload_group_idx,
    make_offload_key,
)


def test_offload_key_round_trip():
    block_hash = bytes(range(8))
    key = make_offload_key(block_hash, group_idx=3, chunk_idx=7)
    assert get_offload_block_hash(key) == block_hash
    assert get_offload_group_idx(key) == 3
    assert get_offload_chunk_idx(key) == 7


def test_offload_key_round_trip_u32_max():
    # group_idx and chunk_idx are packed as unsigned 32-bit big-endian ints.
    block_hash = b"\xab" * 16
    key = make_offload_key(block_hash, group_idx=2**32 - 1, chunk_idx=2**32 - 2)
    assert get_offload_block_hash(key) == block_hash
    assert get_offload_group_idx(key) == 2**32 - 1
    assert get_offload_chunk_idx(key) == 2**32 - 2


def test_chunk_idx_disambiguates_same_hash_and_group():
    block_hash = bytes(range(8))
    key0 = make_offload_key(block_hash, group_idx=1, chunk_idx=0)
    key1 = make_offload_key(block_hash, group_idx=1, chunk_idx=1)
    assert key0 != key1
    assert get_offload_block_hash(key0) == get_offload_block_hash(key1)
    assert get_offload_group_idx(key0) == get_offload_group_idx(key1)
    assert (get_offload_chunk_idx(key0), get_offload_chunk_idx(key1)) == (0, 1)
