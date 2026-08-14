# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ChunkedTokenDatabase.prepare_values."""

import random

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
)
from vllm.utils.math_utils import cdiv

BLOCK_SIZE = 128


def _reference_prepare_value(
    db: ChunkedTokenDatabase, start: int, end: int, block_ids: list[int]
) -> tuple[list[int], list[int], int]:
    """Compute a token range with the original scalar implementation."""
    addr_list = []
    size_list = []
    block_id = block_ids[start // db.block_size]
    length = len(db.block_len)
    for index, base_addr in enumerate(db.kv_caches_base_addr):
        addr = base_addr + block_id * db.block_len[index % length]
        assert (end - start) % db.block_size == 0
        size = db.block_len[index % length] * cdiv(end - start, db.block_size)
        addr_list.append(addr)
        size_list.append(size)
    return addr_list, size_list, block_id


def _make_db(num_regions: int, num_block_lens: int) -> ChunkedTokenDatabase:
    md = KeyMetadata(model_name="t", tp_rank=1, pcp_rank=0, dcp_rank=0, pp_rank=0)
    db = ChunkedTokenDatabase(md, BLOCK_SIZE)
    db.set_kv_caches_base_addr(
        [0x7F00_0000_0000 + i * (1 << 30) for i in range(num_regions)]
    )
    # Exercise repeated block lengths when there are more cache regions.
    db.set_block_len([30_208 + 512 * i for i in range(num_block_lens)])
    return db


@pytest.mark.parametrize("num_regions,num_block_lens", [(96, 96), (96, 2), (1, 1)])
def test_prepare_values_matches_reference(num_regions: int, num_block_lens: int):
    db = _make_db(num_regions, num_block_lens)
    rng = random.Random(0)
    n_blocks = 300
    block_ids = [rng.randrange(0, 1 << 20) for _ in range(n_blocks)]
    chunks = []
    b = 0
    while b < n_blocks - 4:
        span = rng.choice([1, 1, 1, 2, 4])
        chunks.append((b * BLOCK_SIZE, (b + span) * BLOCK_SIZE))
        b += span + rng.choice([0, 1])

    addrs, sizes, bids = db.prepare_values(chunks, block_ids)
    assert len(addrs) == len(sizes) == len(bids) == len(chunks)
    for (start, end), addr, size, bid in zip(chunks, addrs, sizes, bids):
        ref_addr, ref_size, ref_bid = _reference_prepare_value(
            db, start, end, block_ids
        )
        assert addr == ref_addr
        assert size == ref_size
        assert bid == ref_bid
        # Native bindings require Python ints rather than numpy scalars.
        assert all(type(a) is int for a in addr)
        assert type(bid) is int


def test_prepare_value_single_matches_reference():
    db = _make_db(8, 8)
    block_ids = list(range(64))
    got = db.prepare_value(5 * BLOCK_SIZE, 7 * BLOCK_SIZE, block_ids)
    assert got == _reference_prepare_value(
        db, 5 * BLOCK_SIZE, 7 * BLOCK_SIZE, block_ids
    )


def test_prepare_values_empty():
    db = _make_db(4, 4)
    assert db.prepare_values([], [1, 2, 3]) == ([], [], [])


def test_prepare_values_rejects_unaligned_chunk():
    db = _make_db(4, 4)
    with pytest.raises(AssertionError):
        db.prepare_values([(0, BLOCK_SIZE + 1)], [0, 1])
