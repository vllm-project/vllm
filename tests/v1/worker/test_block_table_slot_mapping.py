# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``BlockTable.compute_slot_mapping``.

``compute_slot_mapping`` maps logical token positions to physical KV cache
slots. A silent off-by-one here corrupts KV cache contents rather than raising,
so the mapping is covered here through ``BlockTable.compute_slot_mapping``
across plain, hybrid-block, and decode-context-parallel configurations.
"""

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.block_table import BlockTable, SlotMappingMode

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="requires CUDA",
)

MAX_NUM_REQS = 8
MAX_NUM_BLOCKS_PER_REQ = 32
# The kernel pads slots for CUDA graph compatibility in tiles of 1024, so keep
# the default buffer larger than one tile to exercise the padding loop.
MAX_NUM_BATCHED_TOKENS = 2048


def _make_block_table(
    kv_cache_block_size: int = 16,
    kernel_block_size: int | None = None,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
    slot_mapping_mode: SlotMappingMode = SlotMappingMode.TOKEN_TO_KV_SLOT,
    max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS,
    max_num_blocks_per_req: int = MAX_NUM_BLOCKS_PER_REQ,
) -> BlockTable:
    if kernel_block_size is None:
        kernel_block_size = kv_cache_block_size
    block_table = BlockTable(
        block_size=kv_cache_block_size,
        max_num_reqs=MAX_NUM_REQS,
        max_num_blocks_per_req=max_num_blocks_per_req,
        max_num_batched_tokens=max_num_batched_tokens,
        pin_memory=False,
        device=torch.device("cuda"),
        kernel_block_size=kernel_block_size,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        slot_mapping_mode=slot_mapping_mode,
    )
    # Decode context parallelism is derived from the DCP process group, which
    # is not initialized in unit tests; set the resolved values directly.
    block_table.dcp_world_size = dcp_world_size
    block_table.dcp_rank = dcp_rank
    return block_table


def _expand_kernel_block_ids(
    kv_manager_block_ids: list[int],
    blocks_per_kv_block: int,
) -> np.ndarray:
    """Expand KV manager block ids into kernel block ids.

    A KV manager block covers ``blocks_per_kv_block`` consecutive kernel
    blocks, so ``kernel_block_id = kv_block_id * blocks_per_kv_block +
    sub_block``. Kept independent of ``BlockTable.map_to_kernel_blocks`` so a
    bug there cannot cancel out in both the expected and actual values.
    """
    return np.array(
        [
            kv_block_id * blocks_per_kv_block + sub_block
            for kv_block_id in kv_manager_block_ids
            for sub_block in range(blocks_per_kv_block)
        ],
        dtype=np.int64,
    )


def _reference_slot_mapping(
    block_table: BlockTable,
    kv_manager_block_ids: list[list[int]],
    positions_per_req: list[list[int]],
) -> np.ndarray:
    """Recompute the expected slot mapping from the ``BlockTable`` input semantics.

    Starts from the KV manager block ids handed to ``add_row``, expands them
    into kernel block ids independently, and rebuilds the mapping step by step.
    It neither calls the production mapping helpers nor reads the block table
    that ``add_row`` produced:

    1. A virtual block spans ``kv_cache_block_size * dcp_world_size`` tokens,
       because a KV cache block is spread over the DCP ranks.
    2. A token is local when its virtual block offset, chunked by the
       interleave size, is assigned to this rank round-robin.
    3. The local offset is the token's position within this rank's share.
    4. The kernel block index advances by the local offset divided by the
       kernel block size, within the expanded ids of the virtual block.
    5. The slot is ``kernel_block_id * kernel_block_size + offset in block``.
    6. Non-local tokens and trailing padding are ``PAD_SLOT_ID``.
    """
    kernel_block_size = block_table.block_size
    kv_cache_block_size = block_table.kv_cache_block_size
    blocks_per_kv_block = block_table.blocks_per_kv_block
    world_size = block_table.dcp_world_size
    rank = block_table.dcp_rank
    interleave = block_table.cp_kv_cache_interleave_size

    expected = np.full(block_table.max_num_batched_tokens, PAD_SLOT_ID, dtype=np.int64)
    kernel_block_ids = [
        _expand_kernel_block_ids(ids, blocks_per_kv_block)
        for ids in kv_manager_block_ids
    ]

    token_idx = 0
    virtual_block_size = kv_cache_block_size * world_size
    for req_idx, positions in enumerate(positions_per_req):
        for position in positions:
            virtual_block_index = position // virtual_block_size
            virtual_block_offset = position % virtual_block_size
            is_local = (virtual_block_offset // interleave) % world_size == rank
            if is_local:
                local_offset = (
                    virtual_block_offset // (world_size * interleave)
                ) * interleave + (virtual_block_offset % interleave)
                block_index = (
                    virtual_block_index * blocks_per_kv_block
                    + local_offset // kernel_block_size
                )
                block_id = int(kernel_block_ids[req_idx][block_index])
                expected[token_idx] = (
                    block_id * kernel_block_size + local_offset % kernel_block_size
                )
            token_idx += 1
    return expected


def _compute(
    block_table: BlockTable,
    kv_manager_block_ids: list[list[int]],
    positions_per_req: list[list[int]],
) -> tuple[np.ndarray, int]:
    """Populate the table, run the kernel, and return (slot_mapping, num_tokens)."""
    assert len(kv_manager_block_ids) == len(positions_per_req)
    num_reqs = len(kv_manager_block_ids)
    for row_idx, block_ids in enumerate(kv_manager_block_ids):
        block_table.add_row(block_ids, row_idx)
    block_table.commit_block_table(num_reqs)

    query_lens = [len(positions) for positions in positions_per_req]
    query_start_loc = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    positions = np.concatenate(positions_per_req).astype(np.int64)

    block_table.compute_slot_mapping(
        num_reqs,
        torch.from_numpy(query_start_loc).cuda(),
        torch.from_numpy(positions).cuda(),
    )
    slot_mapping = block_table.slot_mapping.gpu.cpu().numpy()
    return slot_mapping, int(query_start_loc[-1])


def _check(
    block_table: BlockTable,
    kv_manager_block_ids: list[list[int]],
    positions_per_req: list[list[int]],
) -> np.ndarray:
    """Assert the kernel matches the reference and pads the unused tail."""
    actual, num_tokens = _compute(block_table, kv_manager_block_ids, positions_per_req)
    expected = _reference_slot_mapping(
        block_table, kv_manager_block_ids, positions_per_req
    )
    np.testing.assert_array_equal(actual[:num_tokens], expected[:num_tokens])
    assert (actual[num_tokens:] == PAD_SLOT_ID).all()
    return actual[:num_tokens]


@pytest.mark.parametrize(
    "kv_manager_block_ids,positions_per_req,expected",
    [
        # block_size 16, blocks [5, 6, 7]:
        #   position 0  -> block 5, offset 0  -> 5 * 16 + 0  = 80
        #   position 15 -> block 5, offset 15 -> 5 * 16 + 15 = 95
        #   position 16 -> block 6, offset 0  -> 6 * 16 + 0  = 96
        #   position 47 -> block 7, offset 15 -> 7 * 16 + 15 = 127
        ([[5, 6, 7]], [[0, 15, 16, 47]], [80, 95, 96, 127]),
        # Two requests sharing the batch, each with its own block table row:
        #   req 0 position 3  -> block index 0 -> row 0 block 9, offset 3
        #                     -> 9 * 16 + 3 = 147
        #   req 1 position 20 -> block index 1 -> row 1 block 4, offset 4
        #                     -> 4 * 16 + 4 = 68
        ([[9, 2], [7, 4]], [[3], [20]], [147, 68]),
    ],
    ids=[
        "single_request_block_boundaries",
        "multiple_request_rows",
    ],
)
def test_slot_mapping_matches_hand_computed_slots(
    kv_manager_block_ids: list[list[int]],
    positions_per_req: list[list[int]],
    expected: list[int],
) -> None:
    """Anchor both the kernel and the reference on hand-computed slots."""
    block_table = _make_block_table()
    expected_array = np.array(expected, dtype=np.int64)

    actual, num_tokens = _compute(block_table, kv_manager_block_ids, positions_per_req)
    np.testing.assert_array_equal(actual[:num_tokens], expected_array)

    reference = _reference_slot_mapping(
        block_table, kv_manager_block_ids, positions_per_req
    )
    np.testing.assert_array_equal(reference[:num_tokens], expected_array)

    assert (actual[num_tokens:] == PAD_SLOT_ID).all()


@pytest.mark.parametrize(
    "kv_manager_block_ids,positions_per_req",
    [
        ([[9]], [[0]]),
        ([[8, 2, 5]], [[0, 15, 16, 31, 32]]),
        ([[3, 7, 1, 5]], [list(range(64))]),
        ([[100, 3, 57, 12, 88, 1]], [list(range(96))]),
        (
            [[1, 2, 3, 4], [10, 11], [20, 21, 22], [30], [40, 41]],
            [list(range(60)), list(range(20)), list(range(33)), [0], list(range(17))],
        ),
        # Tokens scheduled after a cached prefix use absolute, non-zero
        # positions rather than indices into the current batch.
        ([[5, 6, 7, 8, 9, 10]], [list(range(70, 90))]),
    ],
    ids=[
        "single_token",
        "first_and_last_position_in_block",
        "spans_many_blocks",
        "non_contiguous_blocks",
        "ragged_multi_request",
        "prefix_offset",
    ],
)
def test_slot_mapping_covers_block_boundaries_and_ragged_batches(
    kv_manager_block_ids: list[list[int]],
    positions_per_req: list[list[int]],
) -> None:
    _check(_make_block_table(), kv_manager_block_ids, positions_per_req)


def test_slot_mapping_pads_beyond_a_triton_tile() -> None:
    """Real tokens and padding stay correct across the kernel's 1024 tile."""
    # 1030 tokens crosses one tile and is not a multiple of the tile size; the
    # padded tail then crosses a second tile boundary.
    num_tokens = 1030
    block_size = 16
    num_blocks = (num_tokens + block_size - 1) // block_size
    block_table = _make_block_table(
        kv_cache_block_size=block_size,
        max_num_batched_tokens=2100,
        max_num_blocks_per_req=num_blocks,
    )
    actual = _check(
        block_table, [list(range(1, num_blocks + 1))], [list(range(num_tokens))]
    )
    # Anchor the real tokens that cross from the first tile into the second:
    # block ids start at 1, so position 1023 is block 64 offset 15 (slot 1039)
    # and position 1024 is block 65 offset 0 (slot 1040).
    np.testing.assert_array_equal(
        actual[1023:1030], np.arange(1039, 1046, dtype=np.int64)
    )


@pytest.mark.parametrize("dcp_world_size", [2, 4])
@pytest.mark.parametrize("cp_kv_cache_interleave_size", [1, 4])
def test_slot_mapping_partitions_tokens_across_dcp_ranks(
    dcp_world_size: int, cp_kv_cache_interleave_size: int
) -> None:
    """Every token is owned by exactly one DCP rank; others see PAD_SLOT_ID."""
    kv_manager_block_ids = [[1, 2, 3]]
    positions_per_req = [list(range(96))]

    owning_ranks: list[np.ndarray] = []
    for dcp_rank in range(dcp_world_size):
        block_table = _make_block_table(
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )
        actual = _check(block_table, kv_manager_block_ids, positions_per_req)
        owning_ranks.append(actual != PAD_SLOT_ID)

    ownership = np.stack(owning_ranks, axis=0)
    np.testing.assert_array_equal(
        ownership.sum(axis=0),
        np.ones(len(positions_per_req[0]), dtype=np.int64),
    )


def test_slot_mapping_dcp_matches_hand_computed_slots() -> None:
    """Hand-computed DCP mapping for world size 2, interleave 1.

    A virtual block spans 16 * 2 = 32 tokens and rank 1 owns the odd offsets,
    so with KV blocks [5, 6] the expected slots are:
        position 1  -> offset 1  -> local offset 0 -> 5 * 16 + 0 = 80
        position 3  -> offset 3  -> local offset 1 -> 5 * 16 + 1 = 81
        position 5  -> offset 5  -> local offset 2 -> 5 * 16 + 2 = 82
        position 33 -> virtual block 1, offset 1 -> local offset 0
                    -> block 6 -> 6 * 16 + 0 = 96
    while the even positions, including 32, belong to rank 0.
    """
    block_table = _make_block_table(dcp_world_size=2, dcp_rank=1)
    actual, num_tokens = _compute(block_table, [[5, 6]], [[0, 1, 2, 3, 4, 5, 32, 33]])
    np.testing.assert_array_equal(
        actual[:num_tokens],
        np.array(
            [
                PAD_SLOT_ID,
                80,
                PAD_SLOT_ID,
                81,
                PAD_SLOT_ID,
                82,
                PAD_SLOT_ID,
                96,
            ],
            dtype=np.int64,
        ),
    )


@pytest.mark.parametrize(
    "kv_cache_block_size,kernel_block_size",
    [(32, 16), (64, 16)],
)
def test_slot_mapping_with_hybrid_kernel_blocks(
    kv_cache_block_size: int, kernel_block_size: int
) -> None:
    """Slots use kernel blocks when allocation and kernel sizes differ."""
    block_table = _make_block_table(
        kv_cache_block_size=kv_cache_block_size,
        kernel_block_size=kernel_block_size,
    )
    assert block_table.use_hybrid_blocks
    assert block_table.blocks_per_kv_block == kv_cache_block_size // kernel_block_size

    # Cross both kernel block and KV allocation block boundaries.
    num_tokens = 3 * kv_cache_block_size
    _check(block_table, [[5, 9, 2]], [list(range(num_tokens))])


def test_slot_mapping_hybrid_matches_hand_computed_slots() -> None:
    """Hand-computed hybrid mapping for 32-token KV blocks over 16-token kernel blocks.

    ``add_row`` expands KV blocks [5, 9] into kernel blocks [10, 11, 18, 19].
    A virtual block spans the 32-token allocation block, so the kernel block is
    picked by the offset within it, and slots use the kernel block size of 16:
        position 0  -> kernel block 10 offset 0  -> 10 * 16 + 0  = 160
        position 15 -> kernel block 10 offset 15 -> 10 * 16 + 15 = 175
        position 16 -> kernel block 11 offset 0  -> 11 * 16 + 0  = 176
        position 31 -> kernel block 11 offset 15 -> 11 * 16 + 15 = 191
        position 32 -> kernel block 18 offset 0  -> 18 * 16 + 0  = 288
        position 47 -> kernel block 18 offset 15 -> 18 * 16 + 15 = 303
        position 48 -> kernel block 19 offset 0  -> 19 * 16 + 0  = 304
    """
    block_table = _make_block_table(kv_cache_block_size=32, kernel_block_size=16)
    actual, num_tokens = _compute(block_table, [[5, 9]], [[0, 15, 16, 31, 32, 47, 48]])
    np.testing.assert_array_equal(
        actual[:num_tokens],
        np.array([160, 175, 176, 191, 288, 303, 304], dtype=np.int64),
    )


def test_slot_mapping_mode_none_leaves_buffer_untouched() -> None:
    """Mamba-like groups keep the block table as state indices, not token slots."""
    block_table = _make_block_table(slot_mapping_mode=SlotMappingMode.NONE)
    sentinel = -12345
    assert sentinel != PAD_SLOT_ID
    block_table.slot_mapping.gpu.fill_(sentinel)

    actual, _ = _compute(block_table, [[5, 6]], [[0, 1, 2]])

    np.testing.assert_array_equal(actual, np.full_like(actual, sentinel))
