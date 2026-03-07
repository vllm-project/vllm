# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm.distributed.parallel_state import GroupCoordinator

# Test configuration constants
BLOCK_SIZE = 16
MAX_NUM_REQS = 4
MAX_NUM_BLOCKS_PER_REQ = 16
MAX_NUM_BATCHED_TOKENS = 512
PIN_MEMORY = False
DEVICE = torch.device("cpu")
KERNEL_BLOCK_SIZE = 16


def create_block_table(
    dcp_world_size, dcp_rank, pcp_world_size, pcp_rank, cp_kv_cache_interleave_size
):
    """Helper function to create BlockTable with mocked distributed groups.

    Args:
        dcp_world_size: Number of DCP ranks
        dcp_rank: Current DCP rank
        pcp_world_size: Number of PCP ranks
        pcp_rank: Current PCP rank
        cp_kv_cache_interleave_size: Interleave size for KV cache

    Returns:
        BlockTable instance with mocked distributed groups
    """

    with (
        patch("vllm.v1.worker.block_table.get_dcp_group") as mock_get_dcp_group,
        patch("vllm.v1.worker.block_table.get_pcp_group") as mock_get_pcp_group,
    ):
        # Mock DCP group
        mock_dcp_group = MagicMock(spec=GroupCoordinator)
        mock_dcp_group.world_size = dcp_world_size
        mock_dcp_group.rank_in_group = dcp_rank
        mock_get_dcp_group.return_value = mock_dcp_group

        # Mock PCP group
        mock_pcp_group = MagicMock(spec=GroupCoordinator)
        mock_pcp_group.world_size = pcp_world_size
        mock_pcp_group.rank_in_group = pcp_rank
        mock_get_pcp_group.return_value = mock_pcp_group

        from vllm.v1.worker.block_table import BlockTable

        block_table = BlockTable(
            block_size=BLOCK_SIZE,
            max_num_reqs=MAX_NUM_REQS,
            max_num_blocks_per_req=MAX_NUM_BLOCKS_PER_REQ,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            pin_memory=PIN_MEMORY,
            device=DEVICE,
            kernel_block_size=KERNEL_BLOCK_SIZE,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )

        return block_table


def setup_block_table_data(block_table, num_reqs=2):
    """Helper function to populate block table with test data.

    Args:
        block_table: BlockTable instance to populate
        num_reqs: Number of requests to add
    """
    # Add block IDs for each request
    for i in range(num_reqs):
        # [0,1,2,3], [4,5,6,7], etc.
        block_ids = list(range(i * 4, (i + 1) * 4))
        block_table.add_row(block_ids, i)


def test_compute_slot_mapping_dcp1_pcp1_interleave1():
    """Test compute_slot_mapping with DCP=1, PCP=1, interleave_size=1.

    With no parallelism (DCP=1, PCP=1), all tokens are local to the single
    rank.

    Setup:
    - Block size: 16
    - Request 0 has blocks: [0, 1, 2, 3]
    - Request 1 has blocks: [4, 5, 6, 7]

    Test positions for each request:
    - Request 0, position 0: block_id=0, offset=0 → slot = 0*16+0 = 0
    - Request 0, position 1: block_id=0, offset=1 → slot = 0*16+1 = 1
    - Request 1, position 0: block_id=4, offset=0 → slot = 4*16+0 = 64
    - Request 1, position 1: block_id=4, offset=1 → slot = 4*16+1 = 65
    """
    req_indices = np.array([0, 0, 1, 1], dtype=np.int32)
    positions = np.array([0, 1, 0, 1], dtype=np.int32)
    expected_result = np.array([0, 1, 64, 65], dtype=np.int32)

    block_table = create_block_table(
        dcp_world_size=1,
        dcp_rank=0,
        pcp_world_size=1,
        pcp_rank=0,
        cp_kv_cache_interleave_size=1,
    )

    num_reqs = max(req_indices) + 1 if len(req_indices) > 0 else 1
    setup_block_table_data(block_table, num_reqs=num_reqs)

    block_table.compute_slot_mapping(req_indices, positions)

    actual_result = block_table.slot_mapping.np[: len(positions)]
    np.testing.assert_array_equal(
        actual_result,
        expected_result,
        "DCP=1, PCP=1, interleave=1, dcp_rank=0, pcp_rank=0",
    )


@pytest.mark.parametrize(
    "pcp_rank,dcp_rank,expected_result",
    [
        # Rank 0 (pcp=0, dcp=0): positions 0, 8
        (0, 0, [0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1]),
        # Rank 1 (pcp=0, dcp=1): positions 1, 9
        (0, 1, [-1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1]),
        # Rank 2 (pcp=0, dcp=2): positions 2, 10
        (0, 2, [-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1]),
        # Rank 3 (pcp=0, dcp=3): positions 3, 11
        (0, 3, [-1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1]),
        # Rank 4 (pcp=1, dcp=0): positions 4, 12
        (1, 0, [-1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1]),
        # Rank 5 (pcp=1, dcp=1): positions 5, 13
        (1, 1, [-1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1]),
        # Rank 6 (pcp=1, dcp=2): positions 6, 14
        (1, 2, [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1, -1]),
        # Rank 7 (pcp=1, dcp=3): positions 7, 15
        (1, 3, [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 1]),
    ],
)
def test_compute_slot_mapping_dcp4_pcp2_interleave1(
    pcp_rank, dcp_rank, expected_result
):
    """Test compute_slot_mapping with DCP=4, PCP=2, interleave_size=1.

    With interleave_size=1, tokens are distributed round-robin across all 8
    ranks:
    - Position 0 → Rank 0
    - Position 1 → Rank 1
    - Position 2 → Rank 2
    - ...
    - Position 7 → Rank 7
    - Position 8 → Rank 0 (wraps around)
    """
    req_indices = np.array([0] * 16, dtype=np.int32)
    positions = np.arange(16, dtype=np.int32)
    expected_result = np.array(expected_result, dtype=np.int32)

    block_table = create_block_table(
        dcp_world_size=4,
        dcp_rank=dcp_rank,
        pcp_world_size=2,
        pcp_rank=pcp_rank,
        cp_kv_cache_interleave_size=1,
    )

    num_reqs = max(req_indices) + 1 if len(req_indices) > 0 else 1
    setup_block_table_data(block_table, num_reqs=num_reqs)

    block_table.compute_slot_mapping(req_indices, positions)

    actual_result = block_table.slot_mapping.np[: len(positions)]
    np.testing.assert_array_equal(
        actual_result,
        expected_result,
        f"DCP=4, PCP=2, interleave=1, dcp_rank={dcp_rank}, pcp_rank={pcp_rank}",
    )


@pytest.mark.parametrize(
    "pcp_rank,dcp_rank,expected_positions",
    [
        # Rank 0 gets positions 0-15
        (0, 0, (0, 16)),
        # Rank 1 gets positions 16-17
        (0, 1, (16, 18)),
        # Rank 2 gets no positions
        (0, 2, None),
        # Rank 3 gets no positions
        (0, 3, None),
        # Rank 4 gets no positions
        (1, 0, None),
        # Rank 5 gets no positions
        (1, 1, None),
        # Rank 6 gets no positions
        (1, 2, None),
        # Rank 7 gets no positions
        (1, 3, None),
    ],
)
def test_compute_slot_mapping_dcp4_pcp2_interleave16(
    pcp_rank, dcp_rank, expected_positions
):
    """Test compute_slot_mapping with DCP=4, PCP=2, interleave_size=16.

    With interleave_size=16, tokens are distributed in chunks of 16 across
    ranks. Virtual block size = 16 * 4 * 2 = 16

    Token distribution with interleave_size=16:
    - Positions 0-15 belong to rank 0 (first chunk of 16)
    - Positions 16-31 belong to rank 1 (second chunk of 16)
    - Positions 32-47 belong to rank 2 (third chunk of 16)
    - And so on...

    Using 18 positions ensures we test both rank 0 (positions 0-15) and rank
    1 (positions 16-17).
    """
    num_positions = 18
    req_indices = np.array([0] * num_positions, dtype=np.int32)
    positions = np.arange(num_positions, dtype=np.int32)

    block_table = create_block_table(
        dcp_world_size=4,
        dcp_rank=dcp_rank,
        pcp_world_size=2,
        pcp_rank=pcp_rank,
        cp_kv_cache_interleave_size=16,
    )

    num_reqs = max(req_indices) + 1 if len(req_indices) > 0 else 1
    setup_block_table_data(block_table, num_reqs=num_reqs)

    block_table.compute_slot_mapping(req_indices, positions)

    actual_result = block_table.slot_mapping.np[: len(positions)]

    # Build expected result based on which positions this rank owns
    expected_result = np.full(num_positions, -1, dtype=np.int32)
    if expected_positions is not None:
        start_pos, end_pos = expected_positions
        # For positions this rank owns, map to local slot indices
        for i, pos in enumerate(range(start_pos, end_pos)):
            expected_result[pos] = i

    np.testing.assert_array_equal(
        actual_result,
        expected_result,
        f"DCP=4, PCP=2, interleave=16, dcp_rank={dcp_rank}, pcp_rank={pcp_rank}",
    )
