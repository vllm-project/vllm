# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the routed experts virtual slot mapping kernel.

Tests correctness against a pure-numpy reference implementation.
The kernel assigns a unique virtual slot to every token position
under context parallelism (DCP/PCP), using the same interleave
decomposition as the attention slot_mapping kernel.
"""

import numpy as np
import pytest
import torch

from vllm.v1.worker.re_slot_mapping import compute_re_slot_mapping


def _reference_re_slot_mapping(
    positions: np.ndarray,
    block_table: np.ndarray,
    block_size: int,
    total_cp: int,
    interleave: int,
) -> np.ndarray:
    """Pure-numpy reference for virtual slot computation."""
    virtual_block_size = block_size * total_cp
    block_indices = positions // virtual_block_size
    block_numbers = block_table[block_indices].astype(np.int64)

    virtual_block_offsets = positions - block_indices * virtual_block_size
    token_rank = (virtual_block_offsets // interleave) % total_cp
    local_block_offsets = (
        virtual_block_offsets // (total_cp * interleave)
    ) * interleave + (virtual_block_offsets % interleave)

    return (block_numbers * block_size + local_block_offsets) * total_cp + token_rank


@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("total_cp", [2, 4])
@pytest.mark.parametrize("interleave", [1, 2])
@pytest.mark.parametrize("num_reqs", [1, 4])
def test_re_slot_mapping_matches_reference(
    block_size: int,
    total_cp: int,
    interleave: int,
    num_reqs: int,
) -> None:
    """Triton kernel output should match the numpy reference."""
    device = "cuda"
    rng = np.random.RandomState(42)

    tokens_per_req = rng.randint(1, 129, size=num_reqs)
    num_tokens = int(tokens_per_req.sum())

    max_blocks_per_req = 64
    block_table_np = rng.randint(0, 500, size=(num_reqs, max_blocks_per_req)).astype(
        np.int32
    )

    query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(tokens_per_req, out=query_start_loc_np[1:])

    virtual_block_size = block_size * total_cp
    all_positions = []
    for r in range(num_reqs):
        max_pos = block_table_np.shape[1] * virtual_block_size - 1
        positions_r = np.arange(tokens_per_req[r], dtype=np.int64)
        positions_r = np.minimum(positions_r, max_pos)
        all_positions.append(positions_r)
    positions_np = np.concatenate(all_positions)

    query_start_loc = torch.tensor(query_start_loc_np, dtype=torch.int32, device=device)
    positions = torch.tensor(positions_np, dtype=torch.int64, device=device)
    block_table_gpu = torch.tensor(block_table_np, dtype=torch.int32, device=device)
    out = torch.empty(num_tokens, dtype=torch.int64, device=device)

    compute_re_slot_mapping(
        num_reqs=num_reqs,
        query_start_loc=query_start_loc,
        positions=positions,
        block_table=block_table_gpu,
        block_table_stride=max_blocks_per_req,
        block_size=block_size,
        total_cp_world_size=total_cp,
        cp_kv_cache_interleave_size=interleave,
        out=out,
    )

    for r in range(num_reqs):
        start = query_start_loc_np[r]
        end = query_start_loc_np[r + 1]
        pos_r = positions_np[start:end]
        ref = _reference_re_slot_mapping(
            pos_r,
            block_table_np[r],
            block_size,
            total_cp,
            interleave,
        )
        actual = out[start:end].cpu().numpy()
        np.testing.assert_array_equal(
            actual,
            ref,
            err_msg=f"req {r}: virtual slots differ (block_size={block_size}, "
            f"total_cp={total_cp}, interleave={interleave})",
        )


@pytest.mark.parametrize("total_cp", [2, 4])
@pytest.mark.parametrize("interleave", [1, 2])
def test_re_slot_mapping_unique_slots(
    total_cp: int,
    interleave: int,
) -> None:
    """Every position in a single request must map to a unique virtual slot."""
    device = "cuda"
    block_size = 16
    num_reqs = 1
    num_tokens = 256
    max_blocks_per_req = 64

    rng = np.random.RandomState(7)
    block_ids = rng.choice(500, size=max_blocks_per_req, replace=False).astype(np.int32)
    block_table_np = block_ids.reshape(1, -1)

    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    block_table_gpu = torch.tensor(block_table_np, dtype=torch.int32, device=device)
    out = torch.empty(num_tokens, dtype=torch.int64, device=device)

    compute_re_slot_mapping(
        num_reqs=num_reqs,
        query_start_loc=query_start_loc,
        positions=positions,
        block_table=block_table_gpu,
        block_table_stride=max_blocks_per_req,
        block_size=block_size,
        total_cp_world_size=total_cp,
        cp_kv_cache_interleave_size=interleave,
        out=out,
    )

    slots = out.cpu().numpy()
    assert len(np.unique(slots)) == num_tokens, (
        f"Expected {num_tokens} unique virtual slots, "
        f"got {len(np.unique(slots))} (total_cp={total_cp}, interleave={interleave})"
    )


def test_re_slot_mapping_all_non_negative() -> None:
    """Virtual slots must never be negative (no PAD_SLOT_ID)."""
    device = "cuda"
    block_size = 16
    total_cp = 2
    interleave = 1
    num_reqs = 3
    max_blocks_per_req = 32

    rng = np.random.RandomState(99)
    tokens_per_req = rng.randint(10, 65, size=num_reqs)
    num_tokens = int(tokens_per_req.sum())

    block_table_np = rng.randint(0, 200, size=(num_reqs, max_blocks_per_req)).astype(
        np.int32
    )
    query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(tokens_per_req, out=query_start_loc_np[1:])

    positions_np = np.concatenate(
        [np.arange(t, dtype=np.int64) for t in tokens_per_req]
    )

    query_start_loc = torch.tensor(query_start_loc_np, dtype=torch.int32, device=device)
    positions = torch.tensor(positions_np, dtype=torch.int64, device=device)
    block_table_gpu = torch.tensor(block_table_np, dtype=torch.int32, device=device)
    out = torch.empty(num_tokens, dtype=torch.int64, device=device)

    compute_re_slot_mapping(
        num_reqs=num_reqs,
        query_start_loc=query_start_loc,
        positions=positions,
        block_table=block_table_gpu,
        block_table_stride=max_blocks_per_req,
        block_size=block_size,
        total_cp_world_size=total_cp,
        cp_kv_cache_interleave_size=interleave,
        out=out,
    )

    slots = out.cpu().numpy()
    assert np.all(slots >= 0), f"Found negative virtual slots: {slots[slots < 0]}"
