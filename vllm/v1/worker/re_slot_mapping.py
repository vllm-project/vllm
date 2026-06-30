# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Virtual slot mapping kernel for routed experts replay under context parallelism.

With context parallelism (DCP/PCP), the attention slot_mapping uses
PAD_SLOT_ID (-1) for non-local token positions. The routed experts
replay feature needs a slot for EVERY position so routing data can be
stored and reconstructed without cross-rank communication.

This kernel computes a "virtual slot" for each position:

    virtual_slot = (block_id * block_size + local_offset) * total_cp + rank

where ``rank`` and ``local_offset`` are derived from the token's global
position using the CP interleave formula (same decomposition as the
attention slot_mapping kernel in block_table.py).

The virtual slot space is ``num_blocks * block_size * total_cp`` — each
physical slot fans out into ``total_cp`` virtual slots.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _compute_re_slot_mapping_kernel(
    num_tokens,
    query_start_loc_ptr,  # [num_reqs + 1], int32
    positions_ptr,  # [num_tokens], int64
    block_table_ptr,  # [max_num_reqs, max_num_blocks_per_req], int32 (flat)
    block_table_stride,  # max_num_blocks_per_req
    block_size,
    re_slot_mapping_ptr,  # [num_tokens], int64
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Virtual slot mapping for routed experts replay under CP.

    Unlike the attention kernel, this assigns a unique virtual slot to
    EVERY position (no PAD, no is_local filter). The virtual slot space
    is ``num_blocks * block_size * TOTAL_CP_WORLD_SIZE``.
    """
    req_idx = tl.program_id(0)

    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
    row_offset = req_idx * block_table_stride
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)

        block_indices = pos // virtual_block_size
        block_numbers = tl.load(block_table_ptr + row_offset + block_indices).to(
            tl.int64
        )

        virtual_block_offsets = pos - block_indices * virtual_block_size
        token_rank = (
            virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
        ) % TOTAL_CP_WORLD_SIZE
        local_block_offsets = (
            virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
        ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
            virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
        )

        virtual_slot_ids = (
            block_numbers * block_size + local_block_offsets
        ) * TOTAL_CP_WORLD_SIZE + token_rank
        tl.store(re_slot_mapping_ptr + offsets, virtual_slot_ids, mask=mask)


def compute_re_slot_mapping(
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_table_stride: int,
    block_size: int,
    total_cp_world_size: int,
    cp_kv_cache_interleave_size: int,
    out: torch.Tensor,
) -> None:
    """Compute virtual slot mapping for routed experts replay under CP.

    Args:
        num_reqs: Number of requests in the batch.
        query_start_loc: [num_reqs + 1] tensor of cumulative token counts.
        positions: [num_tokens] tensor of global token positions.
        block_table: [max_num_reqs, max_num_blocks_per_req] block table (GPU).
        block_table_stride: Stride of dim-0 in the block table.
        block_size: Number of slots per physical KV cache block.
        total_cp_world_size: Product of DCP and PCP world sizes.
        cp_kv_cache_interleave_size: CP interleave factor.
        out: [num_tokens] output tensor for virtual slot IDs.
    """
    _compute_re_slot_mapping_kernel[(num_reqs,)](
        positions.shape[0],
        query_start_loc,
        positions,
        block_table,
        block_table_stride,
        block_size,
        out,
        TOTAL_CP_WORLD_SIZE=total_cp_world_size,
        CP_KV_CACHE_INTERLEAVE_SIZE=cp_kv_cache_interleave_size,
        BLOCK_SIZE=1024,
    )
