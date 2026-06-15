# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _compressed_slot_mapping_kernel(
    # [num_tokens]
    slot_mapping_ptr,
    # [num_reqs + 1]
    query_start_loc_ptr,
    # [num_reqs]
    seq_lens_ptr,
    # [num_reqs, max_num_blocks]
    block_table_ptr,
    block_table_stride,
    block_size,
    total_cp_world_size,
    total_cp_rank,
    cp_kv_cache_interleave_size,
    COMPRESS_RATIO: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    start_pos = seq_len - query_len

    for i in range(0, query_len, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < query_len

        pos = start_pos + i + tl.arange(0, TRITON_BLOCK_SIZE)
        is_valid = mask & (pos >= 0) & ((pos + 1) % COMPRESS_RATIO == 0)
        pos_after_compress = pos // COMPRESS_RATIO

        virtual_block_size = block_size * total_cp_world_size
        block_ids = pos_after_compress // virtual_block_size
        block_numbers = tl.load(
            block_table_ptr + batch_idx * block_table_stride + block_ids,
            mask=is_valid,
        )
        virtual_block_offsets = pos_after_compress - block_ids * virtual_block_size
        is_local = (
            virtual_block_offsets // cp_kv_cache_interleave_size
        ) % total_cp_world_size == total_cp_rank
        local_block_offsets = (
            virtual_block_offsets // (total_cp_world_size * cp_kv_cache_interleave_size)
        ) * cp_kv_cache_interleave_size + (
            virtual_block_offsets % cp_kv_cache_interleave_size
        )
        slot_ids = block_numbers * block_size + local_block_offsets

        slot_ids = tl.where(is_valid & is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + query_start + offset, slot_ids, mask=mask)


def get_compressed_slot_mapping(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    total_cp_world_size: int = 1,
    total_cp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        # Guard: for padded / invalid sequences.
        # Negative positions produce bogus block indices that lead to illegal memory
        # accesses inside the block_table load.
        # NOTE: Fill -1 to the whole tensor, not just the first `num_tokens`.
        out.fill_(-1)
        slot_mapping = out[:num_tokens]
    else:
        slot_mapping = torch.full(
            (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
        )

    num_reqs = block_table.shape[0]
    _compressed_slot_mapping_kernel[(num_reqs,)](
        slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        block_table.stride(0),
        block_size,
        total_cp_world_size,
        total_cp_rank,
        cp_kv_cache_interleave_size,
        compress_ratio,
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=1024,
    )
    return slot_mapping
