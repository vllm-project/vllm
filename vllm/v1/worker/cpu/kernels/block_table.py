# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch implementations of the block table, staged write and KV zeroing
Triton kernels."""

import ctypes

import torch

from vllm.v1.worker.cpu.kernels.utils import ptr_view, token_to_batch_idx


def gather_block_tables(
    batch_idx_to_req_idx: torch.Tensor,
    src_block_table_ptrs: torch.Tensor,
    dst_block_table_ptrs: torch.Tensor,
    block_table_strides: torch.Tensor,
    num_blocks_ptr: torch.Tensor,
    num_blocks_stride: int,
    num_reqs: int,
    BLOCK_SIZE: int,
    *,
    grid: tuple[int, int],
) -> None:
    num_groups, num_reqs_padded = grid
    max_num_reqs = num_blocks_ptr.shape[1]
    req = batch_idx_to_req_idx[:num_reqs].long()
    for group_id, (src_addr, dst_addr, stride) in enumerate(
        zip(
            src_block_table_ptrs.tolist(),
            dst_block_table_ptrs.tolist(),
            block_table_strides.tolist(),
        )
    ):
        src = ptr_view(src_addr, torch.int32, max_num_reqs * stride)
        dst = ptr_view(dst_addr, torch.int32, num_reqs_padded * stride)
        src = src.view(max_num_reqs, stride)
        dst = dst.view(num_reqs_padded, stride)
        dst[num_reqs:] = 0
        if num_reqs == 0:
            continue
        num_blocks = num_blocks_ptr[group_id, req]
        width = int(num_blocks.max())
        in_range = torch.arange(width) < num_blocks.unsqueeze(1)
        dst[:num_reqs, :width] = torch.where(
            in_range, src[req, :width], dst[:num_reqs, :width]
        )


def compute_slot_mappings(
    max_num_tokens: int,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    pos: torch.Tensor,
    block_table_ptrs: torch.Tensor,
    block_table_strides: torch.Tensor,
    block_sizes: torch.Tensor,
    kernel_block_sizes: torch.Tensor,
    slot_mapping_enabled: torch.Tensor,
    dcp_sharded: torch.Tensor,
    slot_mappings_ptr: torch.Tensor,
    slot_mappings_stride: int,
    cp_rank: int,
    CP_SIZE: int,
    CP_INTERLEAVE: int,
    PAD_ID: int,
    TRITON_BLOCK_SIZE: int,
) -> None:
    num_reqs = idx_mapping.shape[0]
    query_start_loc = query_start_loc[: num_reqs + 1]
    num_tokens = int(query_start_loc[-1])
    batch_idx = token_to_batch_idx(query_start_loc, num_tokens)
    # idx_mapping == -1 marks a dummy (or CUDA-graph padding) request that owns
    # no blocks: never read its block-table row and emit PAD for its tokens.
    req = idx_mapping.long()[batch_idx]
    is_real_req = req >= 0
    req = req.clamp_min(0)
    num_rows = int(req.max()) + 1 if num_tokens else 0
    positions = pos[:num_tokens].long()

    for group_id, (addr, stride) in enumerate(
        zip(block_table_ptrs.tolist(), block_table_strides.tolist())
    ):
        slot_mapping = slot_mappings_ptr[group_id]
        slot_mapping[num_tokens:max_num_tokens] = PAD_ID
        if num_tokens == 0:
            continue
        kv_block_size = int(block_sizes[group_id])
        kernel_block_size = int(kernel_block_sizes[group_id])
        if not bool(slot_mapping_enabled[group_id]):
            slot_mapping[:num_tokens] = PAD_ID
            continue

        is_local = is_real_req
        local_positions = positions
        if CP_SIZE != 1 and bool(dcp_sharded[group_id]):
            virtual_block_size = kv_block_size * CP_SIZE
            virtual_block_offsets = positions % virtual_block_size
            is_local = is_local & (
                virtual_block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            )
            local_offsets = (
                virtual_block_offsets // (CP_INTERLEAVE * CP_SIZE) * CP_INTERLEAVE
                + virtual_block_offsets % CP_INTERLEAVE
            )
            local_positions = (
                positions // virtual_block_size * kv_block_size + local_offsets
            )

        block_table = ptr_view(addr, torch.int32, num_rows * stride)
        block_idx = local_positions // kernel_block_size
        block_numbers = block_table[req * stride + block_idx]
        slot_ids = (
            block_numbers * kernel_block_size + local_positions % kernel_block_size
        )
        slot_mapping[:num_tokens] = torch.where(is_local, slot_ids, PAD_ID)


def apply_write(
    output_ptr: torch.Tensor,
    output_stride: int | torch.Tensor,
    write_indices_ptr: torch.Tensor,
    write_starts_ptr: torch.Tensor,
    write_contents_ptr: torch.Tensor,
    write_cu_lens_ptr: torch.Tensor,
    write_group_ids_ptr: torch.Tensor | None,
    BLOCK_SIZE: int,
    MULTI_GROUP: bool,
) -> None:
    num_writes = write_indices_ptr.shape[0]
    cu_lens = write_cu_lens_ptr[:num_writes].long()
    lens = torch.diff(cu_lens, prepend=cu_lens.new_zeros(1))
    num_elems = int(cu_lens[-1])
    write_idx = token_to_batch_idx(
        torch.cat((cu_lens.new_zeros(1), cu_lens)), num_elems
    )
    elem_offset = torch.arange(num_elems) - (cu_lens - lens)[write_idx]
    rows = write_indices_ptr[:num_writes].long()[write_idx]
    cols = write_starts_ptr[:num_writes].long()[write_idx] + elem_offset
    contents = write_contents_ptr[:num_elems]

    if not MULTI_GROUP:
        assert isinstance(output_stride, int)
        output_ptr.view(-1)[rows * output_stride + cols] = contents.to(output_ptr.dtype)
        return

    assert isinstance(output_stride, torch.Tensor)
    assert write_group_ids_ptr is not None
    group_ids = write_group_ids_ptr[:num_writes].long()[write_idx]
    addrs = output_ptr.tolist()
    strides = output_stride.tolist()
    for group_id in group_ids.unique().tolist():
        in_group = group_ids == group_id
        offsets = rows[in_group] * strides[group_id] + cols[in_group]
        out = ptr_view(addrs[group_id], torch.int32, int(offsets.max()) + 1)
        out[offsets] = contents[in_group].to(torch.int32)


def zero_kv_blocks(
    seg_addrs_ptr: torch.Tensor,
    seg_block_strides_ptr: torch.Tensor,
    seg_page_sizes_ptr: torch.Tensor,
    block_ids_ptr: torch.Tensor,
    BLOCK_SIZE: int,
) -> None:
    # Strides and page sizes are in int32 elements.
    block_ids = block_ids_ptr.tolist()
    for addr, block_stride, page_size in zip(
        seg_addrs_ptr.tolist(),
        seg_block_strides_ptr.tolist(),
        seg_page_sizes_ptr.tolist(),
    ):
        for block_id in block_ids:
            ctypes.memset(addr + block_id * block_stride * 4, 0, page_size * 4)


def dcp_local_seq_lens(
    out_ptr: torch.Tensor,
    seq_lens_ptr: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    cp_interleave: int,
    num_reqs: int,
    max_num_reqs: int,
    BLOCK_SIZE: int,
) -> None:
    # Distribute KV cache among different ranks, in a round-robin manner.
    seq_lens = seq_lens_ptr[:num_reqs]
    rounds = seq_lens // (dcp_size * cp_interleave)
    remainder = seq_lens % (dcp_size * cp_interleave)
    remainder = (remainder - dcp_rank * cp_interleave).clamp(0, cp_interleave)
    out_ptr[:num_reqs] = rounds * cp_interleave + remainder
    out_ptr[num_reqs:max_num_reqs] = 0
