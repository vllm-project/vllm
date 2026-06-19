# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for sparse MLA backends."""

import torch

from vllm.triton_utils import tl, triton


# Kernel with prefill workspace support and valid count tracking
@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    valid_count_ptr,  # int32 [num_tokens] - output valid count per row
    prefill_request_id_ptr,  # int32 [num_tokens], -1 for decode, >=0 for prefill
    workspace_starts_ptr,  # int32 [num_prefill_reqs+1] or nullptr
    # shapes (compile-time where possible)
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    HAS_PREFILL: tl.constexpr,
    COUNT_VALID: tl.constexpr,  # whether to count valid indices
    # strides (in elements)
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    # program_id(0) -> token_id (row)
    # program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token (no mask: grid is exact)
    req = tl.load(req_id_ptr + token_id)

    # Load token indices for this tile
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # Only token == -1 should propagate as -1
    is_invalid_tok = tok < 0
    is_prefill = False
    if HAS_PREFILL:
        prefill_req_id = tl.load(prefill_request_id_ptr + token_id)
        is_prefill = prefill_req_id >= 0
    # Compute block id and in-block offset
    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    # Guard block_table access
    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    is_invalid_tok |= ~valid_block
    base = tl.load(bt_ptr, mask=valid_block & ~is_prefill, other=0)
    out_val = base * BLOCK_SIZE + inblock_off

    # Override with prefill output if prefill is enabled
    if HAS_PREFILL:
        workspace_start = tl.load(
            workspace_starts_ptr + prefill_req_id, mask=is_prefill, other=0
        )
        prefill_out = workspace_start + tok
        out_val = tl.where(is_prefill, prefill_out, out_val)
    out_val = tl.where(is_invalid_tok, -1, out_val)

    # Store results
    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)

    # Count valid indices in this tile and atomically add to row total
    if COUNT_VALID:
        tile_valid_count = tl.sum((~is_invalid_tok).to(tl.int32))
        tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,  # int32 [num_tokens]
    block_table: torch.Tensor,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
    HAS_PREFILL_WORKSPACE: bool = False,
    prefill_workspace_request_ids: torch.Tensor | None = None,
    prefill_workspace_starts: torch.Tensor | None = None,
    return_valid_counts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.

    When HAS_PREFILL_WORKSPACE is True, prefill tokens are mapped to workspace offsets
    instead of global cache slots. prefill_workspace_request_ids and
    prefill_workspace_starts must be provided.

    prefill_workspace_request_ids: int32 [num_tokens], -1 for decode else
        prefill request index (maps to prefill_workspace_starts)
    prefill_workspace_starts: int32 [num_prefills], 0-indexed workspace
        starts for each prefill request

    When return_valid_counts is True, also returns the count of valid (non -1)
    indices per row, computed during the same kernel pass (no extra overhead).
    """
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by BLOCK_N ({BLOCK_N})"
    )

    if HAS_PREFILL_WORKSPACE:
        assert prefill_workspace_request_ids is not None
        assert prefill_workspace_starts is not None
        assert prefill_workspace_request_ids.dtype == torch.int32
        assert prefill_workspace_starts.dtype == torch.int32

    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    # Allocate valid count buffer if needed (must be zero-initialized for atomics)
    valid_counts: torch.Tensor | None = None
    if return_valid_counts:
        valid_counts = torch.zeros(
            num_tokens, dtype=torch.int32, device=token_indices.device
        )

    # Strides in elements
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    # Prepare prefill pointers
    if HAS_PREFILL_WORKSPACE:
        assert prefill_workspace_request_ids is not None  # for mypy
        assert prefill_workspace_starts is not None  # for mypy
        assert prefill_workspace_request_ids.is_contiguous()
        assert prefill_workspace_starts.is_contiguous()

    # Exact 2D grid: tokens × column tiles
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        valid_counts,
        prefill_workspace_request_ids,
        prefill_workspace_starts,
        # shapes / constexprs
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        HAS_PREFILL_WORKSPACE,
        return_valid_counts,
        # strides
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )

    if return_valid_counts:
        assert valid_counts is not None
        return out, valid_counts
    return out


@triton.jit
def _convert_req_index_to_global_index_dcp_kernel(
    req_id_ptr,
    block_table_ptr,
    token_indices_ptr,
    out_ptr,
    valid_count_ptr,
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    COUNT_VALID: tl.constexpr,
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    req = tl.load(req_id_ptr + token_id)
    tok = tl.load(token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1)

    is_invalid_tok = tok < 0
    owning_rank = tok % DCP_SIZE
    is_remote = owning_rank != DCP_RANK

    local_idx = tok // DCP_SIZE
    block_id = local_idx // BLOCK_SIZE
    inblock_off = local_idx % BLOCK_SIZE

    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    is_invalid_tok |= ~valid_block | is_remote

    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    base = tl.load(bt_ptr, mask=valid_block & ~is_remote & ~is_invalid_tok, other=0)
    out_val = base * BLOCK_SIZE + inblock_off
    out_val = tl.where(is_invalid_tok, -1, out_val)

    tl.store(out_ptr + token_id * out_stride0 + indice_id * out_stride1, out_val)

    if COUNT_VALID:
        tile_valid_count = tl.sum((~is_invalid_tok).to(tl.int32))
        tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)


def _compact_valid_to_front(
    slots: torch.Tensor,
    valid_counts: torch.Tensor,
) -> torch.Tensor:
    is_invalid = (slots < 0).to(torch.int32)
    sort_idx = torch.argsort(is_invalid, dim=-1, stable=True)
    return torch.gather(slots, -1, sort_idx)


def triton_filter_and_convert_dcp_index(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int = 1,
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,
    return_valid_counts: bool = False,
    compact_valid_to_front: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Filter global per-request indices to this DCP rank's local slots."""
    assert dcp_size >= 1
    assert 0 <= dcp_rank < dcp_size
    assert cp_kv_cache_interleave_size == 1, (
        "DCP sparse index conversion currently supports only "
        "cp_kv_cache_interleave_size=1."
    )
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0

    if dcp_size == 1:
        return triton_convert_req_index_to_global_index(
            req_id,
            block_table,
            token_indices,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_TOPK_TOKENS=NUM_TOPK_TOKENS,
            BLOCK_N=BLOCK_N,
            return_valid_counts=return_valid_counts,
        )

    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    valid_counts: torch.Tensor | None = None
    if return_valid_counts:
        valid_counts = torch.zeros(
            num_tokens, dtype=torch.int32, device=token_indices.device
        )

    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    _convert_req_index_to_global_index_dcp_kernel[(num_tokens, tiles_per_row)](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        valid_counts,
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        dcp_size,
        dcp_rank,
        return_valid_counts,
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )

    if compact_valid_to_front and return_valid_counts:
        assert valid_counts is not None
        out = _compact_valid_to_front(out, valid_counts)

    if return_valid_counts:
        assert valid_counts is not None
        return out, valid_counts
    return out
