# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch packing of the DeepSeek V4.1 decode top-k into global slots.

compute_global_topk_ragged_indices_and_indptr in vLLM's V4.1 ROCm model runs
once a step for each index source layer, 8 times a step for
DeepSeek-V4.1-Flash. For a batch that this file does not take, it calls
_compute_global_topk_ragged_indices_and_indptr_old, which launches 4 kernels:
the per-row count of valid top-k entries, the zero fill and the cumsum of
indptr, and the mapping of the entries through the compressed cache's block
table. On MI325X they took 17.4 us of each index layer at 128k context,
inside the decode graph.

_topk_pack_kernel writes the same 3 tensors with one launch, a program per
row: the row lengths (0 for a padded token), indptr, and the global slots of
each row's non-negative entries, in order and without holes. It does not
write the ragged entries past indptr[-1], which
_compute_global_topk_ragged_indices_and_indptr_old also leaves uninitialized.

On gfx942 with --monokernel (`ops.enabled()`), batches
of up to MAX_ROWS rows use it. tests/kernels/models/amd/test_dsv41_gfx942_decode_meta.py
compares it with _compute_global_topk_ragged_indices_and_indptr_old.
"""

import torch

from vllm.models.deepseek_v41.amd.ops import enabled
from vllm.triton_utils import tl, triton

MAX_ROWS = 16


@triton.jit(do_not_specialize=["num_tokens", "topk_stride", "block_table_stride"])
def _topk_pack_kernel(
    ragged_ptr,
    indptr_ptr,
    lens_ptr,
    topk_ptr,
    topk_stride,
    is_valid_ptr,
    token_to_req_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    num_tokens,
    topk,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    in_batch = rows < num_tokens
    in_row = cols < topk

    # A row's length counts its non-negative entries, as
    # _compute_topk_lens_kernel does, and is 0 for a padded token. Each
    # program counts every row to find where its own row starts.
    local = tl.load(
        topk_ptr + rows[:, None] * topk_stride + cols[None, :],
        mask=in_batch[:, None] & in_row[None, :],
        other=-1,
    )
    token_ok = tl.load(is_valid_ptr + rows, mask=in_batch, other=0)
    lens = tl.where(token_ok, tl.sum((local >= 0).to(tl.int32), axis=1), 0)
    start = tl.sum(tl.where(rows < row, lens, 0), axis=0)
    length = tl.sum(tl.where(rows == row, lens, 0), axis=0)
    tl.store(lens_ptr + row, length)
    tl.store(indptr_ptr + 1 + row, start + length)
    if row == 0:
        tl.store(indptr_ptr, 0)

    # As in _pack_global_topk_ragged_kernel, holes are compacted out, so every
    # slot inside the row's indptr range is valid.
    local = tl.load(topk_ptr + row * topk_stride + cols, mask=in_row, other=-1)
    req = tl.load(token_to_req_ptr + row)
    take = tl.load(is_valid_ptr + row) & in_row & (local >= 0)
    blk = tl.load(
        block_table_ptr + req * block_table_stride + local // block_size,
        mask=take,
        other=0,
    )
    rank = tl.cumsum(take.to(tl.int32), axis=0) - 1
    tl.store(
        ragged_ptr + start + rank, blk * block_size + local % block_size, mask=take
    )


def launch(
    topk_indices,
    token_to_req_indices,
    block_table,
    block_size,
    is_valid_token,
    poison=False,
):
    """(global_topk_ragged, topk_indptr, topk_lens) from one launch, as
    compute_global_topk_ragged_indices_and_indptr returns them. poison fills
    the outputs with -7 first, so that a test sees every value the launch
    does not write, even when the allocator hands back the last call's
    memory."""
    num_tokens, topk = topk_indices.shape
    dev = topk_indices.device
    lens = torch.empty(num_tokens, dtype=torch.int32, device=dev)
    indptr = torch.empty(num_tokens + 1, dtype=torch.int32, device=dev)
    ragged = torch.empty(num_tokens * topk, dtype=torch.int32, device=dev)
    if poison:
        for t in (lens, indptr, ragged):
            t.fill_(-7)
    _topk_pack_kernel[(num_tokens,)](
        ragged,
        indptr,
        lens,
        topk_indices,
        topk_indices.stride(0),
        is_valid_token,
        token_to_req_indices,
        block_table,
        block_table.stride(0),
        block_size,
        num_tokens,
        topk,
        ROWS=triton.next_power_of_2(num_tokens),
        BLOCK=triton.next_power_of_2(topk),
        num_warps=4,
    )
    return ragged, indptr, lens


def pack(
    topk_indices,
    token_to_req_indices,
    block_table,
    block_size,
    is_valid_token,
):
    """The fused packing, or None when the batch takes
    _compute_global_topk_ragged_indices_and_indptr_old."""
    if not enabled():
        return None
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1)
    num_tokens = topk_indices.shape[0]
    if (
        not 0 < num_tokens <= MAX_ROWS
        or topk_indices.stride(1) != 1
        or topk_indices.dtype != torch.int32
        or is_valid_token.shape[0] != num_tokens
    ):
        return None
    return launch(
        topk_indices,
        token_to_req_indices,
        block_table,
        block_size,
        is_valid_token,
    )
