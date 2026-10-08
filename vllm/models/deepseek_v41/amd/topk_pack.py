# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch packing of the DeepSeek V4.1 decode top-k into global slots.

compute_global_topk_ragged_indices_and_indptr in vLLM's V4.1 ROCm model runs
once a step for each index source layer, 8 times a step for
DeepSeek-V4.1-Flash. It launched 4 kernels: the per-row count of valid top-k
entries, the zero fill and the cumsum of indptr, and the mapping of the
entries through the compressed cache's block table. On MI325X they took
17.4 us of each index layer at 128k context, inside the decode graph.

_topk_pack_kernel writes the same 3 tensors with one program: the row
lengths (0 for a padded token), indptr, and the global slots of the first
length entries of each row (-1 for a negative entry). It does not write the
ragged entries past indptr[-1], which the old path also left uninitialized.

On gfx942 with VLLM_ROCM_MONO_DECODE=1 (``dsv41_gfx942.enabled()``), batches
of up to MAX_ROWS rows use it. tests/kernels/test_dsv41_gfx942_decode_meta.py
compares it with the old function.
"""

import torch

from vllm.model_executor.layers.dsv41_gfx942 import enabled
from vllm.triton_utils import tl, triton

# One program holds a ROWS x topk tile of int32. 16 rows of top-512 are 16
# values per thread with 8 warps.
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
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    in_batch = rows < num_tokens
    in_row = cols < topk
    local = tl.load(
        topk_ptr + rows[:, None] * topk_stride + cols[None, :],
        mask=in_batch[:, None] & in_row[None, :],
        other=-1,
    )
    req = tl.load(token_to_req_ptr + rows, mask=in_batch, other=0)
    token_ok = tl.load(is_valid_ptr + rows, mask=in_batch, other=0)

    # A row's length counts its non-negative entries, as
    # _compute_topk_lens_kernel does, and is 0 for a padded token.
    count = tl.sum((local >= 0).to(tl.int32), axis=1)
    lens = tl.where(token_ok, count, 0)
    tl.store(lens_ptr + rows, lens, mask=in_batch)
    ends = tl.cumsum(lens, axis=0)
    tl.store(indptr_ptr, 0)
    tl.store(indptr_ptr + 1 + rows, ends, mask=in_batch)

    # As in _pack_global_topk_ragged_kernel, row t writes its first lens[t]
    # entries in place, and a negative entry among them becomes -1.
    take = in_batch[:, None] & (cols[None, :] < lens[:, None]) & in_row[None, :]
    ok = take & (local >= 0)
    blk = tl.load(
        block_table_ptr + req[:, None] * block_table_stride + local // block_size,
        mask=ok,
        other=0,
    )
    slots = tl.where(ok, blk * block_size + local % block_size, -1)
    tl.store(ragged_ptr + (ends - lens)[:, None] + cols[None, :], slots, mask=take)


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
    _topk_pack_kernel[(1,)](
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
        num_warps=8,
    )
    return ragged, indptr, lens


def pack(
    topk_indices,
    token_to_req_indices,
    block_table,
    block_size,
    is_valid_token,
):
    """The fused packing, or None when the batch takes the old
    compute_global_topk_ragged_indices_and_indptr."""
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
