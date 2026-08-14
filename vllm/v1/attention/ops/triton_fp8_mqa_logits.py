# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary gfx942 fallback for AITER's fp8_mqa_logits kernel.

This module vendors AITER's Triton fp8_mqa_logits kernel with the gfx942
tile-size workaround from ROCm/aiter#3257. It is used only while vLLM's
pinned AITER version lacks that fix.

TODO: Remove this vendored copy once vLLM pins an AITER version that includes
ROCm/aiter#3257 bugfix for gfx942.
"""

import torch

from vllm.triton_utils import tl, triton

# gfx942 (MI300X) has 64 KiB of LDS per CU. We accept the default
# (BLOCK_KV=128, num_stages=2) tile only when *both* of these hold:
#
# 1. Occupancy gate. With waves_per_eu=2 and num_warps=4 we target two
#    workgroups co-resident on a CU -> per-WG LDS budget = 32 KiB. Triton
#    keeps Q in registers (loop-invariant) and the fp32 scores accumulator
#    in VGPRs (heavy VALU), so only the double-buffered KV tile is
#    expected to live in LDS. A 0.9 safety factor leaves headroom for any
#    LDS overhead the compiler may add.
#
# 2. Hardware ceiling. Defensive upper bound that also counts Q and
#    scores against the 64 KiB CU limit, in case a Triton version (older
#    or future) decides to spill them to LDS. False positives here only
#    shrink the tile; false negatives are JIT-aborts, so we lean
#    conservative.
_GFX942_CU_LDS_BYTES = 64 * 1024
_GFX942_PER_WG_LDS_BUDGET_BYTES = _GFX942_CU_LDS_BYTES * 9 // 20  # ~28.8 KiB


def _gfx942_default_tile_fits_lds(num_heads: int, head_size: int) -> bool:
    """Return True iff (BLOCK_KV=128, num_stages=2) fits in MI300X LDS."""
    BLOCK_KV = 128
    NUM_STAGES = 2
    kv_bytes = head_size * BLOCK_KV * NUM_STAGES
    scores_bytes = num_heads * BLOCK_KV * 4
    q_bytes = num_heads * head_size
    fits_occupancy = kv_bytes < _GFX942_PER_WG_LDS_BUDGET_BYTES
    fits_hardware = q_bytes + kv_bytes + scores_bytes <= _GFX942_CU_LDS_BYTES
    return fits_occupancy and fits_hardware


@triton.jit
def _fp8_mqa_logits_kernel(
    Q_ptr,  # fp8e4m3 [seq_len, H, D]
    KV_ptr,  # fp8e4m3 [seq_len_kv, D]
    kv_scales_ptr,  # fp32 [seq_len_kv]
    weights_ptr,  # fp32 [seq_len, H]
    cu_start_ptr,  # int32 [seq_len]
    cu_end_ptr,  # int32 [seq_len]
    logits_ptr,  # fp32 [seq_len, seq_len_kv]
    seq_len,
    seq_len_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    # strides
    stride_q_s: tl.int64,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_s: tl.int64,
    stride_kv_d: tl.constexpr,
    stride_w_s: tl.int64,
    stride_w_h: tl.constexpr,
    stride_logits_s: tl.int64,
    stride_logits_k: tl.int64,
    # block sizes
    BLOCK_KV: tl.constexpr,
):
    row_id = tl.program_id(0)
    # go from larger to smaller in terms of work
    # to reduce the tail effect
    row_id = tl.num_programs(0) - row_id - 1
    tl.assume(row_id >= 0)
    tl.assume(stride_q_s > 0)
    tl.assume(stride_q_h > 0)
    tl.assume(stride_q_d > 0)
    tl.assume(stride_kv_s > 0)
    tl.assume(stride_kv_d > 0)
    tl.assume(stride_w_s > 0)
    tl.assume(stride_w_h > 0)

    logits_row_ptrs = logits_ptr + row_id * stride_logits_s

    h_inds = tl.arange(0, NUM_HEADS)[:, None]
    d_inds = tl.arange(0, HEAD_SIZE)

    # load Q[BLOCK_Q, NUM_HEADS, HEAD_SIZE]
    q_ptrs = (
        Q_ptr + row_id * stride_q_s + h_inds * stride_q_h + d_inds[None, :] * stride_q_d
    )

    q_block = tl.load(q_ptrs, cache_modifier=".cg")
    w_ptrs = weights_ptr + row_id * stride_w_s + h_inds * stride_w_h
    w_block = tl.load(w_ptrs, cache_modifier=".cg").to(tl.float32)

    # Load start/end for each row in this block
    start_ind = tl.load(cu_start_ptr + row_id)
    end_ind = tl.load(cu_end_ptr + row_id)

    start_ind = tl.maximum(start_ind, 0)
    end_ind = tl.minimum(end_ind, seq_len_kv)
    shifted_end = end_ind - start_ind
    shifted_unmasked_end = shifted_end // BLOCK_KV * BLOCK_KV

    kv_col_offsets = tl.arange(0, BLOCK_KV) + start_ind
    kv_ptrs = (
        KV_ptr + kv_col_offsets[None, :] * stride_kv_s + d_inds[:, None] * stride_kv_d
    )

    kv_scales_ptrs = kv_scales_ptr + kv_col_offsets

    logits_ptrs = logits_row_ptrs + kv_col_offsets * stride_logits_k

    # Loop over KV tiles
    for _ in tl.range(0, shifted_unmasked_end, BLOCK_KV):
        kv_block = tl.load(kv_ptrs)
        kv_scales = tl.load(kv_scales_ptrs)

        # [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV]
        scores = tl.dot(q_block, kv_block, input_precision="ieee")
        # Multiply by kv_scales (broadcast along rows)
        scores = scores * kv_scales[None, :]
        # ReLU
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        # [NUM_HEADS, BLOCK_KV] -> [BLOCK_KV, ]
        scores = tl.sum(scores, axis=0)
        tl.store(logits_ptrs, scores)

        kv_ptrs += BLOCK_KV * stride_kv_s
        kv_scales_ptrs += BLOCK_KV
        logits_ptrs += BLOCK_KV * stride_logits_k
        kv_col_offsets += BLOCK_KV

    # masked load
    kv_col_mask = kv_col_offsets < end_ind
    kv_block = tl.load(kv_ptrs, mask=kv_col_mask[None, :], other=0.0)
    kv_scales = tl.load(kv_scales_ptrs, mask=kv_col_mask, other=0.0)

    # [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV]
    scores = tl.dot(q_block, kv_block, input_precision="ieee")
    # Multiply by kv_scales (broadcast along rows)
    scores = scores * kv_scales[None, :]
    # ReLU
    scores = tl.maximum(scores, 0.0)
    scores = scores * w_block
    # [NUM_HEADS, BLOCK_KV] -> [BLOCK_KV, ]
    scores = tl.sum(scores, axis=0)
    # masked store
    in_window = (kv_col_offsets >= start_ind) & (kv_col_offsets < end_ind)
    tl.store(logits_ptrs, scores, mask=in_window)


def fp8_mqa_logits_gfx942(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits on MI300X (gfx942) using the vendored kernel.

    Drop-in replacement for ``aiter.ops.triton.attention.fp8_mqa_logits.
    fp8_mqa_logits`` on MI300X. Selects ``(BLOCK_KV, num_stages)`` based on
    whether the default tile fits within the 64 KiB LDS budget of a gfx942
    CU (see module docstring).

    Args:
        q: Query tensor of shape ``[M, H, D]``, FP8 dtype.
        k_fp8: Key tensor of shape ``[N, D]``, FP8 dtype.
        kv_scales: K scales of shape ``[N]`` (or ``[N, 1]`` -- viewed as
            ``[N]``), float32.
        weights: Per-head weights of shape ``[M, H]``, float32.
        cu_starts: Start indices (inclusive) of shape ``[M]``, int32.
        cu_ends: End indices (exclusive) of shape ``[M]``, int32.

    Returns:
        Logits of shape ``[M, N]``, float32 -- positions outside
        ``[cu_starts[i], cu_ends[i])`` for row ``i`` are pre-filled with
        ``-inf`` so the caller can run a top-k without masking.
    """
    seq_len, num_heads, head_size = q.shape
    seq_len_kv = k_fp8.shape[0]
    assert num_heads & (num_heads - 1) == 0, (
        f"num_heads must be a power of two (got {num_heads})"
    )
    assert head_size & (head_size - 1) == 0, (
        f"head_size must be a power of two (got {head_size})"
    )

    # The kernel walks ``kv_scales`` as a 1-D contiguous array of size N
    # (it indexes by ``kv_scales_ptr + kv_col_offsets``). The vLLM caller
    # passes a ``[N, 4]`` uint8 view-cast-to-float32 which lands as
    # ``[N, 1]`` contiguous -- byte-identical to ``[N]`` -- but flatten
    # explicitly to keep the kernel's pointer arithmetic intent clear.
    kv_scales_1d = kv_scales.reshape(-1)

    # Initialise with -inf so positions outside [cu_starts, cu_ends) read
    # as ``-inf`` after the masked store path -- this matches AITER's
    # ``fp8_mqa_logits`` semantics and is what the top-k consumer expects.
    logits = torch.full(
        (seq_len, seq_len_kv),
        fill_value=-float("inf"),
        dtype=torch.float32,
        device=q.device,
    )

    if _gfx942_default_tile_fits_lds(num_heads, head_size):
        block_kv = 128
        num_stages = 2
    else:
        # DSv4 sparse indexer (NUM_HEADS=64, HEAD_SIZE=128) lands here:
        # default tile spills past gfx942's 64 KiB LDS budget. (64, 1)
        # needs ~33 KiB and clears the per-WG budget with margin.
        block_kv = 64
        num_stages = 1

    # heuristic for MFMA instruction shape, identical to AITER's choice
    matrix_instr_nonkdim = 32
    if seq_len <= 1024:
        matrix_instr_nonkdim = 16

    stride_q_s, stride_q_h, stride_q_d = q.stride()
    stride_kv_s, stride_kv_d = k_fp8.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()

    _fp8_mqa_logits_kernel[(seq_len,)](
        Q_ptr=q,
        KV_ptr=k_fp8,
        kv_scales_ptr=kv_scales_1d,
        weights_ptr=weights,
        cu_start_ptr=cu_starts,
        cu_end_ptr=cu_ends,
        logits_ptr=logits,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        stride_q_s=stride_q_s,
        stride_q_h=stride_q_h,
        stride_q_d=stride_q_d,
        stride_kv_s=stride_kv_s,
        stride_kv_d=stride_kv_d,
        stride_w_s=stride_w_s,
        stride_w_h=stride_w_h,
        stride_logits_s=stride_logits_s,
        stride_logits_k=stride_logits_k,
        BLOCK_KV=block_kv,
        num_warps=4,
        num_stages=num_stages,
        waves_per_eu=2,
        matrix_instr_nonkdim=matrix_instr_nonkdim,
    )

    return logits
