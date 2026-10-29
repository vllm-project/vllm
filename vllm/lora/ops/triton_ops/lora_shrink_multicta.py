# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental deterministic LoRA shrink with an in-cluster K reduction."""

import torch

from vllm.triton_utils import gl, gluon, triton


@gluon.jit(do_not_specialize=["M", "scaling"])
def _lora_shrink_multicta_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    sorted_token_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    M,
    N: gl.constexpr,
    K: gl.constexpr,
    scaling,
    lora_d0_stride: gl.constexpr,
    lora_d1_stride: gl.constexpr,
    lora_d2_stride: gl.constexpr,
    NUM_SLICES: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SPLIT_K: gl.constexpr,
    CGA_LAYOUT: gl.constexpr,
):
    pid = gl.program_id(0)
    pid_m = pid // gl.cdiv(N, BLOCK_N)
    pid_n = pid % gl.cdiv(N, BLOCK_N)
    slice_id = gl.program_id(1)
    lora_idx = gl.program_id(2)
    lora_id = gl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return
    lora_m_size = gl.load(num_tokens_per_lora + lora_idx)
    if pid_m * BLOCK_M >= lora_m_size:
        return
    token_start = gl.load(lora_token_start_loc + lora_idx)

    # The leading dimension represents independent K splits, one per CTA.
    a_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 4],
        [1, 4, 8],
        [1, gl.num_warps(), 1],
        [2, 1, 0],
        cga_layout=CGA_LAYOUT,
    )
    b_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4, 1],
        [1, 8, 4],
        [1, 1, gl.num_warps()],
        [1, 2, 0],
        cga_layout=CGA_LAYOUT,
    )
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        [2, 0], [1, gl.num_warps() // 2, 2], [1, 16, 8], cga_layout=CGA_LAYOUT
    )
    sa = gl.arange(0, SPLIT_K, layout=gl.SliceLayout(1, gl.SliceLayout(2, a_layout)))
    ma = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, gl.SliceLayout(2, a_layout)))
    ka = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, gl.SliceLayout(1, a_layout)))
    sb = gl.arange(0, SPLIT_K, layout=gl.SliceLayout(1, gl.SliceLayout(2, b_layout)))
    kb = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, gl.SliceLayout(2, b_layout)))
    nb = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, gl.SliceLayout(1, b_layout)))
    row_offsets = pid_m * BLOCK_M + ma
    rows = gl.load(
        sorted_token_ids + token_start + row_offsets,
        row_offsets < lora_m_size,
        other=0,
    ).to(gl.int64)
    cols = pid_n * BLOCK_N + nb
    if NUM_SLICES == 1:
        weight_ptr = lora_ptr
    else:
        weight_ptr = gl.load(lora_ptr + slice_id).to(
            gl.pointer_type(input_ptr.dtype.element_ty)
        )
    weight_ptr += lora_id.to(gl.int64) * lora_d0_stride

    accumulator = gl.full((SPLIT_K, BLOCK_M, BLOCK_N), 0, gl.float32, mma_layout)
    for iteration in range(gl.cdiv(K, SPLIT_K * BLOCK_K)):
        a_k = sa[:, None, None] * BLOCK_K + ka[None, None, :]
        a_k += iteration * SPLIT_K * BLOCK_K
        b_k = sb[:, None, None] * BLOCK_K + kb[None, :, None]
        b_k += iteration * SPLIT_K * BLOCK_K
        a = gl.load(
            input_ptr + rows[None, :, None] * K + a_k,
            (row_offsets[None, :, None] < lora_m_size) & (a_k < K),
            other=0,
        )
        b = gl.load(
            weight_ptr + cols[None, None, :] * lora_d1_stride + b_k * lora_d2_stride,
            (cols[None, None, :] < N) & (b_k < K),
            other=0,
        )
        a = gl.convert_layout(a, gl.DotOperandLayout(0, mma_layout, 2))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, mma_layout, 2))
        accumulator = gl.nvidia.ampere.mma_v2(a, b, accumulator)

    reduce_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 2],
        [1, 4, 8],
        [1, gl.num_warps(), 1],
        [2, 1, 0],
        cga_layout=CGA_LAYOUT,
    )
    partials = gl.convert_layout(accumulator, reduce_layout)
    result = gl.sum(partials, axis=0)
    out_layout: gl.constexpr = gl.SliceLayout(0, reduce_layout)
    om = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, out_layout))
    on = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, out_layout))
    out_rows = gl.convert_layout(rows, gl.SliceLayout(1, out_layout))
    out_cols = pid_n * BLOCK_N + on
    out_offsets = (slice_id.to(gl.int64) * M + out_rows[:, None]) * N + out_cols[
        None, :
    ]
    gl.store(
        out_ptr + out_offsets,
        result * scaling,
        (pid_m * BLOCK_M + om[:, None] < lora_m_size) & (out_cols[None, :] < N),
    )


def _multicta_config(m: int, k: int, rank: int) -> tuple[int, int, int, int]:
    # Only M tiling changes with batch size; K arithmetic stays fixed.
    block_k = 128 if 256 < k <= 1024 else 256
    block_k = min(block_k, max(32, triton.next_power_of_2(k)))
    k_blocks = max(1, triton.cdiv(k, block_k))
    num_ctas = min(4 if rank >= 32 else 8, 1 << (k_blocks.bit_length() - 1))
    block_m = 16 if m < 128 else 32
    block_n = min(32, max(16, triton.next_power_of_2(rank)))
    return block_m, block_n, block_k, num_ctas


def launch_lora_shrink_multicta(
    inputs: torch.Tensor,
    lora_ptr: torch.Tensor,
    output: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    num_active_loras: int,
    scaling: float,
    lora_strides: tuple[int, int, int],
) -> None:
    m, k = inputs.shape
    nslices, _, rank = output.shape
    bm, bn, bk, num_ctas = _multicta_config(m, k, rank)
    cga_layout = tuple((1 << bit, 0, 0) for bit in range(num_ctas.bit_length() - 1))
    grid = (triton.cdiv(m, bm) * triton.cdiv(rank, bn), nslices, num_active_loras)
    _lora_shrink_multicta_kernel[grid](
        inputs,
        lora_ptr,
        output,
        sorted_token_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        m,
        rank,
        k,
        scaling,
        *lora_strides,
        nslices,
        bm,
        bn,
        bk,
        num_ctas,
        cga_layout,
        num_ctas=num_ctas,
        num_warps=2 if bm == 16 and bn == 16 else 4,
        enable_fp_fusion=False,
    )
