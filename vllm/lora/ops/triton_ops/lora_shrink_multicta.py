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
    M,
    N: gl.constexpr,
    K: gl.constexpr,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    scaling,
    lora_d0_stride: gl.constexpr,
    lora_d1_stride: gl.constexpr,
    lora_d2_stride: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SPLIT_K: gl.constexpr,
    SLICE_NUM: gl.constexpr,
    CGA_LAYOUT: gl.constexpr,
):
    cta_n_num = gl.cdiv(N, BLOCK_N)
    pid_m_n = gl.program_id(axis=0)
    pid_m = pid_m_n // cta_n_num
    pid_n = pid_m_n % cta_n_num
    slice_id = gl.program_id(axis=1)
    lora_idx = gl.program_id(axis=2)
    lora_id = gl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return
    lora_m_size = gl.load(num_tokens_per_lora + lora_idx)
    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        return
    lora_m_indices_start = gl.load(lora_token_start_loc + lora_idx)

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
    # A and B need separate K offsets because their layouts differ.
    offset_sk_a = gl.arange(
        0, SPLIT_K, layout=gl.SliceLayout(1, gl.SliceLayout(2, a_layout))
    )
    offset_m = cta_m_offset + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(0, gl.SliceLayout(2, a_layout))
    )
    offset_ak = gl.arange(
        0, BLOCK_K, layout=gl.SliceLayout(0, gl.SliceLayout(1, a_layout))
    )
    offset_sk_b = gl.arange(
        0, SPLIT_K, layout=gl.SliceLayout(1, gl.SliceLayout(2, b_layout))
    )
    offset_bk = gl.arange(
        0, BLOCK_K, layout=gl.SliceLayout(0, gl.SliceLayout(2, b_layout))
    )
    offset_n = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, gl.SliceLayout(1, b_layout))
    )
    ram = gl.load(
        token_indices_sorted_by_lora_ids + lora_m_indices_start + offset_m,
        offset_m < lora_m_size,
        other=0,
    ).to(gl.int64)
    if SLICE_NUM == 1:
        cur_lora_ptr = lora_ptr
    else:
        cur_lora_ptr = gl.load(lora_ptr + slice_id).to(
            gl.pointer_type(input_ptr.dtype.element_ty)
        )
    cur_lora_ptr += lora_id.to(gl.int64) * lora_d0_stride

    accumulator = gl.full((SPLIT_K, BLOCK_M, BLOCK_N), 0, gl.float32, mma_layout)
    for iteration in range(gl.cdiv(K, SPLIT_K * BLOCK_K)):
        offset_k_a = offset_sk_a[:, None, None] * BLOCK_K + offset_ak[None, None, :]
        offset_k_a += iteration * SPLIT_K * BLOCK_K
        offset_k_b = offset_sk_b[:, None, None] * BLOCK_K + offset_bk[None, :, None]
        offset_k_b += iteration * SPLIT_K * BLOCK_K
        a = gl.load(
            input_ptr + ram[None, :, None] * K + offset_k_a,
            (offset_m[None, :, None] < lora_m_size) & (offset_k_a < K),
            other=0,
        )
        b = gl.load(
            cur_lora_ptr
            + offset_n[None, None, :] * lora_d1_stride
            + offset_k_b * lora_d2_stride,
            (offset_n[None, None, :] < N) & (offset_k_b < K),
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
    offset_cm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, out_layout))
    offset_cn = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, out_layout)
    )
    output_rows = gl.convert_layout(ram, gl.SliceLayout(1, out_layout))
    c_ptr = (
        out_ptr
        + (slice_id.to(gl.int64) * M + output_rows[:, None]) * N
        + offset_cn[None, :]
    )
    c_mask = (cta_m_offset + offset_cm[:, None] < lora_m_size) & (
        offset_cn[None, :] < N
    )
    gl.store(c_ptr, result * scaling, mask=c_mask)


def launch_lora_shrink_multicta(
    inputs: torch.Tensor,
    lora_ptr_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    num_active_loras: int,
    scaling: float,
    lora_strides_d0: int,
    lora_strides_d1: int,
    lora_strides_d2: int,
    *,
    BLOCK_M: int | None,
    BLOCK_N: int | None,
    BLOCK_K: int | None,
    SPLIT_K: int | None,
    NUM_WARPS: int | None,
    NUM_CTAS: int | None,
) -> None:
    M, K = inputs.shape
    NUM_SLICES, _, N = output_tensor.shape
    assert NUM_CTAS is not None
    CGA_LAYOUT = tuple((1 << bit, 0, 0) for bit in range(NUM_CTAS.bit_length() - 1))
    grid = (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        NUM_SLICES,
        num_active_loras,
    )
    _lora_shrink_multicta_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        M,
        N,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        scaling,
        lora_strides_d0,
        lora_strides_d1,
        lora_strides_d2,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        SPLIT_K,
        NUM_SLICES,
        CGA_LAYOUT,
        num_ctas=NUM_CTAS,
        num_warps=NUM_WARPS,
        enable_fp_fusion=False,
    )
