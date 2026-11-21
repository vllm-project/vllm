# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental deterministic LoRA shrink with an in-cluster K reduction."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import gl, gluon, triton


@gluon.jit(do_not_specialize=["M", "scaling"])
def _lora_shrink_tile(
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
    lora_m_size,
    lora_m_indices_start,
    lora_id,
    lora_d0_stride: gl.constexpr,
    lora_d1_stride: gl.constexpr,
    lora_d2_stride: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SPLIT_K: gl.constexpr,
    SLICE_NUM: gl.constexpr,
    CGA_LAYOUT: gl.constexpr,
    REDUCE_CGA_LAYOUT: gl.constexpr,
    USE_ADAPTIVE: gl.constexpr,
    USE_ASYNC: gl.constexpr,
):
    SPLITS_PER_CTA: gl.constexpr = SPLIT_K // gl.num_ctas() if USE_ADAPTIVE else 1
    K_STEP: gl.constexpr = BLOCK_K // SPLITS_PER_CTA
    LOAD_K: gl.constexpr = min(8, K_STEP // 8) if USE_ADAPTIVE else 8
    THREADS_K: gl.constexpr = min(32, K_STEP // 8) if USE_ADAPTIVE else 8
    WARPS_N: gl.constexpr = min(2, gl.num_warps() // SPLITS_PER_CTA)

    cta_n_num = gl.cdiv(N, BLOCK_N)
    pid_m_n = gl.program_id(0)
    pid_m = pid_m_n // cta_n_num
    pid_n = pid_m_n % cta_n_num
    slice_id = gl.program_id(1)
    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        return
    # Logical K splits can span both CTAs and warps within a CTA.
    a_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, LOAD_K],
        [1, 32 // THREADS_K, THREADS_K],
        [SPLITS_PER_CTA, gl.num_warps() // SPLITS_PER_CTA, 1],
        [2, 1, 0],
        cga_layout=CGA_LAYOUT,
    )
    b_layout: gl.constexpr = gl.BlockedLayout(
        [1, LOAD_K, 1],
        [1, THREADS_K, 32 // THREADS_K],
        [SPLITS_PER_CTA, 1, gl.num_warps() // SPLITS_PER_CTA],
        [1, 2, 0],
        cga_layout=CGA_LAYOUT,
    )
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        [2, 0],
        [SPLITS_PER_CTA, gl.num_warps() // (SPLITS_PER_CTA * WARPS_N), WARPS_N],
        [1, 16, 8],
        cga_layout=CGA_LAYOUT,
    )
    # A and B need separate K offsets because their layouts differ.
    offset_sk_a = gl.arange(
        0, SPLIT_K, layout=gl.SliceLayout(1, gl.SliceLayout(2, a_layout))
    )
    offset_m = cta_m_offset + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(0, gl.SliceLayout(2, a_layout))
    )
    offset_ak = gl.arange(
        0, K_STEP, layout=gl.SliceLayout(0, gl.SliceLayout(1, a_layout))
    )
    offset_sk_b = gl.arange(
        0, SPLIT_K, layout=gl.SliceLayout(1, gl.SliceLayout(2, b_layout))
    )
    offset_bk = gl.arange(
        0, K_STEP, layout=gl.SliceLayout(0, gl.SliceLayout(2, b_layout))
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

    if USE_ASYNC and SLICE_NUM > 1:
        async_aligned = cur_lora_ptr.to(gl.int64) % 16 == 0
    else:
        async_aligned = USE_ASYNC
    if async_aligned:
        if SLICE_NUM > 1:
            # Keep the alignment hint off the original fallback pointer.
            async_lora_ptr = (cur_lora_ptr.to(gl.int64) & -16).to(
                gl.pointer_type(input_ptr.dtype.element_ty)
            )
        else:
            async_lora_ptr = cur_lora_ptr
        async_lora_ptr = gl.multiple_of(async_lora_ptr, 16)
        a_smem = gl.allocate_shared_memory(
            input_ptr.dtype.element_ty,
            (SPLIT_K, BLOCK_M, K_STEP),
            gl.SwizzledSharedLayout(8, 1, 8, [2, 1, 0], cga_layout=CGA_LAYOUT),
        )
        b_smem = gl.allocate_shared_memory(
            input_ptr.dtype.element_ty,
            (SPLIT_K, K_STEP, BLOCK_N),
            gl.SwizzledSharedLayout(8, 1, 8, [1, 2, 0], cga_layout=CGA_LAYOUT),
        )
        offset_k_a = offset_sk_a[:, None, None] * BLOCK_K + offset_ak[None, None, :]
        offset_k_b = offset_sk_b[:, None, None] * BLOCK_K + offset_bk[None, :, None]
        a_ptr = input_ptr + ram[None, :, None] * K + offset_k_a
        b_ptr = (
            async_lora_ptr
            + offset_n[None, None, :] * lora_d1_stride
            + offset_k_b * lora_d2_stride
        )
        a_mask = (offset_m[None, :, None] < lora_m_size) & (offset_k_a < K)
        b_mask = (offset_n[None, None, :] < N) & (offset_k_b < K)
        gl.nvidia.ampere.async_copy.async_load(b_smem, b_ptr, b_mask)
        gl.nvidia.ampere.async_copy.commit_group()
        gl.nvidia.ampere.async_copy.async_load(a_smem, a_ptr, a_mask)
        gl.nvidia.ampere.async_copy.commit_group()
        accumulator = gl.full((SPLIT_K, BLOCK_M, BLOCK_N), 0, gl.float32, mma_layout)
        for iteration in range(gl.cdiv(K, SPLIT_K * BLOCK_K) * SPLITS_PER_CTA):
            gl.nvidia.ampere.async_copy.wait_group(0)
            gl.barrier()
            a = a_smem.load(gl.DotOperandLayout(0, mma_layout, 2))
            b = b_smem.load(gl.DotOperandLayout(1, mma_layout, 2))
            # All warps must finish reading before the buffer is reused.
            gl.barrier()
            if iteration + 1 < gl.cdiv(K, SPLIT_K * BLOCK_K) * SPLITS_PER_CTA:
                k_base = ((iteration + 1) // SPLITS_PER_CTA) * SPLIT_K * BLOCK_K
                k_base += ((iteration + 1) % SPLITS_PER_CTA) * K_STEP
                offset_k_a = (
                    offset_sk_a[:, None, None] * BLOCK_K
                    + offset_ak[None, None, :]
                    + k_base
                )
                offset_k_b = (
                    offset_sk_b[:, None, None] * BLOCK_K
                    + offset_bk[None, :, None]
                    + k_base
                )
                a_ptr = input_ptr + ram[None, :, None] * K + offset_k_a
                b_ptr = (
                    async_lora_ptr
                    + offset_n[None, None, :] * lora_d1_stride
                    + offset_k_b * lora_d2_stride
                )
                a_mask = (offset_m[None, :, None] < lora_m_size) & (offset_k_a < K)
                b_mask = (offset_n[None, None, :] < N) & (offset_k_b < K)
                gl.nvidia.ampere.async_copy.async_load(b_smem, b_ptr, b_mask)
                gl.nvidia.ampere.async_copy.commit_group()
                gl.nvidia.ampere.async_copy.async_load(a_smem, a_ptr, a_mask)
                gl.nvidia.ampere.async_copy.commit_group()
            accumulator = gl.nvidia.ampere.mma_v2(a, b, accumulator)
        gl.nvidia.ampere.async_copy.wait_group(0)
        a_smem._keep_alive()
        b_smem._keep_alive()
    else:
        accumulator = gl.full((SPLIT_K, BLOCK_M, BLOCK_N), 0, gl.float32, mma_layout)
        for iteration in range(gl.cdiv(K, SPLIT_K * BLOCK_K) * SPLITS_PER_CTA):
            k_base = (iteration // SPLITS_PER_CTA) * SPLIT_K * BLOCK_K
            k_base += (iteration % SPLITS_PER_CTA) * K_STEP
            offset_k_a = offset_sk_a[:, None, None] * BLOCK_K + offset_ak[None, None, :]
            offset_k_a += k_base
            offset_k_b = offset_sk_b[:, None, None] * BLOCK_K + offset_bk[None, :, None]
            offset_k_b += k_base
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

    # Reduce K splits within each warp and distribute output rows across CTAs.
    reduce_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 2],
        [SPLIT_K, max(1, 4 // SPLIT_K), min(8, 32 // SPLIT_K)],
        [1, gl.num_warps(), 1],
        [2, 1, 0],
        cga_layout=REDUCE_CGA_LAYOUT,
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
    REDUCE_CGA_LAYOUT: gl.constexpr,
    USE_ADAPTIVE: gl.constexpr,
    USE_ASYNC: gl.constexpr,
    SM_COUNT: gl.constexpr,
):
    first_id = gl.load(lora_ids)
    lora_idx = gl.program_id(2) + (first_id == -1).to(gl.int32)
    lora_id = gl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return
    lora_m_size = gl.load(num_tokens_per_lora + lora_idx)
    if USE_ADAPTIVE and gl.program_id(0) >= gl.cdiv(lora_m_size, 16) * gl.cdiv(N, 16):
        return
    if not USE_ADAPTIVE:
        pid_m = gl.program_id(0) // gl.cdiv(N, BLOCK_N)
        if pid_m * BLOCK_M >= lora_m_size:
            return
    lora_start = gl.load(lora_token_start_loc + lora_idx)
    if not USE_ADAPTIVE:
        _lora_shrink_tile(
            input_ptr,
            lora_ptr,
            out_ptr,
            M,
            N,
            K,
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora,
            lora_token_start_loc,
            lora_ids,
            scaling,
            lora_m_size,
            lora_start,
            lora_id,
            lora_d0_stride,
            lora_d1_stride,
            lora_d2_stride,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            SPLIT_K,
            SLICE_NUM,
            CGA_LAYOUT,
            REDUCE_CGA_LAYOUT,
            False,
            False,
        )
        return
    else:
        valid_m = M
        if first_id == -1:
            valid_m -= gl.load(num_tokens_per_lora)
        m_tiles = gl.cdiv(valid_m, 32)

        # M16 tiles must still fit within the host's M32-based launch grid.
        if m_tiles * gl.cdiv(N, 16) * SLICE_NUM * gl.num_ctas() < SM_COUNT and (
            lora_m_size * 2 <= M or lora_m_size <= 16
        ):
            _lora_shrink_tile(
                input_ptr,
                lora_ptr,
                out_ptr,
                M,
                N,
                K,
                token_indices_sorted_by_lora_ids,
                num_tokens_per_lora,
                lora_token_start_loc,
                lora_ids,
                scaling,
                lora_m_size,
                lora_start,
                lora_id,
                lora_d0_stride,
                lora_d1_stride,
                lora_d2_stride,
                16,
                16,
                BLOCK_K,
                SPLIT_K,
                SLICE_NUM,
                CGA_LAYOUT,
                REDUCE_CGA_LAYOUT,
                True,
                USE_ASYNC,
            )

        elif (
            USE_ASYNC
            and N > 32
            and m_tiles * gl.cdiv(N, 64) * SLICE_NUM * gl.num_ctas() >= SM_COUNT
        ):
            _lora_shrink_tile(
                input_ptr,
                lora_ptr,
                out_ptr,
                M,
                N,
                K,
                token_indices_sorted_by_lora_ids,
                num_tokens_per_lora,
                lora_token_start_loc,
                lora_ids,
                scaling,
                lora_m_size,
                lora_start,
                lora_id,
                lora_d0_stride,
                lora_d1_stride,
                lora_d2_stride,
                32,
                64,
                BLOCK_K,
                SPLIT_K,
                SLICE_NUM,
                CGA_LAYOUT,
                REDUCE_CGA_LAYOUT,
                True,
                USE_ASYNC,
            )

        elif (
            N > 16 and m_tiles * gl.cdiv(N, 32) * SLICE_NUM * gl.num_ctas() >= SM_COUNT
        ):
            _lora_shrink_tile(
                input_ptr,
                lora_ptr,
                out_ptr,
                M,
                N,
                K,
                token_indices_sorted_by_lora_ids,
                num_tokens_per_lora,
                lora_token_start_loc,
                lora_ids,
                scaling,
                lora_m_size,
                lora_start,
                lora_id,
                lora_d0_stride,
                lora_d1_stride,
                lora_d2_stride,
                32,
                32,
                BLOCK_K,
                SPLIT_K,
                SLICE_NUM,
                CGA_LAYOUT,
                REDUCE_CGA_LAYOUT,
                True,
                USE_ASYNC,
            )

        else:
            _lora_shrink_tile(
                input_ptr,
                lora_ptr,
                out_ptr,
                M,
                N,
                K,
                token_indices_sorted_by_lora_ids,
                num_tokens_per_lora,
                lora_token_start_loc,
                lora_ids,
                scaling,
                lora_m_size,
                lora_start,
                lora_id,
                lora_d0_stride,
                lora_d1_stride,
                lora_d2_stride,
                32,
                16,
                BLOCK_K,
                SPLIT_K,
                SLICE_NUM,
                CGA_LAYOUT,
                REDUCE_CGA_LAYOUT,
                True,
                USE_ASYNC,
            )


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
    sm_count = current_platform.num_compute_units(inputs.device.index or 0)
    candidate_ctas = min(NUM_CTAS, 4)
    work = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N) * NUM_SLICES
    use_adaptive = (
        (BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS) == (32, 16, 256, 4)
        and SPLIT_K == NUM_CTAS
        and M > BLOCK_M
        and work * candidate_ctas >= sm_count
    )
    # Keep the logical K partition fixed when reducing physical CTA count.
    if use_adaptive:
        NUM_CTAS = candidate_ctas
    CGA_LAYOUT = tuple((1 << bit, 0, 0) for bit in range(NUM_CTAS.bit_length() - 1))
    REDUCE_CGA_LAYOUT = tuple(
        (0, 1 << bit, 0) for bit in range(NUM_CTAS.bit_length() - 1)
    )
    # LoRAKernelMeta reserves one extra slot for tokens without LoRA.
    grid = (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        NUM_SLICES,
        min(num_active_loras, lora_ids.numel() - 1),
    )
    # Multi-slice staging needs enough work to support an N32 tile.
    async_work = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, 32) * NUM_SLICES * NUM_CTAS
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
        REDUCE_CGA_LAYOUT,
        use_adaptive,
        use_adaptive
        and (NUM_SLICES == 1 or async_work >= sm_count)
        and inputs.data_ptr() % 16 == 0
        and lora_ptr_tensor.data_ptr() % 16 == 0
        and K % 8 == 0
        and lora_strides_d0 % 8 == 0
        and lora_strides_d1 % 8 == 0
        and lora_strides_d2 == 1,
        sm_count,
        num_ctas=NUM_CTAS,
        num_warps=NUM_WARPS,
        enable_fp_fusion=False,
    )
