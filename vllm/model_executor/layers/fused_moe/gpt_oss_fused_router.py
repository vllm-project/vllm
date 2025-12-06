# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.triton_bitonic_sort import (
    bitonic_sort32_descending,
)
from vllm.triton_utils import tl, triton


@triton.autotune(
    configs=[
        triton.Config({"ROWS_PER_PID": r}, num_warps=num_warps, num_stages=num_stages)
        for r in [1, 2, 4, 8, 16, 32]
        for num_warps in [1, 2, 4, 8, 16]
        for num_stages in [1, 2, 3]
    ],
    key=["N", "topk"],
    cache_results=True,
)
@triton.jit
def _topk_softmax_kernel(
    logits_ptr,
    weights_ptr,
    indices_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    topk: tl.constexpr,
    stride_lm,
    stride_ln,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    topk_padded: tl.constexpr,
    RENORM: tl.constexpr,
    ROWS_PER_PID: tl.constexpr,
    num_stages: tl.constexpr,
    USE_BITONIC: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, topk_padded)
    mask_n = offs_n < N
    store_mask = offs_k < topk
    warp_size: tl.constexpr = 32

    # impl topk<=2 and RENORM specialization by tl.constexpr,
    # same as constexpr if in C++17
    if topk == 1:
        for row_idx in tl.range(pid, M, num_programs, num_stages, warp_specialize=True):
            if BLOCK_N != N:
                logits = tl.load(
                    logits_ptr + row_idx * stride_lm + offs_n * stride_ln,
                    mask=mask_n,
                    other=float("-inf"),
                )
            else:
                logits = tl.load(
                    logits_ptr + row_idx * stride_lm + offs_n * stride_ln,
                )

            if not RENORM:
                row_sub_max = logits - tl.max(logits, axis=0)
                numerator = tl.exp(row_sub_max)
                denominator = tl.sum(numerator, axis=0)
                logits = numerator / denominator

            cur_max = 1 if RENORM else tl.max(logits, axis=0)
            cur_idx = tl.argmax(logits, axis=0)

            tl.store(weights_ptr + row_idx * stride_wm + 0 * stride_wk, cur_max)
            tl.store(indices_ptr + row_idx * stride_im + 0 * stride_ik, cur_idx)

    elif topk == 2:
        for row_idx in tl.range(pid, M, num_programs, num_stages, warp_specialize=True):
            if BLOCK_N != N:
                logits = tl.load(
                    logits_ptr + row_idx * stride_lm + offs_n * stride_ln,
                    mask=mask_n,
                    other=float("-inf"),
                )
            else:
                logits = tl.load(
                    logits_ptr + row_idx * stride_lm + offs_n * stride_ln,
                )

            if not RENORM:
                row_sub_max = logits - tl.max(logits, axis=0)
                numerator = tl.exp(row_sub_max)
                denominator = tl.sum(numerator, axis=0)
                logits = numerator / denominator

            val0 = tl.max(logits, axis=0)
            idx0 = tl.argmax(logits, axis=0)
            logits = tl.where(offs_n == idx0, float("-inf"), logits)
            val1 = tl.max(logits, axis=0)
            idx1 = tl.argmax(logits, axis=0)

            if RENORM:
                max_val = tl.maximum(val0, val1)
                exp0 = tl.exp(val0 - max_val)
                exp1 = tl.exp(val1 - max_val)
                val0 = exp0 / (exp0 + exp1)
                val1 = exp1 / (exp0 + exp1)

            tl.store(weights_ptr + row_idx * stride_wm, val0)
            tl.store(indices_ptr + row_idx * stride_im, idx0)
            tl.store(weights_ptr + row_idx * stride_wm + 1 * stride_wk, val1)
            tl.store(indices_ptr + row_idx * stride_im + 1 * stride_ik, idx1)

    else:
        rows = tl.arange(0, ROWS_PER_PID)
        for row_idx in tl.range(
            pid * ROWS_PER_PID,
            M,
            num_programs * ROWS_PER_PID,
            num_stages,
            warp_specialize=True,
        ):
            topk_vals = tl.full(
                [ROWS_PER_PID, topk_padded], float("-inf"), dtype=tl.float32
            )
            topk_idxs = tl.zeros([ROWS_PER_PID, topk_padded], dtype=tl.int32)
            row_indices = row_idx + rows  # [ROWS_PER_POD,]
            row_mask = row_indices < M

            # broadcast to [ROWS_PER_PID, BLOCKN]
            ptr_off = (
                logits_ptr
                + row_indices[:, None] * stride_lm
                + offs_n[None, :] * stride_ln
            )
            if BLOCK_N == N and BLOCK_M == M:
                logits = tl.load(ptr_off)
            elif BLOCK_N != N and BLOCK_M != M:
                logits = tl.load(
                    ptr_off,
                    mask=row_mask[:, None] & mask_n[None, :],
                    other=float("-inf"),
                )
            elif BLOCK_N != N:
                logits = tl.load(ptr_off, mask=mask_n[None, :], other=float("-inf"))
            elif BLOCK_M != M:
                logits = tl.load(ptr_off, mask=row_mask[:, None], other=float("-inf"))

            if not RENORM:
                row_sub_max = logits - tl.max(
                    logits, axis=1, keep_dims=True
                )  # [ROWS_PER_PID, BLOCK_N] - [ROWS_PER_PID,1]
                numerator = tl.exp(row_sub_max)
                denominator = tl.sum(
                    numerator, axis=1, keep_dims=True
                )  # [ROWS_PER_PID, BLOCKN]
                logits = numerator / denominator

            if warp_size == N:
                idx = tl.arange(0, warp_size)[None, :]
                idxs = tl.broadcast_to(idx, (ROWS_PER_PID, warp_size))
                sorted_val, sorted_idx = bitonic_sort32_descending(
                    val=logits, idx=idxs
                )  # [ROWS_PER_PID, 32]
                tl.static_assert(sorted_val.shape == (ROWS_PER_PID, warp_size))
                # USE_BITONIC: tl.constexpr = True
            else:
                for k in tl.static_range(topk):
                    cur_max = tl.max(
                        logits, axis=1, keep_dims=True
                    )  # [ROWS_PER_PID, 1]
                    cur_idx = tl.argmax(logits, axis=1, keep_dims=True)

                    k_mask = offs_k == k
                    topk_vals = tl.where(
                        k_mask, cur_max, topk_vals
                    )  # [ROWS_PER PID, 1], [ROWS_PER PID, topkpadded]
                    topk_idxs = tl.where(k_mask, cur_idx, topk_idxs)

                    mask_selected = (
                        cur_idx == offs_n[None, :]
                    )  # [ROWSPERPID,1] [1,BLOCKN]
                    logits = tl.where(mask_selected, float("-inf"), logits)
                # USE_BITONIC: tl.constexpr = False

            if RENORM:
                if USE_BITONIC:
                    topk_col_mask = (
                        tl.arange(0, warp_size)[None, :] < topk
                    )  # [1, warp_size]
                    masked_val = tl.where(topk_col_mask, sorted_val, float("-inf"))
                    masked_val = masked_val - tl.max(masked_val, axis=1, keep_dims=True)
                    numerator = tl.exp(masked_val)
                    numerator = tl.where(topk_col_mask, numerator, 0.0)
                    denominator = tl.sum(numerator, axis=1, keep_dims=True)
                    sorted_val = tl.where(
                        topk_col_mask, numerator / denominator, sorted_val
                    )
                else:
                    topk_vals = topk_vals - tl.max(
                        topk_vals, axis=1, keep_dims=True
                    )  # [ROWSPERPID, topkpadded] - [ROWSPERPID,1]
                    numerator = tl.exp(topk_vals)
                    denominator = tl.sum(
                        numerator, axis=1, keep_dims=True
                    )  # [ROWSPERPID,1]
                    topk_vals = numerator / denominator  # [ROWSPERPID,topkpadded]

            if USE_BITONIC:
                offs_warp_size = tl.arange(0, warp_size)
                store_col_mask = offs_warp_size < topk
                tl.store(
                    weights_ptr
                    + row_indices[:, None] * stride_wm
                    + offs_warp_size[None, :] * stride_wk,
                    sorted_val,
                    mask=row_mask[:, None] & store_col_mask[None, :],
                )
                tl.store(
                    indices_ptr
                    + row_indices[:, None] * stride_im
                    + offs_warp_size[None, :] * stride_ik,
                    sorted_idx,
                    mask=row_mask[:, None] & store_col_mask[None, :],
                )
            else:
                if topk == topk_padded and BLOCK_M == M:
                    tl.store(
                        weights_ptr
                        + row_indices[:, None] * stride_wm  # [ROWSPERPID,1]
                        + offs_k[None, :] * stride_wk,  # [1, topkpadded]
                        topk_vals,
                    )
                    tl.store(
                        indices_ptr
                        + row_indices[:, None] * stride_im
                        + offs_k[None, :] * stride_ik,
                        topk_idxs,
                    )
                elif topk != topk_padded and BLOCK_M != M:
                    tl.store(
                        weights_ptr
                        + row_indices[:, None] * stride_wm  # [ROWSPERPID,1]
                        + offs_k[None, :] * stride_wk,  # [1, topkpadded]
                        topk_vals,
                        mask=row_mask[:, None] & store_mask[None, :],  # [1, topkpadded]
                    )
                    tl.store(
                        indices_ptr
                        + row_indices[:, None] * stride_im
                        + offs_k[None, :] * stride_ik,
                        topk_idxs,
                        mask=row_mask[:, None] & store_mask[None, :],
                    )
                elif topk != topk_padded:
                    tl.store(
                        weights_ptr
                        + row_indices[:, None] * stride_wm  # [ROWSPERPID,1]
                        + offs_k[None, :] * stride_wk,  # [1, topkpadded]
                        topk_vals,
                        mask=store_mask[None, :],  # [1, topkpadded]
                    )
                    tl.store(
                        indices_ptr
                        + row_indices[:, None] * stride_im
                        + offs_k[None, :] * stride_ik,
                        topk_idxs,
                        mask=store_mask[None, :],
                    )
                elif BLOCK_M != M:
                    tl.store(
                        weights_ptr
                        + row_indices[:, None] * stride_wm  # [ROWSPERPID,1]
                        + offs_k[None, :] * stride_wk,  # [1, topkpadded]
                        topk_vals,
                        mask=row_mask[:, None],
                    )
                    tl.store(
                        indices_ptr
                        + row_indices[:, None] * stride_im
                        + offs_k[None, :] * stride_ik,
                        topk_idxs,
                        mask=row_mask[:, None],
                    )


def fused_topk_softmax(
    router_logits: torch.Tensor,
    topk: int,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = router_logits.shape  # num_tokens, num_experts
    weights = torch.empty((M, topk), device=router_logits.device, dtype=torch.float32)
    indices = torch.empty((M, topk), device=router_logits.device, dtype=torch.int32)

    BLOCK_N = triton.next_power_of_2(N)  # num_padded_experts
    topk_padded = triton.next_power_of_2(topk)
    BLOCK_M = triton.next_power_of_2(M)
    warp_size = 32

    # enable autotune to find correct num threadblock,
    # refer to https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    grid = lambda META: (triton.cdiv(M, META["ROWS_PER_PID"]),)

    _topk_softmax_kernel[grid](
        logits_ptr=router_logits,
        weights_ptr=weights,
        indices_ptr=indices,
        M=M,
        N=N,
        topk=topk,
        stride_lm=router_logits.stride(0),
        stride_ln=router_logits.stride(1),
        stride_wm=weights.stride(0),
        stride_wk=weights.stride(1),
        stride_im=indices.stride(0),
        stride_ik=indices.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        topk_padded=topk_padded,
        RENORM=renormalize,
        USE_BITONIC=topk > 2 and warp_size == N,
    )

    return weights, indices


def gpt_oss_custom_routing_function(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # only use gating_output to avoid padding issues
    assert gating_output.is_contiguous()
    return fused_topk_softmax(gating_output, topk, renormalize)
