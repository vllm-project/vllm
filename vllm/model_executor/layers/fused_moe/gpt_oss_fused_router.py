# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPT-OSS MoE router with Triton topk kernel."""

import torch

from vllm.triton_utils import tl, triton


def torch_dtype_to_tl(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    elif dtype == torch.float32:
        return tl.float32
    elif dtype == torch.int32:
        return tl.int32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


@triton.jit
def _topk_softmax_kernel(
    logits_ptr,
    weights_ptr,
    indices_ptr,
    M,
    N: tl.constexpr,
    topk: tl.constexpr,
    topk_padded: tl.constexpr,
    stride_lm,
    stride_ln,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    BLOCK_N: tl.constexpr,
    RENORM: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, topk_padded)
    mask_n = offs_n < N
    store_mask = offs_k < topk

    # specify topk<=2 and RENORM specialization by tl.constexpr,
    # similar as `constexpr if` in C++17
    if topk == 1:
        for row_idx in tl.range(pid, M, num_programs, num_stages):
            logits = tl.load(
                logits_ptr + row_idx * stride_lm + offs_n * stride_ln,
                mask=mask_n,
                other=float("-inf"),
            )

            if not RENORM:
                row_sub_max = logits - tl.max(logits, axis=0)
                numerator = tl.exp(row_sub_max)
                denominator = tl.sum(numerator, axis=0)
                logits = numerator / denominator

            cur_max = tl.max(logits, axis=0)
            cur_idx = tl.argmax(logits, axis=0)

            if RENORM:
                cur_max = 1

            tl.store(weights_ptr + row_idx * stride_wm + 0 * stride_wk, cur_max)
            tl.store(indices_ptr + row_idx * stride_im + 0 * stride_wk, cur_idx)

    elif topk == 2:
        for row_idx in tl.range(pid, M, num_programs, num_stages):
            logits = tl.load(
                logits_ptr + row_idx * stride_lm + offs_n * stride_ln,
                mask=mask_n,
                other=float("-inf"),
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
            tl.store(indices_ptr + row_idx * stride_im + 1 * stride_wk, idx1)

    else:
        topk_vals = tl.zeros([topk_padded], dtype=tl.float32) + float("-inf")
        topk_idxs = tl.zeros([topk_padded], dtype=tl.int32)

        for row_idx in tl.range(pid, M, num_programs, num_stages):
            logits = tl.load(
                logits_ptr + row_idx * stride_lm + offs_n * stride_ln,
                mask=mask_n,
                other=float("-inf"),
            )

            if not RENORM:
                row_sub_max = logits - tl.max(logits, axis=0)
                numerator = tl.exp(row_sub_max)
                denominator = tl.sum(numerator, axis=0)
                logits = numerator / denominator

            for k in tl.static_range(topk):
                cur_max = tl.max(logits, axis=0)
                cur_idx = tl.argmax(logits, axis=0)

                k_mask = offs_k == k
                topk_vals = tl.where(k_mask, cur_max, topk_vals)
                topk_idxs = tl.where(k_mask, cur_idx, topk_idxs)

                logits = tl.where(offs_n == cur_idx, float("-inf"), logits)

            if RENORM:
                topk_vals = topk_vals - tl.max(topk_vals, axis=0)
                numerator = tl.exp(topk_vals)
                denominator = tl.sum(numerator, axis=0)
                topk_vals = numerator / denominator

            tl.store(
                weights_ptr + row_idx * stride_wm + offs_k * stride_wk,
                topk_vals,
                mask=store_mask,
            )
            tl.store(
                indices_ptr + row_idx * stride_im + offs_k * stride_ik,
                topk_idxs,
                mask=store_mask,
            )


def fused_topk_softmax(
    router_logits: torch.Tensor,
    topk: int,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = router_logits.shape  # num_tokens, num_experts

    weights = torch.empty(
        (M, topk), device=router_logits.device, dtype=router_logits.dtype
    )
    indices = torch.empty((M, topk), device=router_logits.device, dtype=torch.int32)

    BLOCK_N = triton.next_power_of_2(N)  # num_padded_experts
    topk_padded = triton.next_power_of_2(topk)
    grid = (M,)
    num_stages = 2

    _topk_softmax_kernel[grid](
        logits_ptr=router_logits,
        weights_ptr=weights,
        indices_ptr=indices,
        M=M,
        N=N,
        topk=topk,
        topk_padded=topk_padded,
        stride_lm=router_logits.stride(0),
        stride_ln=router_logits.stride(1),
        stride_wm=weights.stride(0),
        stride_wk=weights.stride(1),
        stride_im=indices.stride(0),
        stride_ik=indices.stride(1),
        BLOCK_N=BLOCK_N,
        RENORM=renormalize,
        num_stages=num_stages,
    )

    return weights, indices


def gpt_oss_custom_routing_function(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # only use gating_output to avoid padding issues
    return fused_topk_softmax(gating_output, topk, renormalize)
