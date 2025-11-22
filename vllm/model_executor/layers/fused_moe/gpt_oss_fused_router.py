# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPT-OSS MoE router with Triton topk kernel."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _topk_softmax_kernel(
    logits_ptr,
    weights_ptr,
    indices_ptr,
    M,
    N,
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
):
    token_idx = tl.program_id(0)

    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    logit_offs = logits_ptr + token_idx * stride_lm + offs * stride_ln
    logits = tl.load(logit_offs, mask=mask, other=float("-inf"))

    topk_vals = tl.zeros([topk_padded], dtype=tl.float32) + float("-inf")
    topk_idxs = tl.zeros([topk_padded], dtype=tl.int32)

    working_logits = logits

    for k in range(topk):
        cur_max = tl.max(working_logits, axis=0)
        cur_idx = tl.argmax(working_logits, axis=0)

        k_mask = tl.arange(0, topk_padded) == k
        topk_vals = tl.where(k_mask, cur_max, topk_vals)
        topk_idxs = tl.where(k_mask, cur_idx, topk_idxs)

        mask_selected = offs == cur_idx
        working_logits = tl.where(mask_selected, float("-inf"), working_logits)

    if RENORM:
        max_val = tl.max(topk_vals, axis=0)
        exp_vals = tl.exp(topk_vals - max_val)
        sum_exp = tl.sum(exp_vals, axis=0)
        topk_vals = exp_vals / sum_exp

    offs_k = tl.arange(0, topk_padded)

    store_mask = offs_k < topk

    weight_ptrs = weights_ptr + token_idx * stride_wm + offs_k * stride_wk
    tl.store(weight_ptrs, topk_vals, mask=store_mask)

    index_ptrs = indices_ptr + token_idx * stride_im + offs_k * stride_ik
    tl.store(index_ptrs, topk_idxs, mask=store_mask)


def fused_topk_softmax(
    router_logits: torch.Tensor,
    topk: int,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = router_logits.shape

    weights = torch.empty((M, topk), device=router_logits.device, dtype=torch.float32)
    indices = torch.empty((M, topk), device=router_logits.device, dtype=torch.int32)

    BLOCK_N = triton.next_power_of_2(N)

    topk_padded = triton.next_power_of_2(topk)

    grid = (M,)

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
