# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register-resident sigmoid + bias top-k for degenerate grouped routing.

Several DeepSeek-V3-architecture checkpoints route with ``use_grouped_topk=True``
but ``num_expert_group=1`` and ``topk_group=1``, so the group stage selects
nothing: Moonlight-16B-A3B (64 experts, top-6) and Kimi-K2 (384 experts, top-8),
both ``scoring_func="sigmoid"`` with a ``noaux_tc`` correction bias.  On ROCm this
lands in ``aiter.biased_grouped_topk_hip`` (``grouped_topk_kernel<...,
NUM_GRP=1>``), which keeps the biased scores in LDS and runs one serial round
per selected expert of ``LDS scan -> shuffle argmax -> LDS invalidate -> LDS
read of the pre-bias sigmoid``.

With a single group the group-selection stage is the identity, so the op is
exactly "sigmoid(logits) + bias -> top-k -> weights = sigmoid at the chosen
experts (optionally renormalised, times routed_scaling_factor)".  This kernel
does that with all scores held in registers: each score is packed with its
expert index into a unique, order-preserving int64 key, so every round is a
single ``tl.max`` plus an equality invalidate (no argmax pair reduction, no
LDS round trips, ties broken towards the lower expert id).
"""

import torch

from vllm.triton_utils import tl, triton

# Shape bounds, not model bounds: no checkpoint is assumed. _MAX_TOKENS keeps
# this to decode / spec-verify batches, where one wave per token wins; larger
# (prefill) batches keep the AITER kernel, which has more parallelism per
# launch. _MAX_EXPERTS and _MAX_TOPK are where this kernel stops beating AITER
# on MI355X (gfx950), measured under CUDA graph replay with bf16 router logits:
# up to 512 experts and top-8 it averages 1.16-1.57x per shape, and no launch
# falls below 0.92x even with a forced score tie. At top-16 one row with an
# exactly tied score sends the launch through the fallback path at 0.62-0.78x
# (512, 896 and 1024 experts), and from 896 experts tie-free launches gain at
# most 1.09x, so those routers keep AITER.
_MAX_TOKENS = 128
_MAX_EXPERTS = 512
_MAX_TOPK = 8
# One wave per token: every round's max is a pure in-wave reduction (no LDS
# round trip / barrier), which is what bounds latency at decode batch sizes.
_NUM_WARPS = 1


@triton.jit
def _sigmoid_bias_flat_topk_kernel(
    logits_ptr,
    bias_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    stride_lm,
    stride_wm,
    stride_im,
    routed_scaling_factor,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RENORM: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_E)
    emask = offs < E
    row = logits_ptr + pid.to(tl.int64) * stride_lm

    x = tl.load(row + offs, mask=emask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + offs, mask=emask, other=0.0).to(tl.float32)
    # Same formula as the AITER kernel: 1 / (1 + 2^(-x * log2(e))).
    s = 1.0 / (1.0 + tl.exp2(x * -1.4426950408889634))
    # NaN logits (e.g. padded CUDA-graph rows) must not win selection: like
    # AITER's '>'-based scan, treat them as score 0.
    s = tl.where(s == s, s, 0.0)
    sc = s + b

    w_row = topk_weights_ptr + pid.to(tl.int64) * stride_wm
    i_row = topk_ids_ptr + pid.to(tl.int64) * stride_im

    # Fast path: TOPK rounds of fp32 max (DPP-fused v_max) that also record the
    # round in which each expert was taken. With no exact score ties exactly
    # TOPK experts are taken, round k == output slot k (descending, as AITER),
    # and the weights come straight from registers.
    v = tl.where(emask, sc, float("-inf"))
    rnd = tl.full([BLOCK_E], -1, tl.int32)
    for k in tl.static_range(TOPK):
        m = tl.max(v, axis=0)
        hit = v == m
        rnd = tl.where(hit, k, rnd)
        v = tl.where(hit, float("-inf"), v)
    picked = rnd >= 0
    n_picked = tl.sum(picked.to(tl.int32), axis=0)

    if n_picked == TOPK:
        if RENORM:
            wsum = tl.sum(tl.where(picked, s, 0.0), axis=0)
            scale = routed_scaling_factor / tl.maximum(wsum, 1e-20)
        else:
            scale = routed_scaling_factor
        tl.store(w_row + rnd, s * scale, mask=picked)
        tl.store(i_row + rnd, offs, mask=picked)
    else:
        # Exact ties: pack (order-preserving score, E-1-idx) into a unique
        # int64 key so equal scores resolve to the lower expert id.
        bits = sc.to(tl.int32, bitcast=True)
        okey = bits ^ ((bits >> 31) & 0x7FFFFFFF)
        key = (okey.to(tl.int64) << 32) | (BLOCK_E - 1 - offs).to(tl.int64)
        sentinel = tl.full([BLOCK_E], -2147483648, tl.int32).to(tl.int64) << 32
        key = tl.where(emask, key, sentinel)

        kk = tl.arange(0, BLOCK_K)
        sel = tl.zeros([BLOCK_K], dtype=tl.int64)
        for k in tl.static_range(TOPK):
            mk = tl.max(key, axis=0)
            sel = tl.where(kk == k, mk, sel)
            key = tl.where(key == mk, sentinel, key)

        kmask = kk < TOPK
        ids = (BLOCK_E - 1) - (sel & (BLOCK_E - 1)).to(tl.int32)
        ids = tl.where(kmask, ids, 0)
        # Pre-bias sigmoid of the chosen experts, re-derived from the logits
        # with the identical formula (bit-exact with the selection value).
        xs = tl.load(row + ids, mask=kmask, other=0.0).to(tl.float32)
        w = 1.0 / (1.0 + tl.exp2(xs * -1.4426950408889634))
        w = tl.where(kmask & (w == w), w, 0.0)
        if RENORM:
            w = w * (routed_scaling_factor / tl.maximum(tl.sum(w, axis=0), 1e-20))
        else:
            w = w * routed_scaling_factor
        tl.store(w_row + kk, w, mask=kmask)
        tl.store(i_row + kk, ids, mask=kmask)


def can_use_sigmoid_bias_flat_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
) -> bool:
    if correction_bias is None:
        return False
    # The grouping stage only eliminates experts when some group is left out.
    # With topk_group >= num_expert_group every group is selected, so the
    # masked top-k degenerates to a plain top-k over all experts. That covers
    # a router declaring a single group and one whose group count was
    # synthesised to fit an AITER kernel.
    if topk_group < num_expert_group:
        return False
    if gating_output.dim() != 2 or gating_output.stride(-1) != 1:
        return False
    num_tokens, num_experts = gating_output.shape
    topk = topk_ids.shape[-1]
    return (
        num_tokens <= _MAX_TOKENS
        and num_experts <= _MAX_EXPERTS
        and topk <= min(_MAX_TOPK, num_experts)
        and correction_bias.numel() == num_experts
        and correction_bias.is_contiguous()
        and gating_output.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and topk_ids.dtype == torch.int32
        and topk_weights.dtype == torch.float32
        and topk_ids.stride(-1) == 1
        and topk_weights.stride(-1) == 1
    )


def sigmoid_bias_flat_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    need_renorm: bool,
    routed_scaling_factor: float = 1.0,
) -> None:
    """In-place equivalent of aiter.biased_grouped_topk when no group is
    eliminated, i.e. topk_group >= num_expert_group."""
    # AITER rejects a bias whose dtype differs from the gating dtype, so
    # rocm_aiter_ops.biased_grouped_topk casts it before dispatching. Routers
    # commonly hold a bf16 gating output next to an fp32 correction bias, so
    # without the same cast here the two paths would score experts at
    # different precision and this function would not be the equivalent its
    # docstring claims: selected expert ids can differ.
    if correction_bias.dtype != gating_output.dtype:
        correction_bias = correction_bias.to(gating_output.dtype)
    num_tokens, num_experts = gating_output.shape
    if num_tokens == 0:
        return
    topk = topk_ids.shape[-1]
    _sigmoid_bias_flat_topk_kernel[(num_tokens,)](
        gating_output,
        correction_bias,
        topk_weights,
        topk_ids,
        gating_output.stride(0),
        topk_weights.stride(0),
        topk_ids.stride(0),
        float(routed_scaling_factor),
        E=num_experts,
        BLOCK_E=triton.next_power_of_2(num_experts),
        TOPK=topk,
        BLOCK_K=triton.next_power_of_2(topk),
        RENORM=bool(need_renorm),
        num_warps=_NUM_WARPS,
    )
