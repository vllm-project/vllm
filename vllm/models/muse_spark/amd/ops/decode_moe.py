# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels of the Muse Spark ROCm decode MoE (`..decode_moe`)."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize_on_alignment=["M"])
def _decode_moe_prep_kernel(
    part_ptr,
    bias_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    x_ptr,
    nw_ptr,
    y_ptr,
    M,
    eps,
    gain_center,
    router_output_multiplier,
    route_scale,
    division_eps,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_PADDED: tl.constexpr,
    ROUTE_ROWS: tl.constexpr,
    HAS_MULTIPLIER: tl.constexpr,
    HAS_EPS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
):
    """Programs [0, M): RMSNorm of one row. Programs >= M: routing of ROUTE_ROWS
    tokens each, one warp per token so the top-k reductions stay in a warp."""
    pid = tl.program_id(0)
    if pid < M:
        t = pid
        d = tl.arange(0, BLOCK_D)
        dm = d < D
        x = tl.load(x_ptr + t * D + d, mask=dm, other=0.0).to(tl.float32)
        if HAS_WEIGHT:
            w = tl.load(nw_ptr + d, mask=dm, other=0.0).to(tl.float32) + gain_center
            if INPUT_SCALE:
                x = x * w
        y = x * tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / D) + eps)
        if HAS_WEIGHT and not INPUT_SCALE:
            y = y * w
        tl.store(y_ptr + t * D + d, y.to(y_ptr.dtype.element_ty), mask=dm)
    else:
        rows = (pid - M) * ROUTE_ROWS + tl.arange(0, ROUTE_ROWS)
        row_mask = rows < M
        expert_offs = tl.arange(0, N)
        offs = rows[:, None] * N + expert_offs[None, :]
        # router logits: the split-K partials, summed in split order
        gating_scores = tl.load(part_ptr + offs, mask=row_mask[:, None], other=0.0)
        for s in tl.static_range(1, SPLITS):
            gating_scores += tl.load(
                part_ptr + s * M * N + offs, mask=row_mask[:, None], other=0.0
            )
        # sigmoid scores; the expert bias steers the selection only
        if HAS_MULTIPLIER:
            gating_scores = gating_scores * router_output_multiplier
        normalized_scores = tl.sigmoid(gating_scores)
        bias_values = tl.load(bias_ptr + expert_offs)
        scores_for_topk = normalized_scores + bias_values[None, :]
        slots = tl.arange(0, TOPK_PADDED)
        slot_mask = slots < TOPK
        selected_mask = tl.zeros([ROUTE_ROWS, N], dtype=tl.int1)
        ids = tl.zeros([ROUTE_ROWS, TOPK_PADDED], dtype=tl.int32)
        scores = tl.zeros([ROUTE_ROWS, TOPK_PADDED], dtype=tl.float32)
        score_sum = tl.zeros([ROUTE_ROWS], dtype=tl.float32)
        for k in tl.static_range(TOPK):
            masked_scores = tl.where(selected_mask, -float("inf"), scores_for_topk)
            max_idx = tl.argmax(masked_scores, axis=1)
            is_selected = expert_offs[None, :] == max_idx[:, None]
            selected_score = tl.sum(
                tl.where(is_selected, normalized_scores, 0.0), axis=1
            )
            selected_mask = selected_mask | is_selected
            slot = (slots == k)[None, :]
            ids = tl.where(slot, max_idx.to(tl.int32)[:, None], ids)
            scores = tl.where(slot, selected_score[:, None], scores)
            score_sum = score_sum + selected_score
        if HAS_EPS:
            score_sum = score_sum + division_eps
        weights = scores * (route_scale / score_sum)[:, None]
        out_offs = rows[:, None] * TOPK + slots[None, :]
        out_mask = row_mask[:, None] & slot_mask[None, :]
        tl.store(topk_ids_ptr + out_offs, ids, mask=out_mask)
        tl.store(topk_weights_ptr + out_offs, weights, mask=out_mask)


def supports_decode_moe_prep(num_experts: int, top_k: int) -> bool:
    return num_experts >= 8 and num_experts & (num_experts - 1) == 0 and 0 < top_k <= 16


def decode_moe_prep(
    router_logits: torch.Tensor,
    expert_bias: torch.Tensor,
    top_k: int,
    route_scale: float,
    router_output_multiplier: float | None,
    division_eps: float | None,
    pooled: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    gain_center: float,
    input_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Router split-K reduce, sigmoid top-k routing (`MuseSparkRouter`) and the
    pre-expert RMSNorm of `pooled` in one launch.

    `router_logits` is `[T, E]` or its `[splits, T, E]` split-K partials.
    Returns `(topk_weights [T, k] fp32, topk_ids [T, k] int32, normed [T, D])`.
    """
    if router_logits.dim() == 2:
        router_logits = router_logits.unsqueeze(0)
    splits, T, E = router_logits.shape
    assert router_logits.dtype == torch.float32 and router_logits.is_contiguous()
    assert pooled.dim() == 2 and pooled.is_contiguous() and pooled.shape[0] == T
    D = pooled.shape[1]
    dev = pooled.device
    bias = expert_bias if expert_bias.dtype == torch.float32 else expert_bias.float()
    assert bias.is_contiguous() and bias.shape[0] == E
    topk_weights = torch.empty((T, top_k), dtype=torch.float32, device=dev)
    topk_ids = torch.empty((T, top_k), dtype=torch.int32, device=dev)
    normed = torch.empty_like(pooled)
    if T == 0:
        return topk_weights, topk_ids, normed
    block_d = triton.next_power_of_2(D)
    num_warps = 8 if block_d >= 4096 else 4
    _decode_moe_prep_kernel[(T + triton.cdiv(T, num_warps),)](
        router_logits,
        bias,
        topk_weights,
        topk_ids,
        pooled,
        norm_weight if norm_weight is not None else pooled,
        normed,
        T,
        norm_eps,
        gain_center,
        router_output_multiplier if router_output_multiplier is not None else 1.0,
        route_scale,
        division_eps if division_eps is not None else 0.0,
        N=E,
        SPLITS=splits,
        TOPK=top_k,
        TOPK_PADDED=triton.next_power_of_2(top_k),
        ROUTE_ROWS=num_warps,
        HAS_MULTIPLIER=router_output_multiplier is not None,
        HAS_EPS=division_eps is not None,
        D=D,
        BLOCK_D=block_d,
        HAS_WEIGHT=norm_weight is not None,
        INPUT_SCALE=input_scale and norm_weight is not None,
        num_warps=num_warps,
    )
    return topk_weights, topk_ids, normed
