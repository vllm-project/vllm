# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE gate GEMM and sigmoid top-k with a score-correction bias, fused.

For a router that holds an fp32 gate weight and selects experts by
`sigmoid(logits) + e_score_correction_bias` (MiniMax-M3), the decode path
otherwise runs a cast, a split-K router GEMM with its reduction, and the top-k
kernel. Here it is two Triton launches:

1. `_gate_partial_kernel`: split-K `x @ W^T` reading the BF16 activations
   directly. `W` (fp32) is split into three BF16 pieces whose products with
   BF16 `x` are exact in the fp32 accumulator, as an fp32 GEMM's are; only
   the summation order differs.
2. `_sigmoid_topk_bias_kernel`: sums the split-K partials in a fixed order,
   then sigmoid, + bias for the selection, top-k (ties go to the lower expert
   id), weights = the selected experts' sigmoid scores, renormalized and
   scaled.

The result equals the unfused path up to the GEMM's summation order (the
logits agree to ~1e-6 relative), so an expert id can only differ where the
k-th and (k+1)-th biased scores tie to that precision.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gate_partial_kernel(
    x_ptr,
    w_ptr,
    partial_ptr,
    M,
    K,
    stride_xm,
    stride_wn,
    k_per_split,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_e = tl.arange(0, BLOCK_E)
    mask_m = offs_m < M
    mask_e = offs_e < E
    acc = tl.zeros((BLOCK_M, BLOCK_E), dtype=tl.float32)
    k_start = pid_s * k_per_split
    for kk in range(0, k_per_split, BLOCK_K):
        offs_k = k_start + kk + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_e[:, None] * stride_wn + offs_k[None, :],
            mask=mask_e[:, None] & mask_k[None, :],
            other=0.0,
        )
        # W = hi + mid + lo exactly (three 8-bit mantissas); x is BF16, so
        # every product is exact in fp32.
        w_hi = w.to(tl.bfloat16)
        rest = w - w_hi.to(tl.float32)
        w_mid = rest.to(tl.bfloat16)
        w_lo = (rest - w_mid.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(x, tl.trans(w_lo), acc)
        acc = tl.dot(x, tl.trans(w_mid), acc)
        acc = tl.dot(x, tl.trans(w_hi), acc)
    tl.store(
        partial_ptr + (pid_s * M + offs_m[:, None]) * E + offs_e[None, :],
        acc,
        mask=mask_m[:, None] & mask_e[None, :],
    )


@triton.jit
def _sigmoid_topk_bias_kernel(
    partial_ptr,
    bias_ptr,
    padding_ptr,
    weights_ptr,
    ids_ptr,
    M,
    num_splits,
    routed_scaling_factor,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_S: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    HAS_PADDING: tl.constexpr,
):
    row = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E
    offs_s = tl.arange(0, BLOCK_S)
    partial = tl.load(
        partial_ptr + (offs_s[:, None] * M + row) * E + offs_e[None, :],
        mask=(offs_s[:, None] < num_splits) & mask_e[None, :],
        other=0.0,
    )
    logits = tl.sum(partial, axis=0)
    bias = tl.load(bias_ptr + offs_e, mask=mask_e, other=0.0).to(tl.float32)
    scores = 1.0 / (1.0 + tl.exp2(logits * -1.4426950408889634))
    scores = tl.where(scores == scores, scores, 0.0)  # NaN logits never win
    choice = scores + bias
    # Unique, order-preserving int64 keys (score bits, BLOCK_E - 1 - id): the
    # maximum is the highest score, and among equal scores the lowest id.
    bits = choice.to(tl.int32, bitcast=True)
    ordered = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    key = (ordered.to(tl.int64) << 32) | (BLOCK_E - 1 - offs_e).to(tl.int64)
    lowest = tl.full([BLOCK_E], -2147483648, tl.int32).to(tl.int64) << 32
    key = tl.where(mask_e, key, lowest)
    offs_k = tl.arange(0, BLOCK_K)
    ids = tl.zeros([BLOCK_K], dtype=tl.int32)
    weights = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        best = tl.max(key, axis=0)
        expert = (BLOCK_E - 1) - (best & (BLOCK_E - 1)).to(tl.int32)
        weight = tl.sum(tl.where(offs_e == expert, scores, 0.0), axis=0)
        ids = tl.where(offs_k == k, expert, ids)
        weights = tl.where(offs_k == k, weight, weights)
        key = tl.where(key == best, lowest, key)
    mask_k = offs_k < TOPK
    weights = tl.where(mask_k, weights, 0.0)
    if RENORMALIZE:
        weights = weights * (
            routed_scaling_factor / tl.maximum(tl.sum(weights, axis=0), 1e-20)
        )
    else:
        weights = weights * routed_scaling_factor
    if HAS_PADDING:
        # Padding rows are skipped downstream: expert -1, weight 0.
        is_padding = tl.load(padding_ptr + row) != 0
        ids = tl.where(is_padding, -1, ids)
        weights = tl.where(is_padding, 0.0, weights)
    tl.store(weights_ptr + row * TOPK + offs_k, weights, mask=mask_k)
    tl.store(ids_ptr + row * TOPK + offs_k, ids, mask=mask_k)


def _launch_shape(num_tokens: int, hidden: int) -> tuple[int, int, int, int]:
    """(BLOCK_M, k_per_split, BLOCK_K, num_warps) for the gate GEMM.

    From a sweep on MI355X at hidden 6144, 128 experts: M <= 128 is best at
    16 rows, 32 splits, BLOCK_K 64 and 8 warps; M <= 512 at 32 rows, 32
    splits and 4 warps; larger M at 64 rows, 16 splits and 8 warps.
    """
    if num_tokens <= 128:
        block_m, splits, block_k, warps = 16, 32, 64, 8
    elif num_tokens <= 512:
        block_m, splits, block_k, warps = 32, 32, 64, 4
    else:
        block_m, splits, block_k, warps = 64, 16, 64, 8
    k_per_split = triton.cdiv(triton.cdiv(hidden, splits), block_k) * block_k
    return block_m, k_per_split, block_k, warps


def fused_gate_sigmoid_topk_bias(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
    is_padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route BF16 `hidden_states` [M, K] with an fp32 gate `gate_weight`
    [E, K] and an fp32 bias [E]; returns fp32 weights and int32 expert ids,
    both [M, topk]. Rows where `is_padding` is set get expert -1, weight 0."""
    assert hidden_states.dim() == 2 and hidden_states.dtype == torch.bfloat16
    assert hidden_states.stride(-1) == 1
    assert gate_weight.dtype == torch.float32 and gate_weight.is_contiguous()
    assert e_score_correction_bias.dtype == torch.float32
    assert e_score_correction_bias.is_contiguous()
    num_tokens, hidden = hidden_states.shape
    num_experts = gate_weight.shape[0]
    assert gate_weight.shape[1] == hidden
    assert e_score_correction_bias.numel() == num_experts and topk <= num_experts

    device = hidden_states.device
    weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=device)
    ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
    if num_tokens == 0:
        return weights, ids

    block_m, k_per_split, block_k, warps = _launch_shape(num_tokens, hidden)
    num_splits = triton.cdiv(hidden, k_per_split)
    partial = torch.empty(
        (num_splits, num_tokens, num_experts), dtype=torch.float32, device=device
    )
    block_e = triton.next_power_of_2(num_experts)
    _gate_partial_kernel[(triton.cdiv(num_tokens, block_m), num_splits)](
        hidden_states,
        gate_weight,
        partial,
        num_tokens,
        hidden,
        hidden_states.stride(0),
        gate_weight.stride(0),
        k_per_split,
        E=num_experts,
        BLOCK_M=block_m,
        BLOCK_E=block_e,
        BLOCK_K=block_k,
        num_warps=warps,
    )
    padding = None
    if is_padding is not None:
        padding = is_padding[:num_tokens].contiguous().view(torch.uint8)
    _sigmoid_topk_bias_kernel[(num_tokens,)](
        partial,
        e_score_correction_bias,
        ids if padding is None else padding,  # unused without padding
        weights,
        ids,
        num_tokens,
        num_splits,
        float(routed_scaling_factor),
        E=num_experts,
        BLOCK_E=block_e,
        BLOCK_S=triton.next_power_of_2(num_splits),
        TOPK=topk,
        BLOCK_K=triton.next_power_of_2(topk),
        RENORMALIZE=renormalize,
        HAS_PADDING=padding is not None,
        num_warps=1,
    )
    return weights, ids


def fused_gate_sigmoid_topk_bias_ref(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
    is_padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The unfused computation in PyTorch, for tests."""
    logits = torch.nn.functional.linear(hidden_states.float(), gate_weight)
    scores = logits.sigmoid()
    scores = torch.where(scores == scores, scores, torch.zeros_like(scores))
    ids = torch.topk(scores + e_score_correction_bias, k=topk, dim=-1)[1]
    weights = scores.gather(1, ids)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    weights = (weights * routed_scaling_factor).to(torch.float32)
    ids = ids.to(torch.int32)
    if is_padding is not None:
        pad = is_padding[: hidden_states.shape[0]].unsqueeze(1)
        weights = weights.masked_fill(pad, 0)
        ids = ids.masked_fill(pad, -1)
    return weights, ids
