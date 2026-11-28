# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused pointwise operations for DSpine's FP32 last-write refinement.

Adapted from Tencent/DSpine's sampled refinement path. The final RMS
denominator and full predecessor/successor logits are preserved.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _extract_exact_features(
    Codebook,
    CandidateIds,
    AnchorIds,
    Features,
    ANCHOR_STRIDE: tl.constexpr,
    STEPS: tl.constexpr,
    TOP_K: tl.constexpr,
    C: tl.constexpr,
    SQRT_C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0)
    row = pair // TOP_K
    if row % STEPS == 0:
        token = tl.load(AnchorIds + row // STEPS * ANCHOR_STRIDE).to(tl.int64)
    else:
        token = tl.load(CandidateIds + pair - TOP_K).to(tl.int64)
    c = tl.arange(0, BLOCK)
    code = tl.load(Codebook + token * C + c, c < C, 0).to(tl.float32)
    tl.store(Features + pair * C + c, SQRT_C * code, c < C)


@triton.jit
def _feature_message(
    Query,
    Predecessor,
    Bias,
    Message,
    TOP_K: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    q = tl.load(Query + (pair // TOP_K) * RANK + r, r < RANK, 0).to(tl.float32)
    gate = tl.load(Predecessor + pair * (2 * RANK) + r, r < RANK, 0)
    value = tl.load(Predecessor + pair * (2 * RANK) + RANK + r, r < RANK, 0)
    bias = tl.load(Bias + r, r < RANK, 0).to(tl.float32)
    message = tl.sigmoid(q + gate + bias) * value
    tl.store(Message + pair * RANK + r, message, r < RANK)


@triton.jit
def _feature_finish(
    Raw,
    Delta,
    NormWeight,
    Normalized,
    TOP_K: tl.constexpr,
    HIDDEN: tl.constexpr,
    NORM_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0)
    h = tl.arange(0, BLOCK)
    raw = tl.load(Raw + (pair // TOP_K) * HIDDEN + h, h < HIDDEN, 0).to(tl.float32)
    delta = tl.load(Delta + pair * HIDDEN + h, h < HIDDEN, 0)
    write_rms = tl.sqrt(
        tl.maximum(tl.sum(raw * raw, 0) / HIDDEN, 1.1754943508222875e-38)
    )
    refined = raw + write_rms * delta
    norm = tl.sqrt(tl.sum(refined * refined, 0) / HIDDEN + NORM_EPS)
    weight = tl.load(NormWeight + h, h < HIDDEN, 0).to(tl.float32)
    normalized = tl.div_rn(refined, norm) * weight
    tl.store(Normalized + pair * HIDDEN + h, normalized, h < HIDDEN)


def exact_features(
    codebook: torch.Tensor,
    candidate_ids: torch.Tensor,
    anchor_ids: torch.Tensor,
    sqrt_c: float,
) -> torch.Tensor:
    _, steps, top_k = candidate_ids.shape
    c = codebook.shape[-1]
    features = torch.empty(
        (*candidate_ids.shape, c), device=codebook.device, dtype=torch.float32
    )
    _extract_exact_features[(candidate_ids.numel(),)](
        codebook,
        candidate_ids.contiguous(),
        anchor_ids,
        features,
        anchor_ids.stride(0),
        steps,
        top_k,
        c,
        sqrt_c,
        triton.next_power_of_2(c),
    )
    return features


def feature_message(
    query: torch.Tensor, predecessor: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    rank = query.shape[-1]
    top_k = predecessor.shape[-2]
    out = torch.empty(
        (*predecessor.shape[:-1], rank), device=query.device, dtype=torch.float32
    )
    _feature_message[(out.numel() // rank,)](
        query.contiguous(),
        predecessor,
        bias,
        out,
        top_k,
        rank,
        triton.next_power_of_2(rank),
        enable_fp_fusion=False,
    )
    return out


def feature_finish(
    raw: torch.Tensor, delta: torch.Tensor, norm_weight: torch.Tensor, eps: float
) -> torch.Tensor:
    hidden = raw.shape[-1]
    top_k = delta.shape[-2]
    normalized = torch.empty_like(delta)
    _feature_finish[(delta.numel() // hidden,)](
        raw.contiguous(),
        delta,
        norm_weight,
        normalized,
        top_k,
        hidden,
        eps,
        triton.next_power_of_2(hidden),
        enable_fp_fusion=False,
    )
    return normalized


@triton.jit
def _score_shortlist(
    Weight,
    Hidden,
    Ids,
    Scores,
    H: tl.constexpr,
    C: tl.constexpr,
    STRIDE_W0: tl.constexpr,
    STRIDE_W1: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    candidate = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    token = tl.load(Ids + row * C + candidate, candidate < C, 0).to(tl.int64)
    hidden_idx = tl.arange(0, BLOCK_H)
    score = tl.full((BLOCK_C, BLOCK_H), 0.0, tl.float32)
    for offset in tl.static_range((H + BLOCK_H - 1) // BLOCK_H):
        h = offset * BLOCK_H + hidden_idx
        hidden = tl.load(Hidden + row * H + h, h < H, 0).to(tl.float32)
        weight = tl.load(
            Weight + token[:, None] * STRIDE_W0 + h[None, :] * STRIDE_W1,
            (candidate[:, None] < C) & (h[None, :] < H),
            0,
        ).to(tl.float32)
        score += hidden[None, :] * weight
    tl.store(Scores + row * C + candidate, tl.sum(score, axis=1), candidate < C)


def score_shortlist(
    weight: torch.Tensor, hidden: torch.Tensor, ids: torch.Tensor
) -> torch.Tensor:
    """Gathered BF16 head scores without a token-by-candidate-by-hidden buffer."""
    rows, candidates = ids.shape
    scores = hidden.new_empty((rows, candidates))
    _score_shortlist[(rows, triton.cdiv(candidates, 4))](
        weight,
        hidden.contiguous(),
        ids.contiguous(),
        scores,
        hidden.shape[1],
        candidates,
        weight.stride(0),
        weight.stride(1),
        1024,
        4,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return scores
