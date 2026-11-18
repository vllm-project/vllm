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
