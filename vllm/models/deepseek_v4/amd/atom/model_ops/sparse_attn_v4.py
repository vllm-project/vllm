# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
DeepSeek-V4 sparse attention + Sinkhorn projection torch fallbacks.

Reference TileLang implementations live in /data/DeepSeek-V4-Pro/inference/kernel.py.
PR1 ships pure-torch fallbacks for numerical correctness; AITER kernels land in PR4
(see /app/logs_claude/aiter_v4_sparse_attn_spec.md).

Both functions are written for clarity over speed. They allocate intermediate
tensors and use FP32 internally for numerical stability — matching the reference
kernel's accumulation precision. They are correct but not performant.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# sparse_attn — FlashAttention-style sparse MQA with attention sink
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_attn_triton_kernel(
    q_ptr,
    kv_ptr,
    attn_sink_ptr,
    topk_idxs_ptr,
    out_ptr,
    q_stride_b: tl.constexpr,
    q_stride_m: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    kv_stride_b: tl.constexpr,
    kv_stride_n: tl.constexpr,
    kv_stride_d: tl.constexpr,
    topk_stride_b: tl.constexpr,
    topk_stride_m: tl.constexpr,
    topk_stride_k: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_m: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bm = tl.program_id(0)
    pid_h = tl.program_id(1)
    m = pid_bm % M
    b = pid_bm // M

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    d_mask = d_offs < D

    q_base = b * q_stride_b + m * q_stride_m
    q = tl.load(
        q_ptr + q_base + h_offs[:, None] * q_stride_h + d_offs[None, :] * q_stride_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    for k_start in tl.range(0, K, BLOCK_K):
        k_pos = k_start + k_offs
        in_range = k_pos < K
        idx = tl.load(
            topk_idxs_ptr
            + b * topk_stride_b
            + m * topk_stride_m
            + k_pos * topk_stride_k,
            mask=in_range,
            other=-1,
        )
        valid = in_range & (idx >= 0)

        kv = tl.load(
            kv_ptr
            + b * kv_stride_b
            + idx[:, None] * kv_stride_n
            + d_offs[None, :] * kv_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * softmax_scale
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    sink = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(tl.float32)
    m_final = tl.maximum(m_i, sink)
    alpha = tl.exp(m_i - m_final)
    l_final = l_i * alpha + tl.exp(sink - m_final)

    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(l_final[:, None] > 0.0, (acc * alpha[:, None]) / denom[:, None], 0.0)
    out_base = b * out_stride_b + m * out_stride_m
    tl.store(
        out_ptr
        + out_base
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :] * out_stride_d,
        out,
        mask=h_mask[:, None] & d_mask[None, :],
    )


def _sparse_attn_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    if not q.is_cuda:
        raise RuntimeError("Triton sparse_attn requires CUDA/HIP tensors")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"Triton sparse_attn expects fp16/bf16 q, got {q.dtype}")
    if kv.dtype != q.dtype:
        raise RuntimeError(
            f"Triton sparse_attn expects kv dtype {q.dtype}, got {kv.dtype}"
        )

    B, M, H, D = q.shape
    K = topk_idxs.shape[-1]
    out = torch.empty_like(q)
    topk_idxs = topk_idxs.to(torch.int32)

    # Process a head tile per program. BLOCK_H must be >= 16 on AMD
    # MFMA-enabled GPUs (gfx9xx/gfx950): TritonAMDGPUOptimizeDotOperands
    # cannot lower tl.dot operands smaller than the smallest bf16 MFMA
    # tile (16x16x16) and crashes the pass pipeline rather than falling
    # back to FMA.
    block_h = 16
    block_d = triton.next_power_of_2(D)
    block_k = 16 if D >= 256 else 32
    _sparse_attn_triton_kernel[(B * M, triton.cdiv(H, block_h))](
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        topk_idxs.stride(0),
        topk_idxs.stride(1),
        topk_idxs.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        M,
        H,
        D,
        K,
        float(softmax_scale),
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return out


@triton.jit
def _sparse_attn_ragged_triton_kernel(
    q_ptr,
    kv_ptr,
    attn_sink_ptr,
    topk_idxs_ptr,
    out_ptr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    kv_stride_n: tl.constexpr,
    kv_stride_d: tl.constexpr,
    topk_stride_t: tl.constexpr,
    topk_stride_k: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    d_mask = d_offs < D

    q = tl.load(
        q_ptr
        + t * q_stride_t
        + h_offs[:, None] * q_stride_h
        + d_offs[None, :] * q_stride_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    for k_start in tl.range(0, K, BLOCK_K):
        k_pos = k_start + k_offs
        in_range = k_pos < K
        idx = tl.load(
            topk_idxs_ptr + t * topk_stride_t + k_pos * topk_stride_k,
            mask=in_range,
            other=-1,
        )
        valid = in_range & (idx >= 0)

        kv = tl.load(
            kv_ptr + idx[:, None] * kv_stride_n + d_offs[None, :] * kv_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * softmax_scale
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    sink = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(tl.float32)
    m_final = tl.maximum(m_i, sink)
    alpha = tl.exp(m_i - m_final)
    l_final = l_i * alpha + tl.exp(sink - m_final)

    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(l_final[:, None] > 0.0, (acc * alpha[:, None]) / denom[:, None], 0.0)
    tl.store(
        out_ptr
        + t * out_stride_t
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :] * out_stride_d,
        out,
        mask=h_mask[:, None] & d_mask[None, :],
    )


def _sparse_attn_ragged_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    if not q.is_cuda:
        raise RuntimeError("Triton sparse_attn_ragged requires CUDA/HIP tensors")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            f"Triton sparse_attn_ragged expects fp16/bf16 q, got {q.dtype}"
        )
    if kv.dtype != q.dtype:
        raise RuntimeError(
            f"Triton sparse_attn_ragged expects kv dtype {q.dtype}, got {kv.dtype}"
        )

    T, H, D = q.shape
    K = topk_idxs.shape[-1]
    out = torch.empty_like(q)
    topk_idxs = topk_idxs.to(torch.int32)

    # See _sparse_attn_triton: BLOCK_H must be >= 16 for AMD MFMA lowering.
    block_h = 16
    block_d = triton.next_power_of_2(D)
    block_k = 16 if D >= 256 else 32
    _sparse_attn_ragged_triton_kernel[(T, triton.cdiv(H, block_h))](
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        topk_idxs.stride(0),
        topk_idxs.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H,
        D,
        K,
        float(softmax_scale),
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return out


def _sparse_attn_ragged_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    return _sparse_attn_torch(
        q.unsqueeze(0),
        kv.unsqueeze(0),
        attn_sink,
        topk_idxs.unsqueeze(0),
        softmax_scale,
    ).squeeze(0)


def sparse_attn_ragged(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse attention over flat ragged sequences.

    Args:
        q: [num_tokens, H, D]
        kv: [total_kv, D]
        topk_idxs: [num_tokens, K] global indices into `kv`; -1 entries are skipped.
    """
    return _sparse_attn_ragged_triton(q, kv, attn_sink, topk_idxs, softmax_scale)


def _sparse_attn_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse multi-head attention with per-query top-k KV gather and per-head sink.

    Reference: /data/DeepSeek-V4-Pro/inference/kernel.py:276-368

    For each query position (b, m), gathers the top-k KV positions selected by
    `topk_idxs[b, m, :]` (with -1 entries skipped), computes scaled dot-product
    attention with all H heads sharing the single-headed `kv`, and includes a
    per-head learnable `attn_sink` logit in the softmax denominator only.

    Args:
        q:           [B, M, H, D]  query, BF16
        kv:          [B, N, D]     shared key=value, single head (MQA), BF16
        attn_sink:   [H,]          per-head sink logit, FP32
        topk_idxs:   [B, M, K]     selected KV positions, INT32. -1 = skip.
        softmax_scale: scalar      softmax scale (typically D ** -0.5)

    Returns:
        o: [B, M, H, D] BF16

    Notes:
        - The sink contributes only to the denominator; it never appears as
          attention weight on a KV position. Letting `attn_sink[h] = -inf`
          recovers standard sparse attention without sink.
        - Internal accumulation is FP32. Output is cast back to q.dtype.
        - Invalid (-1) topk entries set their logit to -inf, contributing 0 to
          softmax and producing zero contribution to the output.
        - When all K entries are invalid for some (b, m, h), the result is 0
          (sum_exp = exp(sink - (-inf)) = 0; division below uses safe eps).
    """
    B, M, H, D = q.shape
    _, N, D_kv = kv.shape
    K = topk_idxs.shape[-1]
    assert kv.shape[0] == B, f"batch mismatch: q={B} vs kv={kv.shape[0]}"
    assert D_kv == D, f"head_dim mismatch: q={D} vs kv={D_kv}"
    assert attn_sink.shape == (H,), f"attn_sink shape {attn_sink.shape} != ({H},)"
    assert topk_idxs.shape == (B, M, K)

    out_dtype = q.dtype
    device = q.device

    # ----- Gather KV per query position -----
    # safe_idxs avoids out-of-bounds for the -1 sentinel; we mask the result below.
    valid = topk_idxs != -1  # [B, M, K] bool
    safe_idxs = topk_idxs.clamp(min=0).long()  # [B, M, K] int64

    # Advanced indexing: kv_gathered[b, m, k, :] = kv[b, safe_idxs[b, m, k], :]
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, M, K)
    kv_gathered = kv[batch_idx, safe_idxs]  # [B, M, K, D]

    # Promote to FP32 for accumulation; zero out invalid positions in value tensor
    # so they contribute nothing to weighted sum even before masking the logits.
    kv_f32 = kv_gathered.float()
    kv_f32 = torch.where(
        valid.unsqueeze(-1), kv_f32, torch.zeros((), dtype=kv_f32.dtype, device=device)
    )

    # ----- Scores: q @ kv^T -----
    # q: [B, M, H, D]  ;  kv_f32: [B, M, K, D]  ->  scores: [B, M, H, K]
    q_f32 = q.float()
    scores = torch.einsum("bmhd,bmkd->bmhk", q_f32, kv_f32) * float(softmax_scale)
    # Mask invalid positions in logits with -inf so they contribute 0 weight.
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

    # ----- Softmax with sink in denominator -----
    # Concat sink logit at the end: combined[..., :K] are real positions,
    # combined[..., K] is the per-head sink. Take softmax over (K+1), then drop
    # the sink column from the weights — its contribution stays in the
    # denominator via softmax normalization.
    sink = attn_sink.float().view(1, 1, H, 1).expand(B, M, H, 1)
    combined = torch.cat([scores, sink], dim=-1)  # [B, M, H, K+1]

    # Numerically stable softmax: subtract max along K+1 axis.
    # When all entries (including sink) are -inf for some (b,m,h), softmax of
    # all -inf is undefined; we get NaN. Replace with 0 in that pathological
    # case (matches kernel's behavior since `acc_o` stays 0 in that case).
    cmax = combined.amax(dim=-1, keepdim=True)
    cmax = torch.where(
        cmax == float("-inf"),
        torch.zeros((), dtype=cmax.dtype, device=device),
        cmax,
    )
    weights = (combined - cmax).exp()
    denom = weights.sum(dim=-1, keepdim=True)
    weights = weights / denom.clamp(min=1e-30)
    weights_kv = weights[..., :K]  # drop sink contribution from output side

    # ----- Weighted sum -----
    # weights_kv: [B, M, H, K]  ;  kv_f32: [B, M, K, D]  ->  out: [B, M, H, D]
    out = torch.einsum("bmhk,bmkd->bmhd", weights_kv, kv_f32)
    return out.to(out_dtype)


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse multi-head attention with optional Triton backend.

    Uses the Triton kernel; ``_sparse_attn_torch`` is kept as the reference.
    """
    return _sparse_attn_triton(q, kv, attn_sink, topk_idxs, softmax_scale)


# ---------------------------------------------------------------------------
# hc_split_sinkhorn — Manifold-Constrained Hyper-Connections projection
# ---------------------------------------------------------------------------


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split + project mHC mixing parameters.

    Reference: /data/DeepSeek-V4-Pro/inference/kernel.py:371-440

    Splits the raw `mixes` tensor into three components and applies the per-component
    constraints required by Manifold-Constrained Hyper-Connections (Xie et al., 2026):
        - pre  : Sigmoid → non-negative input gate  [..., hc_mult]
        - post : 2*Sigmoid → bounded output gate     [..., hc_mult]
        - comb : Sinkhorn-Knopp projection onto the Birkhoff polytope
                 (doubly stochastic matrices)        [..., hc_mult, hc_mult]

    Sinkhorn schedule (matches reference exactly):
        Iteration 1: row softmax + eps, then column normalize.
        Iterations 2..sinkhorn_iters: alternate row-normalize and column-normalize.

    Args:
        mixes:    [..., (2+hc_mult)*hc_mult] FP32
        hc_scale: [3] FP32 — scaling factors for (pre, post, comb)
        hc_base:  [(2+hc_mult)*hc_mult] FP32 — additive bias before activation
        hc_mult:  number of HC copies (default 4 for V4)
        sinkhorn_iters: number of Sinkhorn iterations (default 20)
        eps:      stability epsilon

    Returns:
        pre:  [..., hc_mult]
        post: [..., hc_mult]
        comb: [..., hc_mult, hc_mult]   (rows and columns each sum ~ 1.0)
    """
    expected = (2 + hc_mult) * hc_mult
    assert (
        mixes.shape[-1] == expected
    ), f"mixes last dim {mixes.shape[-1]} != (2+hc_mult)*hc_mult = {expected}"
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (expected,)

    prefix = mixes.shape[:-1]
    mixes_f = mixes.float()
    scale_f = hc_scale.float()
    base_f = hc_base.float()

    # Split mixes into the three logical components.
    pre_raw = mixes_f[..., :hc_mult]
    post_raw = mixes_f[..., hc_mult : 2 * hc_mult]
    comb_raw = mixes_f[..., 2 * hc_mult :].reshape(*prefix, hc_mult, hc_mult)

    base_pre = base_f[:hc_mult]
    base_post = base_f[hc_mult : 2 * hc_mult]
    base_comb = base_f[2 * hc_mult :].reshape(hc_mult, hc_mult)

    # Apply per-component activations.
    pre = torch.sigmoid(pre_raw * scale_f[0] + base_pre) + eps
    post = 2.0 * torch.sigmoid(post_raw * scale_f[1] + base_post)
    comb = comb_raw * scale_f[2] + base_comb

    # ----- First Sinkhorn iteration (special: row-softmax + eps, then col-norm) -----
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    # ----- Remaining iterations: alternate row-normalize and column-normalize -----
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


# ---------------------------------------------------------------------------
# Self-test (run as `python -m atom.model_ops.sparse_attn_v4`)
# ---------------------------------------------------------------------------


def _selftest():
    torch.manual_seed(0)

    # Sinkhorn: doubly-stochastic property
    hc = 4
    B, S = 2, 7
    mixes = torch.randn(B, S, (2 + hc) * hc, dtype=torch.float32)
    scale = torch.tensor([0.1, 0.1, 0.5])
    base = torch.zeros((2 + hc) * hc)
    pre, post, comb = hc_split_sinkhorn(
        mixes, scale, base, hc_mult=hc, sinkhorn_iters=20
    )
    assert pre.shape == (B, S, hc) and post.shape == (B, S, hc)
    assert comb.shape == (B, S, hc, hc)
    # Each row and column should sum ~ 1
    row_err = (comb.sum(dim=-1) - 1.0).abs().max().item()
    col_err = (comb.sum(dim=-2) - 1.0).abs().max().item()
    assert row_err < 1e-4, f"row sum drift: {row_err}"
    assert col_err < 1e-4, f"col sum drift: {col_err}"
    print(f"[hc_split_sinkhorn] OK  row_err={row_err:.2e} col_err={col_err:.2e}")

    # sparse_attn: equivalence with dense attention when topk covers all positions
    B, M, H, D = 2, 4, 8, 16
    N, K = 12, 12
    q = torch.randn(B, M, H, D, dtype=torch.bfloat16)
    kv = torch.randn(B, N, D, dtype=torch.bfloat16)
    attn_sink = torch.full((H,), float("-inf"), dtype=torch.float32)  # disable sink
    # topk picks all N positions in order
    topk_idxs = (
        torch.arange(N, dtype=torch.int32).view(1, 1, N).expand(B, M, K).contiguous()
    )
    out = sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale=D**-0.5)
    # Reference dense softmax (single-head MQA: same kv broadcast across H)
    qf, kvf = q.float(), kv.float()
    scores = torch.einsum("bmhd,bnd->bmhn", qf, kvf) * (D**-0.5)
    weights = scores.softmax(dim=-1)
    ref = torch.einsum("bmhn,bnd->bmhd", weights, kvf).to(torch.bfloat16)
    err = (out - ref).abs().max().item()
    assert err < 1e-2, f"dense-equivalence err {err}"
    print(f"[sparse_attn dense-equiv] OK  max_abs_err={err:.2e}")

    # sparse_attn: -1 indices contribute zero; when all -1, output is 0
    topk_all_invalid = torch.full((B, M, K), -1, dtype=torch.int32)
    out_all_invalid = sparse_attn(
        q, kv, attn_sink, topk_all_invalid, softmax_scale=D**-0.5
    )
    # With sink=-inf and all logits=-inf, our implementation yields 0
    assert out_all_invalid.abs().max().item() < 1e-6
    print("[sparse_attn all-invalid] OK")

    # sparse_attn: attn_sink dominates -> output approaches 0
    attn_sink_large = torch.full((H,), 1e6, dtype=torch.float32)
    out_sink = sparse_attn(q, kv, attn_sink_large, topk_idxs, softmax_scale=D**-0.5)
    assert (
        out_sink.abs().max().item() < 1e-3
    ), f"sink-dominated err {out_sink.abs().max()}"
    print("[sparse_attn sink-dominates] OK")

    print("ALL OK")


if __name__ == "__main__":
    _selftest()
