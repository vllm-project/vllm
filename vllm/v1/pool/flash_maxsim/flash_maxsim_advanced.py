# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Advanced Flash-MaxSim kernels: Q-reuse and split-K.

Opt 2 — Q-reuse across docs:
  Standard kernel: each CTA loads Q (Lq×d) independently for every doc.
  With N=100K docs and Lq=1024 that's 100K × 256KB = 25.6GB of Q reads.
  Q-reuse CTA processes DOCS_PER_CTA docs while loading Q only once.
  Best for large Lq (ColPali) + large N where Q bandwidth dominates.

Opt 3 — split-K over Ld:
  Standard kernel: B CTAs, one per doc.  When B < num_SMs (~108 on A100)
  most SMs are idle.  Split-K launches B×num_splits CTAs — each handles
  Ld/num_splits doc tokens — then reduces partial maxima.
  Best for small-B, large-Ld (long-doc, single query, ColPali small corpus).
"""

import torch

from vllm.triton_utils import tl, triton

# ---------------------------------------------------------------------------
# Opt 2: Q-reuse kernel — one CTA handles DOCS_PER_CTA consecutive docs,
#        loading each Q tile exactly once.
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_Q": 16, "BLOCK_D": 64, "DOCS_PER_CTA": 2},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_Q": 16, "BLOCK_D": 64, "DOCS_PER_CTA": 4},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_Q": 32, "BLOCK_D": 64, "DOCS_PER_CTA": 2},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_Q": 32, "BLOCK_D": 64, "DOCS_PER_CTA": 4},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_Q": 32, "BLOCK_D": 128, "DOCS_PER_CTA": 2},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_Q": 32, "BLOCK_D": 128, "DOCS_PER_CTA": 4},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_Q": 64, "BLOCK_D": 64, "DOCS_PER_CTA": 2},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_Q": 64, "BLOCK_D": 64, "DOCS_PER_CTA": 4},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_Q": 64, "BLOCK_D": 128, "DOCS_PER_CTA": 2},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_Q": 64, "BLOCK_D": 128, "DOCS_PER_CTA": 4},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["Lq", "Ld", "d_pad"],
)
@triton.jit
def _maxsim_qreuse_kernel(
    Q_ptr,
    D_ptr,
    lengths_ptr,
    scores_ptr,
    B,
    Lq: tl.constexpr,
    Ld: tl.constexpr,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    stride_d_b,
    stride_d_l,
    stride_d_d,
    stride_q_l,
    stride_q_d,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DOCS_PER_CTA: tl.constexpr,
):
    """Q-reuse: one CTA handles DOCS_PER_CTA docs, loading Q tiles once each."""
    cta_id = tl.program_id(0)
    doc_base = cta_id * DOCS_PER_CTA

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d

    # Per-doc score accumulators (one scalar per doc in this CTA)
    score0 = tl.zeros([], dtype=tl.float32)
    score1 = tl.zeros([], dtype=tl.float32)
    score2 = tl.zeros([], dtype=tl.float32)
    score3 = tl.zeros([], dtype=tl.float32)

    # Load doc lengths upfront
    doc0 = doc_base + 0
    doc1 = doc_base + 1
    doc2 = doc_base + 2
    doc3 = doc_base + 3

    len0 = tl.load(lengths_ptr + doc0).to(tl.int32) if DOCS_PER_CTA >= 1 else 0
    len1 = tl.load(lengths_ptr + doc1).to(tl.int32) if DOCS_PER_CTA >= 2 else 0
    len2 = tl.load(lengths_ptr + doc2).to(tl.int32) if DOCS_PER_CTA >= 3 else 0
    len3 = tl.load(lengths_ptr + doc3).to(tl.int32) if DOCS_PER_CTA >= 4 else 0

    for q_start in range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        # Load Q block ONCE — reused for all DOCS_PER_CTA docs below
        Q_block = tl.load(
            Q_ptr + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float16)  # [BLOCK_Q, d_pad]

        # Per-doc running max for this Q tile
        m0 = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        m1 = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        m2 = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        m3 = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for d_start in range(0, Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)

            if DOCS_PER_CTA >= 1:
                dv0 = d_off < len0
                D0 = tl.load(
                    D_ptr
                    + tl.cast(doc0, tl.int64) * stride_d_b
                    + d_off[:, None] * stride_d_l
                    + k_off[None, :] * stride_d_d,
                    mask=dv0[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float16)
                S0 = tl.dot(Q_block, tl.trans(D0))
                S0 = tl.where(dv0[None, :], S0, float("-inf"))
                m0 = tl.maximum(m0, tl.max(S0, axis=1))

            if DOCS_PER_CTA >= 2:
                dv1 = d_off < len1
                D1 = tl.load(
                    D_ptr
                    + tl.cast(doc1, tl.int64) * stride_d_b
                    + d_off[:, None] * stride_d_l
                    + k_off[None, :] * stride_d_d,
                    mask=dv1[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float16)
                S1 = tl.dot(Q_block, tl.trans(D1))
                S1 = tl.where(dv1[None, :], S1, float("-inf"))
                m1 = tl.maximum(m1, tl.max(S1, axis=1))

            if DOCS_PER_CTA >= 3:
                dv2 = d_off < len2
                D2 = tl.load(
                    D_ptr
                    + tl.cast(doc2, tl.int64) * stride_d_b
                    + d_off[:, None] * stride_d_l
                    + k_off[None, :] * stride_d_d,
                    mask=dv2[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float16)
                S2 = tl.dot(Q_block, tl.trans(D2))
                S2 = tl.where(dv2[None, :], S2, float("-inf"))
                m2 = tl.maximum(m2, tl.max(S2, axis=1))

            if DOCS_PER_CTA >= 4:
                dv3 = d_off < len3
                D3 = tl.load(
                    D_ptr
                    + tl.cast(doc3, tl.int64) * stride_d_b
                    + d_off[:, None] * stride_d_l
                    + k_off[None, :] * stride_d_d,
                    mask=dv3[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float16)
                S3 = tl.dot(Q_block, tl.trans(D3))
                S3 = tl.where(dv3[None, :], S3, float("-inf"))
                m3 = tl.maximum(m3, tl.max(S3, axis=1))

        if DOCS_PER_CTA >= 1:
            score0 += tl.sum(tl.where(q_valid, m0, 0.0))
        if DOCS_PER_CTA >= 2:
            score1 += tl.sum(tl.where(q_valid, m1, 0.0))
        if DOCS_PER_CTA >= 3:
            score2 += tl.sum(tl.where(q_valid, m2, 0.0))
        if DOCS_PER_CTA >= 4:
            score3 += tl.sum(tl.where(q_valid, m3, 0.0))

    if DOCS_PER_CTA >= 1 and doc0 < B:
        tl.store(scores_ptr + doc0, score0)
    if DOCS_PER_CTA >= 2 and doc1 < B:
        tl.store(scores_ptr + doc1, score1)
    if DOCS_PER_CTA >= 3 and doc2 < B:
        tl.store(scores_ptr + doc2, score2)
    if DOCS_PER_CTA >= 4 and doc3 < B:
        tl.store(scores_ptr + doc3, score3)


def flash_maxsim_qreuse(
    Q: torch.Tensor,
    D: torch.Tensor,
    doc_lengths=None,
) -> torch.Tensor:
    """MaxSim with Q-reuse: loads Q once per CTA for DOCS_PER_CTA docs.

    Reduces Q HBM reads by DOCS_PER_CTA vs standard kernel.
    Biggest win: large Lq (ColPali T=1024) + large N.

    Q: [Lq, d], D: [B, Ld, d] -> [B].
    """
    assert Q.dim() == 2 and D.dim() == 3 and Q.shape[1] == D.shape[2]
    Lq, d = Q.shape
    B, Ld, _ = D.shape

    if doc_lengths is None:
        lengths = torch.full((B,), Ld, device=D.device, dtype=torch.int32)
    else:
        lengths = doc_lengths.to(torch.int32).contiguous()

    Q2 = Q.contiguous().half()
    D2 = D.contiguous()
    scores = torch.empty(B, device=Q.device, dtype=torch.float32)

    from .flash_maxsim import _next_pow2

    d_pad = _next_pow2(d)

    # Grid size: ceil(B / DOCS_PER_CTA) — resolved by autotune
    # Triton autotune chooses DOCS_PER_CTA; we pass a conservative ceil
    # using DOCS_PER_CTA=1 for the grid (autotune overrides the block param)
    # Use _maxsim_qreuse_kernel directly — autotune picks best DOCS_PER_CTA
    # Grid must be large enough for all docs regardless of DOCS_PER_CTA choice.
    # We use B as upper bound (1 doc/CTA = standard); autotune reduces this.
    # Triton's autotune does NOT change the grid — we need to launch with the
    # right grid for the chosen DOCS_PER_CTA.  Work around: launch with
    # grid=lambda meta: (triton.cdiv(B, meta["DOCS_PER_CTA"]),)
    grid = lambda meta: (triton.cdiv(B, meta["DOCS_PER_CTA"]),)
    _maxsim_qreuse_kernel[grid](
        Q2,
        D2,
        lengths,
        scores,
        B,
        Lq,
        Ld,
        d,
        d_pad,
        D2.stride(0),
        D2.stride(1),
        D2.stride(2),
        Q2.stride(0),
        Q2.stride(1),
    )
    return scores


# ---------------------------------------------------------------------------
# Opt 3: Split-K over Ld — two-phase kernel for small-B large-Ld regimes.
#
# Phase 1: each CTA handles a Ld/num_splits slice of one doc,
#          writes partial per-Q-token maxima to a [B, num_splits, Lq] buffer.
# Phase 2: reduce partial maxima over splits, then sum over Q tokens.
# ---------------------------------------------------------------------------


@triton.jit
def _maxsim_splitk_phase1(
    Q_ptr,
    D_ptr,
    lengths_ptr,
    partial_ptr,
    B,
    num_splits,
    Lq: tl.constexpr,
    Ld: tl.constexpr,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    stride_d_b,
    stride_d_l,
    stride_d_d,
    stride_q_l,
    stride_q_d,
    stride_p_b,
    stride_p_s,
    stride_p_q,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Phase 1: compute per-split partial max-per-Q-token."""
    pid = tl.program_id(0)
    doc_id = pid // num_splits
    split_id = pid % num_splits

    if doc_id >= B:
        return

    doc_id64 = tl.cast(doc_id, tl.int64)
    doc_len = tl.load(lengths_ptr + doc_id64).to(tl.int32)

    # This split owns tokens [split_start, split_end)
    tokens_per_split = (Ld + num_splits - 1) // num_splits
    split_start = split_id * tokens_per_split
    split_end = tl.minimum(split_start + tokens_per_split, doc_len)

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d

    for q_start in range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        Q_block = tl.load(
            Q_ptr + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float16)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for d_start in range(split_start, split_end, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = (d_off < split_end) & (d_off < doc_len)

            D_ptrs = (
                D_ptr
                + doc_id64 * stride_d_b
                + d_off[:, None] * stride_d_l
                + k_off[None, :] * stride_d_d
            )
            D_block = tl.load(
                D_ptrs,
                mask=d_valid[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float16)

            S = tl.dot(Q_block, tl.trans(D_block))
            S = tl.where(d_valid[None, :], S, float("-inf"))
            m = tl.maximum(m, tl.max(S, axis=1))

        # Write partial maxima to buffer
        partial_ptrs = (
            partial_ptr
            + doc_id64 * stride_p_b
            + split_id * stride_p_s
            + q_off * stride_p_q
        )
        tl.store(
            partial_ptrs,
            tl.where(q_valid, m, float("-inf")),
            mask=q_valid,
        )


@triton.jit
def _maxsim_splitk_phase2(
    partial_ptr,
    scores_ptr,
    B,
    num_splits: tl.constexpr,
    Lq: tl.constexpr,
    stride_p_b,
    stride_p_s,
    stride_p_q,
    BLOCK_Q: tl.constexpr,
):
    """Phase 2: reduce partial maxima over splits, sum over Q tokens."""
    doc_id = tl.program_id(0)
    doc_id64 = tl.cast(doc_id, tl.int64)
    if doc_id >= B:
        return

    score_acc = tl.zeros([], dtype=tl.float32)

    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        # Max over splits for this Q tile
        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        for s in tl.static_range(0, num_splits):
            pm_ptrs = (
                partial_ptr
                + doc_id64 * stride_p_b
                + s * stride_p_s
                + q_off * stride_p_q
            )
            pm = tl.load(
                pm_ptrs,
                mask=q_valid,
                other=float("-inf"),
            ).to(tl.float32)
            m = tl.maximum(m, pm)

        score_acc += tl.sum(tl.where(q_valid, m, 0.0))

    tl.store(scores_ptr + doc_id64, score_acc)


def flash_maxsim_splitk(
    Q: torch.Tensor,
    D: torch.Tensor,
    doc_lengths=None,
    num_splits: int = 4,
) -> torch.Tensor:
    """MaxSim with split-K: splits Ld across CTAs to fill all SMs.

    Best for small B (B << num_SMs) with large Ld — otherwise standard
    kernel uses more CTAs and is already well-occupied.

    Q: [Lq, d], D: [B, Ld, d] -> [B].
    """
    assert Q.dim() == 2 and D.dim() == 3 and Q.shape[1] == D.shape[2]
    Lq, d = Q.shape
    B, Ld, _ = D.shape

    if doc_lengths is None:
        lengths = torch.full((B,), Ld, device=D.device, dtype=torch.int32)
    else:
        lengths = doc_lengths.to(torch.int32).contiguous()

    Q2 = Q.contiguous().half()
    D2 = D.contiguous()
    scores = torch.empty(B, device=Q.device, dtype=torch.float32)
    partial = torch.full(
        (B, num_splits, Lq),
        float("-inf"),
        device=Q.device,
        dtype=torch.float32,
    )

    from .flash_maxsim import _next_pow2

    d_pad = _next_pow2(d)

    # Phase 1: B * num_splits CTAs
    BLOCK_Q = 32
    BLOCK_D = 64
    _maxsim_splitk_phase1[(B * num_splits,)](
        Q2,
        D2,
        lengths,
        partial,
        B,
        num_splits,
        Lq,
        Ld,
        d,
        d_pad,
        D2.stride(0),
        D2.stride(1),
        D2.stride(2),
        Q2.stride(0),
        Q2.stride(1),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        BLOCK_Q=BLOCK_Q,
        BLOCK_D=BLOCK_D,
    )

    # Phase 2: B CTAs — reduce over splits
    _maxsim_splitk_phase2[(B,)](
        partial,
        scores,
        B,
        num_splits,
        Lq,
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        BLOCK_Q=BLOCK_Q,
    )

    return scores
