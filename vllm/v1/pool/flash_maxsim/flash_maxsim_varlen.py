# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-MaxSim with variable-length sequences (no padding).

Inspired by flash_attn_varlen_func: sequences packed contiguously,
cu_seqlens marks boundaries. Zero padding overhead, zero wasted compute.

Usage:
    from flash_maxsim import flash_maxsim_varlen

    # Pack variable-length pairs
    Q_packed, D_packed, cu_q, cu_d = pack_pairs(q_embs, d_embs)
    scores = flash_maxsim_varlen(Q_packed, D_packed, cu_q, cu_d, max_lq, max_ld)
"""

import torch

from vllm.triton_utils import tl, triton

from .flash_maxsim import _get_configs, _next_pow2, _prune_configs

# ---------------------------------------------------------------------------
# Varlen kernel: one program per pair, uses cu_seqlens for boundaries
# ---------------------------------------------------------------------------


def _round_to_bucket(x):
    """Round up to nearest bucket: 32, 64, 128, 256, 512, 1024, 2048, 4096."""
    for b in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        if x <= b:
            return b
    return _next_pow2(x)


@triton.autotune(
    configs=_get_configs(),
    key=["max_Lq_bucket", "max_Ld_bucket", "d_pad"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _maxsim_varlen_kernel(
    Q_ptr,
    D_ptr,
    cu_q_ptr,
    cu_d_ptr,
    scores_ptr,
    N: tl.constexpr,
    max_Lq_bucket: tl.constexpr,
    max_Ld_bucket: tl.constexpr,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    stride_q_t,
    stride_q_d,
    stride_d_t,
    stride_d_d,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per pair. Q and D packed contiguously."""
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load boundaries from cu_seqlens
    q_start = tl.load(cu_q_ptr + pid).to(tl.int32)
    q_end = tl.load(cu_q_ptr + pid + 1).to(tl.int32)
    d_start = tl.load(cu_d_ptr + pid).to(tl.int32)
    d_end = tl.load(cu_d_ptr + pid + 1).to(tl.int32)
    q_len = q_end - q_start
    d_len = d_end - d_start

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    # Loop bounds use bucketed max — slightly over-iterates but
    # q_valid/d_valid masks handle correctness. Bucket ensures
    # autotune compiles only ~8 variants per dimension, not per unique length.
    for q_blk in tl.static_range(0, max_Lq_bucket, BLOCK_Q):
        q_off = q_blk + tl.arange(0, BLOCK_Q)
        q_valid = q_off < q_len

        Q_ptrs = (
            Q_ptr
            + (q_start + q_off[:, None]) * stride_q_t
            + k_off[None, :] * stride_q_d
        )
        Q_block = tl.load(
            Q_ptrs,
            mask=q_valid[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float16)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for d_blk in range(0, max_Ld_bucket, BLOCK_D):
            d_off = d_blk + tl.arange(0, BLOCK_D)
            d_valid = d_off < d_len

            D_ptrs = (
                D_ptr
                + (d_start + d_off[:, None]) * stride_d_t
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

        m = tl.where(q_valid, m, 0.0)
        score_acc += tl.sum(m)

    tl.store(scores_ptr + pid, score_acc)


# ---------------------------------------------------------------------------
# Shared-Q + packed-D kernel: one query scored against B variable-length docs
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_configs(),
    key=["Lq_bucket", "d_pad"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _maxsim_packed_kernel(
    Q_ptr,
    D_ptr,
    cu_d_ptr,
    scores_ptr,
    Nq,
    B,
    Lq_bucket: tl.constexpr,
    Lq,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    stride_q_n,
    stride_q_l,
    stride_q_d,
    stride_d_t,
    stride_d_d,
    stride_s_n,
    stride_s_b,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per (q_chunk, doc) pair. Q batched, D packed."""
    pid = tl.program_id(0)
    q_idx = pid // B
    doc_id = pid % B
    if q_idx >= Nq:
        return

    d_start = tl.load(cu_d_ptr + doc_id).to(tl.int64)
    d_end = tl.load(cu_d_ptr + doc_id + 1).to(tl.int64)
    doc_len = (d_end - d_start).to(tl.int32)

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    for q_start in range(0, Lq_bucket, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        Q_ptrs = (
            Q_ptr
            + q_idx * stride_q_n
            + q_off[:, None] * stride_q_l
            + k_off[None, :] * stride_q_d
        )
        Q_block = tl.load(
            Q_ptrs,
            mask=q_valid[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float16)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        # D loop: runtime doc_len bound — short docs iterate fewer times.
        # Same pattern as the persistent kernel. No constexpr bucket needed.
        for d_blk in range(0, doc_len, BLOCK_D):
            d_off = d_blk + tl.arange(0, BLOCK_D)
            d_valid = d_off < doc_len

            D_ptrs = (
                D_ptr
                + (d_start + d_off)[:, None] * stride_d_t
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

        m = tl.where(q_valid, m, 0.0)
        score_acc += tl.sum(m)

    tl.store(scores_ptr + q_idx * stride_s_n + doc_id * stride_s_b, score_acc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def flash_maxsim_packed(
    Q: torch.Tensor,
    D_packed: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
    max_seqlen_d: int | None = None,
    query_chunk_size: int = 128,
) -> torch.Tensor:
    """MaxSim: one query against B variable-length packed docs.

    Like flash_maxsim but D is packed contiguously with no padding.
    Saves both HBM storage and bandwidth — the kernel never touches padding.

    Args:
        Q:           [Lq, d]  — single query (shared across all docs)
        D_packed:    [total_d_tokens, d] — all docs packed contiguously
        cu_seqlens_d:[B+1] int32 — cumulative doc lengths (starts at 0)
        max_seqlen_d: max doc length (auto-computed from cu_seqlens_d if None)
        query_chunk_size: chunk Q for better occupancy (default 128)

    Returns:
        scores: [B] float32
    """
    assert Q.dim() == 2 and D_packed.dim() == 2
    assert Q.shape[1] == D_packed.shape[1]

    Lq, d = Q.shape
    d_pad = _next_pow2(max(d, 16))
    cu_seqlens_d = cu_seqlens_d.to(torch.int32).contiguous()
    B = cu_seqlens_d.shape[0] - 1

    Q = Q.contiguous().half()
    D_packed = D_packed.contiguous().half()

    # Query chunking: split Q into [Nq, C, d] batch, ONE kernel launch
    if query_chunk_size is not None and Lq > query_chunk_size:
        C = query_chunk_size
        Nq = (Lq + C - 1) // C
        if Lq % C != 0:
            Q = torch.nn.functional.pad(Q, (0, 0, 0, Nq * C - Lq))
        Q_chunked = Q.view(Nq, C, d)  # [Nq, C, d]
        actual_Lq = C  # each chunk is exactly C (last chunk padded, masked by q_valid)
    else:
        Q_chunked = Q.unsqueeze(0)  # [1, Lq, d]
        Nq = 1
        actual_Lq = Lq

    scores = torch.empty(Nq, B, device=Q.device, dtype=torch.float32)

    # Pad d to d_pad if needed
    if d < d_pad:
        Q_pad = torch.zeros(Nq, actual_Lq, d_pad, device=Q.device, dtype=Q.dtype)
        Q_pad[:, :, :d] = Q_chunked
        D_pad = torch.zeros(
            D_packed.shape[0],
            d_pad,
            device=D_packed.device,
            dtype=D_packed.dtype,
        )
        D_pad[:, :d] = D_packed
    else:
        Q_pad = Q_chunked
        D_pad = D_packed

    lq_bucket = _round_to_bucket(actual_Lq)

    _maxsim_packed_kernel[(Nq * B,)](
        Q_pad,
        D_pad,
        cu_seqlens_d,
        scores,
        Nq,
        B,
        lq_bucket,
        actual_Lq,
        d,
        d_pad,
        Q_pad.stride(0),
        Q_pad.stride(1),
        Q_pad.stride(2),
        D_pad.stride(0),
        D_pad.stride(1),
        scores.stride(0),
        scores.stride(1),
    )
    return scores.squeeze(0) if Nq == 1 else scores.sum(dim=0)  # [Nq, B] -> [B]


def pack_docs(d_embs: list):
    """Pack variable-length docs into a contiguous buffer with cu_seqlens.

    Args:
        d_embs: list of [Ld_i, d] tensors

    Returns:
        D_packed:     [total_d, d]
        cu_seqlens_d: [B+1] int32
        max_seqlen_d: int
    """
    B = len(d_embs)
    device = d_embs[0].device
    d_lens = [dd.shape[0] for dd in d_embs]
    d_lens_t = torch.tensor(d_lens, device=device, dtype=torch.int32)
    cu_d = torch.zeros(B + 1, device=device, dtype=torch.int32)
    cu_d[1:] = d_lens_t.cumsum(0)
    D_packed = torch.cat(d_embs, dim=0)
    return D_packed, cu_d, max(d_lens)


def flash_maxsim_varlen(
    Q_packed: torch.Tensor,
    D_packed: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_d: int,
) -> torch.Tensor:
    """MaxSim with packed variable-length sequences (no padding).

    Args:
        Q_packed: [total_q_tokens, d] — all queries packed contiguously
        D_packed: [total_d_tokens, d] — all docs packed contiguously
        cu_seqlens_q: [N+1] int32 — cumulative query lengths (starts at 0)
        cu_seqlens_d: [N+1] int32 — cumulative doc lengths (starts at 0)
        max_seqlen_q: max query length (for kernel loop bound)
        max_seqlen_d: max doc length (for kernel loop bound)

    Returns:
        scores: [N] float32 — one score per pair
    """
    assert Q_packed.dim() == 2 and D_packed.dim() == 2
    assert Q_packed.shape[1] == D_packed.shape[1]
    assert cu_seqlens_q.shape[0] == cu_seqlens_d.shape[0]

    N = cu_seqlens_q.shape[0] - 1
    d = Q_packed.shape[1]
    d_pad = _next_pow2(max(d, 16))

    Q_packed = Q_packed.contiguous().half()
    D_packed = D_packed.contiguous().half()
    cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    cu_seqlens_d = cu_seqlens_d.to(torch.int32).contiguous()
    scores = torch.empty(N, device=Q_packed.device, dtype=torch.float32)

    # Pad Q/D to d_pad if needed
    if d < d_pad:
        Q_pad = torch.zeros(
            Q_packed.shape[0],
            d_pad,
            device=Q_packed.device,
            dtype=Q_packed.dtype,
        )
        Q_pad[:, :d] = Q_packed
        D_pad = torch.zeros(
            D_packed.shape[0],
            d_pad,
            device=D_packed.device,
            dtype=D_packed.dtype,
        )
        D_pad[:, :d] = D_packed
    else:
        Q_pad = Q_packed
        D_pad = D_packed

    # Bucket max lengths to reduce autotune compilations
    # e.g., max_Lq=27 → bucket=32, max_Lq=100 → bucket=128
    max_lq_bucket = _round_to_bucket(max_seqlen_q)
    max_ld_bucket = _round_to_bucket(max_seqlen_d)

    _maxsim_varlen_kernel[(N,)](
        Q_pad,
        D_pad,
        cu_seqlens_q,
        cu_seqlens_d,
        scores,
        N,
        max_lq_bucket,
        max_ld_bucket,
        d,
        d_pad,
        Q_pad.stride(0),
        Q_pad.stride(1),
        D_pad.stride(0),
        D_pad.stride(1),
    )
    return scores


def pack_pairs(q_embs: list, d_embs: list):
    """Pack variable-length pairs into contiguous buffers with cu_seqlens.

    Args:
        q_embs: list of [Lq_i, d] tensors
        d_embs: list of [Ld_i, d] tensors

    Returns:
        Q_packed: [total_q, d]
        D_packed: [total_d, d]
        cu_seqlens_q: [N+1]
        cu_seqlens_d: [N+1]
        max_lq: int
        max_ld: int
    """
    N = len(q_embs)
    device = q_embs[0].device

    q_lens = [q.shape[0] for q in q_embs]
    d_lens = [dd.shape[0] for dd in d_embs]

    q_lens_t = torch.tensor(q_lens, device=device, dtype=torch.int32)
    d_lens_t = torch.tensor(d_lens, device=device, dtype=torch.int32)
    cu_q = torch.zeros(N + 1, device=device, dtype=torch.int32)
    cu_d = torch.zeros(N + 1, device=device, dtype=torch.int32)
    cu_q[1:] = q_lens_t.cumsum(0)
    cu_d[1:] = d_lens_t.cumsum(0)

    # torch.cat is faster than manual copy loop
    Q_packed = torch.cat(q_embs, dim=0)
    D_packed = torch.cat(d_embs, dim=0)

    max_lq = max(q_lens)
    max_ld = max(d_lens)

    return Q_packed, D_packed, cu_q, cu_d, max_lq, max_ld
