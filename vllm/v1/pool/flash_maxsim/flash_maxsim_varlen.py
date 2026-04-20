# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash-MaxSim: shared query vs B variable-length packed docs.

D is packed contiguously (via torch.cat) with a cu_seqlens[B+1] index;
the kernel skips padding tokens entirely.
"""

import torch

from vllm.triton_utils import tl, triton

from ._common import _get_configs, _next_pow2, _prune_configs


def _round_to_bucket(x):
    """Round up to nearest bucket: 32, 64, 128, 256, 512, 1024, 2048, 4096."""
    for b in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        if x <= b:
            return b
    return _next_pow2(x)


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
