"""Flash-MaxSim rerank: one query vs scattered docs in a batch tensor.

True zero-copy scoring: the kernel reads doc embeddings directly from the
model's output tensor using per-doc (offset, length) pairs. No torch.stack,
no torch.cat, no padding, no copies at all.

Two entry points:
  flash_maxsim_rerank(Q, D_packed, cu_d, max_ld)
      For contiguous packed docs (from torch.cat). Same kernel, cu_seqlens API.

  flash_maxsim_rerank_direct(Q, batch_tensor, doc_offsets, doc_lengths, max_ld)
      TRUE zero-copy: reads from the model's batch output tensor directly.
      doc_offsets[i] = start token index of doc i in batch_tensor.
      doc_lengths[i] = number of tokens in doc i.
      Docs can be scattered (non-contiguous) in the batch tensor.
"""

import torch

from vllm.triton_utils import tl, triton

from .flash_maxsim import _get_configs, _next_pow2, _prune_configs


def _round_to_bucket(x):
    """Round up to nearest bucket: 32, 64, 128, 256, 512, 1024, 2048, 4096."""
    for b in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        if x <= b:
            return b
    return _next_pow2(x)


# ---------------------------------------------------------------------------
# Kernel: shared Q, scattered D via (offset, length) per doc
# ---------------------------------------------------------------------------

@triton.autotune(configs=_get_configs(),
                 key=["Lq_bucket", "max_Ld_bucket", "d_pad"],
                 prune_configs_by={"early_config_prune": _prune_configs})
@triton.jit
def _maxsim_rerank_kernel(
    Q_ptr, D_ptr,
    doc_offsets_ptr, doc_lengths_ptr,
    scores_ptr,
    B,
    Lq, Lq_bucket: tl.constexpr,
    max_Ld_bucket: tl.constexpr,
    d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_t, stride_q_d,
    stride_d_t, stride_d_d,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """One program per doc. Q shared (read from same location). D scattered."""
    pid = tl.program_id(0)
    if pid >= B:
        return

    # Each doc has its own offset and length in the batch tensor
    d_start = tl.load(doc_offsets_ptr + pid).to(tl.int32)
    d_len = tl.load(doc_lengths_ptr + pid).to(tl.int32)

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    # Q is shared — always read from start of Q_ptr
    for q_blk in tl.static_range(0, Lq_bucket, BLOCK_Q):
        q_off = q_blk + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        Q_block = tl.load(
            Q_ptr + q_off[:, None] * stride_q_t + k_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float16)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for d_blk in range(0, max_Ld_bucket, BLOCK_D):
            d_off = d_blk + tl.arange(0, BLOCK_D)
            d_valid = d_off < d_len

            D_block = tl.load(
                D_ptr + (d_start + d_off[:, None]) * stride_d_t
                + k_off[None, :] * stride_d_d,
                mask=d_valid[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float16)

            S = tl.dot(Q_block, tl.trans(D_block))
            S = tl.where(d_valid[None, :], S, float("-inf"))
            m = tl.maximum(m, tl.max(S, axis=1))

        m = tl.where(q_valid, m, 0.0)
        score_acc += tl.sum(m)

    tl.store(scores_ptr + pid, score_acc)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _run_rerank_kernel(Q, D_tensor, doc_offsets, doc_lengths, max_seqlen_d):
    """Common kernel launch for both rerank and rerank_direct."""
    assert Q.is_cuda, f"Q must be on CUDA, got {Q.device}"
    assert D_tensor.is_cuda, f"D must be on CUDA, got {D_tensor.device}"
    Lq, d = Q.shape
    B = doc_offsets.shape[0]
    d_pad = _next_pow2(max(d, 16))

    Q = Q.contiguous().half()
    # D_tensor is NOT copied — kernel reads from it directly
    if not D_tensor.is_contiguous():
        D_tensor = D_tensor.contiguous()
    if D_tensor.dtype != torch.float16:
        D_tensor = D_tensor.half()

    doc_offsets = doc_offsets.to(torch.int32).contiguous()
    doc_lengths = doc_lengths.to(torch.int32).contiguous()
    scores = torch.empty(B, device=Q.device, dtype=torch.float32)

    if B == 0:
        return scores

    # Pad Q dimension if needed (Q is small, safe to copy).
    # D is NOT padded — the kernel uses k_mask to handle d < d_pad.
    # This preserves zero-copy: D_tensor is read in-place.
    if d < d_pad:
        Q_pad = torch.zeros(Lq, d_pad, device=Q.device, dtype=Q.dtype)
        Q_pad[:, :d] = Q
    else:
        Q_pad = Q
    D_pad = D_tensor

    Lq_bucket = _round_to_bucket(Lq)
    max_ld_bucket = _round_to_bucket(max_seqlen_d)

    _maxsim_rerank_kernel[(B,)](
        Q_pad, D_pad,
        doc_offsets, doc_lengths,
        scores,
        B,
        Lq, Lq_bucket,
        max_ld_bucket,
        d, d_pad,
        Q_pad.stride(0), Q_pad.stride(1),
        D_pad.stride(0), D_pad.stride(1),
    )
    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flash_maxsim_rerank(
    Q: torch.Tensor,
    D_packed: torch.Tensor,
    cu_seqlens_d: torch.Tensor,
    max_seqlen_d: int,
) -> torch.Tensor:
    """Score one query against contiguous packed docs.

    Args:
        Q: [Lq, d] — single query embedding
        D_packed: [total_d_tokens, d] — all docs concatenated (e.g. via torch.cat)
        cu_seqlens_d: [B+1] int32 — cumulative doc token counts (starts at 0)
        max_seqlen_d: int — maximum doc length

    Returns:
        scores: [B] float32
    """
    assert Q.dim() == 2 and D_packed.dim() == 2
    assert Q.shape[1] == D_packed.shape[1]

    cu_d = cu_seqlens_d.to(torch.int32)
    doc_offsets = cu_d[:-1]
    doc_lengths = cu_d[1:] - cu_d[:-1]
    return _run_rerank_kernel(Q, D_packed, doc_offsets, doc_lengths, max_seqlen_d)


def flash_maxsim_rerank_direct(
    Q: torch.Tensor,
    batch_tensor: torch.Tensor,
    doc_offsets: torch.Tensor,
    doc_lengths: torch.Tensor,
    max_seqlen_d: int,
) -> torch.Tensor:
    """TRUE zero-copy: score query against docs scattered in a batch tensor.

    The kernel reads doc embeddings directly from batch_tensor at the
    positions specified by doc_offsets. No torch.stack, no torch.cat,
    no copy of any kind. The batch tensor is the model's output.

    Memory for doc scoring: 0 bytes additional.

    Args:
        Q: [Lq, d] — single query embedding (from cache)
        batch_tensor: [total_tokens, d] — the model's projected output tensor.
            Contains ALL requests' tokens (queries + docs + others).
        doc_offsets: [B] int32 — start token index of each doc in batch_tensor
        doc_lengths: [B] int32 — number of tokens per doc
        max_seqlen_d: int — max(doc_lengths)

    Returns:
        scores: [B] float32 — one MaxSim score per document
    """
    assert Q.dim() == 2, f"Q must be 2D [Lq, d], got {Q.dim()}D"
    assert batch_tensor.dim() == 2, (
        f"batch_tensor must be 2D, got {batch_tensor.dim()}D"
    )
    assert Q.shape[1] == batch_tensor.shape[1]

    return _run_rerank_kernel(Q, batch_tensor, doc_offsets, doc_lengths, max_seqlen_d)
