# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Ring Attention for Context Parallelism.

Implements Ring Attention where Q stays local while K/V circulate through
a ring of CP ranks.  Each step computes a partial attention block and
incrementally merges the result using online softmax correction.

This module is a **communication framework**, not a new attention kernel.
It calls the platform's existing flash-attention varlen implementation
(resolved at import time by ``fa_utils``) inside a ring-topology P2P loop.

Performance optimizations:
  - K and V are packed into a single contiguous buffer for P2P transfer,
    halving the number of P2P operations per step.
  - Double buffering: two pre-allocated receive buffers are alternated
    across steps, eliminating per-step memory allocation.
  - The CUDA stream for P2P communication is created once and cached
    (``hipStreamCreate`` costs ~2ms on ROCm; caching avoids this per-call).
  - The ``RingComm``'s dedicated CUDA stream enables overlap between
    P2P communication and attention computation.

Known limitations:
  - Sequence sharding assumes contiguous partitioning (rank i gets the
    i-th chunk).  Stripe/zigzag partitioning is not supported.
  - Not compatible with ``torch.compile`` due to ``dist.batch_isend_irecv``,
    Python-level ring loop, and explicit CUDA stream management.
  - Does not manage KV cache; callers are responsible for reading/writing
    paged KV cache before/after calling these functions.

References:
    - vllm-omni diffusion/attention/backends/ring_flash_attn.py
    - Fang et al., long-context-attention (yunchang)
    - Liu et al., "Ring Attention with Blockwise Transformers" (2023)
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm.distributed.ring_comm import RingComm
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
)

# ---------------------------------------------------------------------------
# Cached CUDA stream for P2P communication.
# hipStreamCreate costs ~2ms on ROCm; caching avoids this on every call.
# TODO: When integrated into a PCP attention backend, create the stream
# at model init time and pass it in via cp_stream instead of using this
# module-level cache.  The module-level cache is bound to the device
# that is current at first call; this is safe in vLLM where each worker
# process owns a single device.
# ---------------------------------------------------------------------------
_cp_stream: torch.cuda.Stream | None = None


def _get_cp_stream() -> torch.cuda.Stream:
    global _cp_stream
    if _cp_stream is None:
        _cp_stream = torch.cuda.Stream()
    return _cp_stream


# ---------------------------------------------------------------------------
# Online softmax merge for varlen packed layout [N, H, D]
# ---------------------------------------------------------------------------


def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge a new attention block into the running output.

    Uses the numerically stable online softmax correction:

        out = out - sigmoid(block_lse - lse) * (out - block_out)
        lse = lse - logsigmoid(lse - block_lse)

    All computation is done in float32 for numerical stability.

    Args:
        out: Running output ``[N, H, D]`` (float32).
        lse: Running log-sum-exp ``[N, H, 1]`` (float32).
        block_out: New block output ``[N, H, D]``.
        block_lse: New block LSE ``[H, N]`` (float32,
            raw flash_attn_varlen_func output layout).
    """
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(0, 1).unsqueeze(-1)  # [H,N] -> [N,H,1]

    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse


def _init_out_and_lse(
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize running output and LSE from the first block.

    Args:
        block_out: ``[N, H, D]``.
        block_lse: ``[H, N]`` (float32).

    Returns:
        (out, lse) -- out ``[N, H, D]`` fp32, lse ``[N, H, 1]`` fp32.
    """
    out = block_out.to(torch.float32)
    lse = block_lse.transpose(0, 1).unsqueeze(-1)
    return out, lse


# ---------------------------------------------------------------------------
# Ring Flash Attention -- varlen packed interface
# ---------------------------------------------------------------------------


def ring_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    cp_group: dist.ProcessGroup,
    cp_stream: torch.cuda.Stream | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Ring Attention for packed variable-length sequences.

    Q stays local while K/V circulate through a ring of CP ranks.
    Each rank holds a contiguous chunk of every request's tokens;
    ``cu_seqlens_q`` and ``cu_seqlens_k`` describe the **local**
    (post-sharding) packed layout.

    Supports GQA (K/V may have fewer heads than Q) -- the underlying
    flash-attention kernel handles KV head repeat internally.

    Note:
        Each request's sequence length must be evenly divisible by
        the CP world size.  The caller is responsible for padding or
        filtering requests that do not meet this requirement.

    For causal attention, the standard Ring Attention schedule is used:
    rank *i* only attends to KV chunks from ranks 0..i.  Step 0 (the
    local chunk) uses ``causal=True``; subsequent steps use full
    (non-causal) attention since those KV chunks are strictly earlier
    in the sequence.

    Args:
        q: Query ``[total_q_tokens, num_q_heads, D]``.
        k: Key   ``[total_k_tokens, num_kv_heads, D]``.
        v: Value ``[total_k_tokens, num_kv_heads, D]``.
        cu_seqlens_q: Cumulative local query lengths ``[B+1]``.
        cu_seqlens_k: Cumulative local key lengths ``[B+1]``.
        max_seqlen_q: Max local query length in batch.
        max_seqlen_k: Max local key length in batch.
        cp_group: Process group for context parallelism.
        cp_stream: Dedicated CUDA stream for P2P comm.  If *None*,
            uses a module-level cached stream.
        softmax_scale: Attention scale.  Defaults to ``1/sqrt(D)``.
        causal: Whether to apply causal masking.

    Returns:
        Output ``[total_q_tokens, num_q_heads, D]`` in original dtype.
    """
    if cp_stream is None:
        cp_stream = _get_cp_stream()

    comm = RingComm(cp_group, cp_stream)

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Pack K and V into a single contiguous buffer for P2P transfer,
    # halving the number of P2P operations per ring step (2 vs 4).
    kv = torch.cat([k, v], dim=0)

    # Double buffering: pre-allocate two receive buffers and alternate
    # across steps for zero per-step memory allocation.
    recv_bufs = (torch.empty_like(kv), torch.empty_like(kv))

    out: torch.Tensor | None = None
    lse: torch.Tensor | None = None

    for step in range(comm.world_size):
        if step + 1 < comm.world_size:
            next_kv = comm.send_recv(kv, recv_tensor=recv_bufs[step % 2])
            comm.commit()

        k, v = kv.chunk(2, dim=0)

        if not causal or step <= comm.rank:
            # NOTE: return_attn_probs=True is the only way to obtain the
            # softmax log-sum-exp (LSE) from this flash_attn version.
            # Despite the name, it does NOT compute the full O(S^2) attention
            # probability matrix; the third return value is always an empty
            # tensor.  LSE is returned as the second value with shape [H, N].
            block_out, block_lse, _ = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
                return_attn_probs=True,
            )

            if out is None:
                out, lse = _init_out_and_lse(block_out, block_lse)
            else:
                out, lse = _update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 < comm.world_size:
            comm.wait()
            kv = next_kv

    assert out is not None
    return out.to(q.dtype)
