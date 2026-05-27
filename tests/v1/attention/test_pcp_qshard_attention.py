# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical equivalence for the Q-sharded prefill PCP path.

The new PCP design (`_run_prefill_new_tokens_pcp` in
`vllm/model_executor/layers/attention/mla_attention.py`) splits each
request's Q into a head half and a tail half using DualChunkSwap, runs
FA twice per rank (head against K[: (r+1)*chunk], tail against
K[: (2W-r)*chunk]), then concatenates and reorders the outputs via
`output_restore_idx`.

This test verifies the *math* of that pattern: assembling the per-rank
[head, tail] outputs at the right global Q positions reproduces single-
rank causal FA over the full Q × K. Pure-CPU FP32, no distributed and
no kernels — runs in a few seconds.

This is the natural complement to
``tests/distributed/test_pcp_prefill_qshard.py`` (which verifies the
K/V data-flow round-trip) and to ``test_pcp_in_prefill.py`` (which
verifies the K-shard LSE merge math used by the chunked-context
branch).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


def _causal_fa_q_aligned_to_end(
    q: torch.Tensor,  # [Sq, H, D]
    k: torch.Tensor,  # [Sk, H, D]   (Sq <= Sk)
    v: torch.Tensor,  # [Sk, H, Dv]
    softmax_scale: float,
) -> torch.Tensor:
    """Reference causal varlen attention with Q aligned to the END of K.

    This is the FA-varlen convention used when cu_seqlens_q < cu_seqlens_k:
    Q[i] attends to K[0 .. Sk - Sq + i]. (When Sq == Sk this reduces to
    the standard ``Q[i] attends to K[0..i]`` causal mask.)
    """
    sq, _, _ = q.shape
    sk = k.shape[0]
    assert sq <= sk
    # scores: [H, Sq, Sk]
    scores = (
        torch.einsum("qhd,khd->hqk", q.float(), k.float()) * softmax_scale
    )
    # causal-with-end-alignment mask
    offset = sk - sq
    q_pos = torch.arange(sq).view(1, sq, 1)
    k_pos = torch.arange(sk).view(1, 1, sk)
    mask = k_pos <= (offset + q_pos)
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    # [Sq, H, Dv]
    return torch.einsum("hqk,khd->qhd", weights, v.float())


def _full_causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """Standard causal attention (Sq == Sk): Q[i] attends to K[0..i]."""
    return _causal_fa_q_aligned_to_end(q, k, v, softmax_scale)


def _dualchunkswap_output_restore_idx_single_request(
    pcp_world_size: int, chunk: int
) -> np.ndarray:
    """Mirror ``vllm.v1.attention.backends.utils.get_pcp_query_restore_idx``
    for a single request with a per-rank chunk of size ``chunk``.

    Each rank contributes ``2*chunk`` outputs in [head, tail] order. The
    full-Q output is laid out per-rank as:

        global_Q[r*chunk : (r+1)*chunk]            <- rank r's head
        global_Q[(2W-r-1)*chunk : (2W-r)*chunk]    <- rank r's tail
    """
    # Construct the list of global Q positions in the all-gather order
    # (heads of each rank in rank order, then tails of each rank).
    # NOTE: This is the SINGLE-RANK restore_idx; we use it for the
    # in-process all-ranks simulation by treating concat across ranks
    # as one big tensor.
    heads = []
    tails = []
    for r in range(pcp_world_size):
        heads.append(np.arange(r * chunk, (r + 1) * chunk))
        tails.append(
            np.arange(
                (2 * pcp_world_size - r - 1) * chunk,
                (2 * pcp_world_size - r) * chunk,
            )
        )
    all_pos = np.concatenate(heads + tails)
    # argsort: where does each global position land in the all-gather order?
    return all_pos.argsort()


@pytest.mark.parametrize("pcp_world_size", [2, 4, 8])
@pytest.mark.parametrize("chunk", [1, 4, 16])
@pytest.mark.parametrize("num_heads", [1, 4, 16])
@pytest.mark.parametrize("head_dim,v_head_dim", [(32, 32), (64, 64), (128, 128)])
def test_pcp_qshard_attention_matches_single_rank(
    pcp_world_size: int,
    chunk: int,
    num_heads: int,
    head_dim: int,
    v_head_dim: int,
):
    """DualChunkSwap PCP attention output must equal single-rank causal FA.

    Setup: 1 request of length S = 2 * pcp_world_size * chunk (no padding).
    For each rank r:
      - Q_head = Q[r*chunk : (r+1)*chunk]
      - Q_tail = Q[(2W-r-1)*chunk : (2W-r)*chunk]
      - K_head_win = K[: (r+1)*chunk],  V_head_win = V[: (r+1)*chunk]
      - K_tail_win = K[: (2W-r)*chunk], V_tail_win = V[: (2W-r)*chunk]
      - out_head = causal_fa(Q_head, K_head_win, V_head_win)
      - out_tail = causal_fa(Q_tail, K_tail_win, V_tail_win)

    Assembling the per-rank outputs at the right global Q positions must
    reproduce single-rank causal FA over the full Q × K. This is the
    invariant the new ``_run_prefill_new_tokens_pcp`` path is built on.
    """
    torch.manual_seed(0)
    seq_len = 2 * pcp_world_size * chunk
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(seq_len, num_heads, head_dim)
    k = torch.randn(seq_len, num_heads, head_dim)
    v = torch.randn(seq_len, num_heads, v_head_dim)

    # Reference: single-rank causal FA over the full sequence.
    out_ref = _full_causal_attention(q, k, v, softmax_scale)

    # PCP path: per-rank head/tail FA, assemble in global Q order.
    out_assembled = torch.zeros_like(out_ref)
    for r in range(pcp_world_size):
        head_start, head_end = r * chunk, (r + 1) * chunk
        tail_start = (2 * pcp_world_size - r - 1) * chunk
        tail_end = (2 * pcp_world_size - r) * chunk

        q_head = q[head_start:head_end]
        q_tail = q[tail_start:tail_end]
        k_head_win = k[: (r + 1) * chunk]
        v_head_win = v[: (r + 1) * chunk]
        k_tail_win = k[: (2 * pcp_world_size - r) * chunk]
        v_tail_win = v[: (2 * pcp_world_size - r) * chunk]

        out_head = _causal_fa_q_aligned_to_end(
            q_head, k_head_win, v_head_win, softmax_scale
        )
        out_tail = _causal_fa_q_aligned_to_end(
            q_tail, k_tail_win, v_tail_win, softmax_scale
        )

        out_assembled[head_start:head_end] = out_head
        out_assembled[tail_start:tail_end] = out_tail

    # FP32 tolerance.
    diff = (out_assembled - out_ref).abs().max().item()
    assert torch.allclose(out_assembled, out_ref, atol=1e-5, rtol=1e-5), (
        f"PCP Q-shard output diverges from single-rank reference: "
        f"max abs diff = {diff} "
        f"(pcp_world_size={pcp_world_size}, chunk={chunk}, "
        f"num_heads={num_heads}, head_dim={head_dim})"
    )


@pytest.mark.parametrize("pcp_world_size", [2, 4])
def test_pcp_qshard_restore_idx_matches_helper(pcp_world_size: int):
    """The in-process restore_idx construction matches
    ``get_pcp_query_restore_idx`` for a single request.
    """
    from vllm.v1.attention.backends.utils import get_pcp_query_restore_idx

    chunk = 4
    local_len = 2 * chunk  # per-rank local Q length (head + tail)
    cu_num_tokens = torch.tensor([0, local_len], dtype=torch.int32)
    helper_idx = get_pcp_query_restore_idx(cu_num_tokens).numpy()

    # The helper's restore_idx operates on a SINGLE rank's [head, tail]
    # concat of length 2*chunk. For one request with halves of size chunk,
    # head positions = [0..chunk-1], tail positions = [chunk..2*chunk-1].
    # Concat = [0..2*chunk-1]; argsort = identity.
    expected = np.arange(local_len, dtype=np.int32)
    assert np.array_equal(helper_idx, expected), (
        f"single-request restore_idx not identity: {helper_idx.tolist()}"
    )


@pytest.mark.parametrize("pcp_world_size", [2, 4])
def test_pcp_qshard_restore_idx_two_requests(pcp_world_size: int):
    """Two-request restore_idx interleaves [req0 heads, req1 heads,
    req0 tails, req1 tails] back to [req0_full, req1_full] (per-request
    [head, tail] contiguous).
    """
    from vllm.v1.attention.backends.utils import get_pcp_query_restore_idx

    chunk0, chunk1 = 4, 6
    L0, L1 = 2 * chunk0, 2 * chunk1
    cu = torch.tensor([0, L0, L0 + L1], dtype=torch.int32)
    idx = get_pcp_query_restore_idx(cu).numpy()

    # Construction (mirrors helper):
    #   heads:  req0 positions [0..chunk0-1]    + req1 positions [L0..L0+chunk1-1]
    #   tails:  req0 positions [chunk0..L0-1]  + req1 positions [L0+chunk1..L0+L1-1]
    # concat = heads ++ tails. argsort gives the inverse permutation.
    heads = np.concatenate(
        [np.arange(0, chunk0), np.arange(L0, L0 + chunk1)]
    )
    tails = np.concatenate(
        [np.arange(chunk0, L0), np.arange(L0 + chunk1, L0 + L1)]
    )
    expected = np.concatenate([heads, tails]).argsort().astype(np.int32)
    assert np.array_equal(idx, expected), (
        f"two-request restore_idx mismatch: got {idx.tolist()}, "
        f"expected {expected.tolist()}"
    )


@pytest.mark.parametrize("pcp_world_size", [2, 4])
def test_pcp_qshard_attention_uneven_request_lengths(pcp_world_size: int):
    """Two requests with different lengths — exercises the per-request
    chunk computation. Both requests are divisible by 2*pcp_world_size
    (avoid padding here; padding is handled by the K/V roundtrip test).
    """
    torch.manual_seed(0)
    num_heads, head_dim = 4, 32
    softmax_scale = 1.0 / math.sqrt(head_dim)
    # Two requests; each length divisible by 2*ws so chunks are integer.
    lens = [2 * pcp_world_size * 3, 2 * pcp_world_size * 5]
    total = sum(lens)

    q = torch.randn(total, num_heads, head_dim)
    k = torch.randn(total, num_heads, head_dim)
    v = torch.randn(total, num_heads, head_dim)

    # Reference: causal FA per request.
    out_ref = torch.zeros_like(q)
    offset = 0
    for n in lens:
        out_ref[offset : offset + n] = _full_causal_attention(
            q[offset : offset + n],
            k[offset : offset + n],
            v[offset : offset + n],
            softmax_scale,
        )
        offset += n

    # PCP path: per-rank head/tail FA, per-request.
    out_assembled = torch.zeros_like(out_ref)
    offset = 0
    for n in lens:
        chunk = n // (2 * pcp_world_size)
        for r in range(pcp_world_size):
            head_start = offset + r * chunk
            head_end = offset + (r + 1) * chunk
            tail_start = offset + (2 * pcp_world_size - r - 1) * chunk
            tail_end = offset + (2 * pcp_world_size - r) * chunk

            q_head = q[head_start:head_end]
            q_tail = q[tail_start:tail_end]
            k_req = k[offset : offset + n]
            v_req = v[offset : offset + n]
            k_head_win = k_req[: (r + 1) * chunk]
            v_head_win = v_req[: (r + 1) * chunk]
            k_tail_win = k_req[: (2 * pcp_world_size - r) * chunk]
            v_tail_win = v_req[: (2 * pcp_world_size - r) * chunk]

            out_head = _causal_fa_q_aligned_to_end(
                q_head, k_head_win, v_head_win, softmax_scale
            )
            out_tail = _causal_fa_q_aligned_to_end(
                q_tail, k_tail_win, v_tail_win, softmax_scale
            )

            out_assembled[head_start:head_end] = out_head
            out_assembled[tail_start:tail_end] = out_tail

        offset += n

    diff = (out_assembled - out_ref).abs().max().item()
    assert torch.allclose(out_assembled, out_ref, atol=1e-5, rtol=1e-5), (
        f"Uneven-length PCP Q-shard output diverges from reference: "
        f"max abs diff = {diff}"
    )
