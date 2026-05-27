# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the math underlying PCP/DCP-in-prefill: splitting K across
N ranks, running FA on each rank's local K shard, and LSE-merging the
partials produces outputs numerically equal to single-rank FA over the
full K.

This is the core correctness invariant our patched
`_context_parallel_compute_prefill_context` relies on. Heads are kept
consistent across CP ranks (TP=1 contract), so the per-rank partial
is "(all heads) x (1/N of K positions)", which is exactly what an
LSE-weighted online-softmax merge handles.

The test does NOT spawn multiple processes — it computes the partials
locally and merges them in-process, which is mathematically identical
to what cp_lse_ag_out_ar's AllReduce does across ranks.
"""

from __future__ import annotations

import math

import pytest
import torch


def _naive_attention(
    q: torch.Tensor,  # [Sq, H, D]
    k: torch.Tensor,  # [Sk, H, D]
    v: torch.Tensor,  # [Sk, H, Dv]
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference single-rank attention. Returns (out [Sq,H,Dv], lse [H,Sq])."""
    # scores: [H, Sq, Sk]
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * softmax_scale
    lse = torch.logsumexp(scores, dim=-1)  # [H, Sq]
    weights = torch.exp(scores - lse.unsqueeze(-1))  # [H, Sq, Sk]
    # out: [Sq, H, Dv]
    out = torch.einsum("hqk,khd->qhd", weights, v.float())
    return out, lse


def _lse_weighted_merge(
    outs: list[torch.Tensor],  # each [Sq, H, Dv]
    lses: list[torch.Tensor],  # each [H, Sq]
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-process equivalent of cp_lse_ag_out_ar across N ranks.

    Given N partials (each computed against a disjoint K shard),
    compute the global attention output via LSE-weighted sum.
    """
    # Stack to [N, H, Sq] and [N, Sq, H, Dv]
    lses_stacked = torch.stack(lses, dim=0)  # [N, H, Sq]
    outs_stacked = torch.stack(outs, dim=0)  # [N, Sq, H, Dv]

    # Global LSE: log(sum_i exp(lse_i)) along N axis
    global_lse = torch.logsumexp(lses_stacked, dim=0)  # [H, Sq]

    # Per-partial weight: exp(lse_i - global_lse)
    weights = torch.exp(lses_stacked - global_lse.unsqueeze(0))  # [N, H, Sq]
    # Permute to [N, Sq, H, 1] for broadcasting
    weights = weights.permute(0, 2, 1).unsqueeze(-1)  # [N, Sq, H, 1]

    merged_out = (outs_stacked * weights).sum(dim=0)  # [Sq, H, Dv]
    return merged_out, global_lse


def _kshard(
    k: torch.Tensor, v: torch.Tensor, rank: int, world_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved sequence shard matching block_table.py:148 layout
    (token i goes to rank i % world_size when interleave_size=1).
    """
    return k[rank::world_size], v[rank::world_size]


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("num_heads", [4, 16])
@pytest.mark.parametrize("sq", [1, 64, 256])
@pytest.mark.parametrize("sk", [128, 1024])
def test_pcp_partial_merge_matches_single_rank(
    world_size: int, num_heads: int, sq: int, sk: int
):
    """Splitting K across `world_size` ranks and LSE-merging the partials
    must equal single-rank attention over the full K.

    Mirrors the math our patched _context_parallel_compute_prefill_context
    performs per chunk: each rank computes FA against its 1/N K-shard,
    then cp_lse_ag_out_ar AllReduces (out, lse) across ranks.
    """
    if sk % world_size != 0:
        pytest.skip("sk must divide evenly across ranks")
    torch.manual_seed(0)
    head_dim = 64
    v_head_dim = 64
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(sq, num_heads, head_dim)
    k = torch.randn(sk, num_heads, head_dim)
    v = torch.randn(sk, num_heads, v_head_dim)

    # Reference: single-rank FA over full K.
    out_ref, lse_ref = _naive_attention(q, k, v, softmax_scale)

    # Sharded: each rank computes against its 1/N of K, then LSE-merge.
    partial_outs = []
    partial_lses = []
    for r in range(world_size):
        k_r, v_r = _kshard(k, v, r, world_size)
        out_r, lse_r = _naive_attention(q, k_r, v_r, softmax_scale)
        partial_outs.append(out_r)
        partial_lses.append(lse_r)
    out_merged, lse_merged = _lse_weighted_merge(partial_outs, partial_lses)

    # Numerical tolerance: pure FP32 math; small accumulation error.
    assert torch.allclose(out_merged, out_ref, atol=1e-5, rtol=1e-5), (
        f"out_merged != out_ref: max diff = "
        f"{(out_merged - out_ref).abs().max().item()}"
    )
    assert torch.allclose(lse_merged, lse_ref, atol=1e-5, rtol=1e-5), (
        f"lse_merged != lse_ref: max diff = "
        f"{(lse_merged - lse_ref).abs().max().item()}"
    )


def test_pcp_partial_merge_handles_uneven_shards():
    """When sk is not divisible by world_size, the last rank may have
    fewer K tokens. The merge math must still hold (it doesn't depend on
    equal shard sizes — only on disjoint K coverage).
    """
    torch.manual_seed(0)
    world_size, num_heads, sq, sk = 4, 8, 32, 130  # 130/4 = uneven
    head_dim = v_head_dim = 32
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(sq, num_heads, head_dim)
    k = torch.randn(sk, num_heads, head_dim)
    v = torch.randn(sk, num_heads, v_head_dim)

    out_ref, lse_ref = _naive_attention(q, k, v, softmax_scale)

    partial_outs, partial_lses = [], []
    for r in range(world_size):
        # Interleaved sharding: rank r gets positions r, r+W, r+2W, ...
        k_r = k[r::world_size]
        v_r = v[r::world_size]
        out_r, lse_r = _naive_attention(q, k_r, v_r, softmax_scale)
        partial_outs.append(out_r)
        partial_lses.append(lse_r)
    out_merged, _ = _lse_weighted_merge(partial_outs, partial_lses)

    assert torch.allclose(out_merged, out_ref, atol=1e-5, rtol=1e-5)


def test_pcp_chunked_iter_merge_equivalence():
    """Simulate the full chunked-context loop: K is split first into
    chunks (outer loop in _context_parallel_compute_prefill_context),
    each chunk is further split across PCP ranks (inner LSE merge), and
    chunks are accumulated via merge_attn_states (online softmax).

    Verify the end-to-end output matches a single-rank FA over the full K.
    """
    torch.manual_seed(0)
    world_size, num_heads, sq, sk = 4, 8, 16, 1024
    num_chunks = 4
    head_dim = v_head_dim = 64
    softmax_scale = 1.0 / math.sqrt(head_dim)
    chunk_size = sk // num_chunks

    q = torch.randn(sq, num_heads, head_dim)
    k = torch.randn(sk, num_heads, head_dim)
    v = torch.randn(sk, num_heads, v_head_dim)

    out_ref, _ = _naive_attention(q, k, v, softmax_scale)

    # Inner: per-chunk PCP merge. Outer: chunk accumulation via online softmax.
    acc_out: torch.Tensor | None = None
    acc_lse: torch.Tensor | None = None
    for c in range(num_chunks):
        k_chunk = k[c * chunk_size : (c + 1) * chunk_size]
        v_chunk = v[c * chunk_size : (c + 1) * chunk_size]

        # PCP-shard the chunk's K across world_size ranks.
        partial_outs, partial_lses = [], []
        for r in range(world_size):
            k_r = k_chunk[r::world_size]
            v_r = v_chunk[r::world_size]
            out_r, lse_r = _naive_attention(q, k_r, v_r, softmax_scale)
            partial_outs.append(out_r)
            partial_lses.append(lse_r)
        chunk_out, chunk_lse = _lse_weighted_merge(partial_outs, partial_lses)

        # Accumulate across chunks (same LSE-weighted merge math).
        if acc_out is None:
            acc_out, acc_lse = chunk_out, chunk_lse
        else:
            acc_out, acc_lse = _lse_weighted_merge(
                [acc_out, chunk_out], [acc_lse, chunk_lse]
            )

    assert torch.allclose(acc_out, out_ref, atol=1e-5, rtol=1e-5), (
        f"chunked+PCP merge != single-rank FA: max diff = "
        f"{(acc_out - out_ref).abs().max().item()}"
    )


def test_pcp_partial_merge_handles_tp_eq_1_head_invariance():
    """The PCP correctness contract requires TP=1 (heads consistent
    across CP ranks). This test verifies that when heads ARE consistent,
    the merge is correct — and shows the failure mode if heads diverged
    (e.g., the broken TP=4+DCP=4 case we identified).
    """
    torch.manual_seed(0)
    world_size, num_heads, sq, sk = 4, 16, 8, 256
    head_dim = v_head_dim = 32
    softmax_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(sq, num_heads, head_dim)
    k = torch.randn(sk, num_heads, head_dim)
    v = torch.randn(sk, num_heads, v_head_dim)

    # Correct (TP=1): each rank holds the same `num_heads`, sharded K.
    out_ref, _ = _naive_attention(q, k, v, softmax_scale)
    partial_outs, partial_lses = [], []
    for r in range(world_size):
        k_r = k[r::world_size]
        v_r = v[r::world_size]
        out_r, lse_r = _naive_attention(q, k_r, v_r, softmax_scale)
        partial_outs.append(out_r)
        partial_lses.append(lse_r)
    out_merged, _ = _lse_weighted_merge(partial_outs, partial_lses)
    assert torch.allclose(out_merged, out_ref, atol=1e-5, rtol=1e-5)

    # Broken (TP=N, simulated): each rank holds DIFFERENT heads. Summing
    # at tensor index [h] across ranks mixes unrelated head values.
    h_per_rank = num_heads // world_size
    partial_outs_broken, partial_lses_broken = [], []
    for r in range(world_size):
        # Rank r sees only heads r*h_per_rank : (r+1)*h_per_rank, but
        # they live at indices [0:h_per_rank] in the local tensor.
        head_slice = slice(r * h_per_rank, (r + 1) * h_per_rank)
        q_r = q[:, head_slice]
        k_r = k[r::world_size][:, head_slice]
        v_r = v[r::world_size][:, head_slice]
        out_r, lse_r = _naive_attention(q_r, k_r, v_r, softmax_scale)
        # Pad back to num_heads so the AllReduce shape matches.
        # In real TP+DCP this padding is "the rank's TP-shard heads at
        # tensor index [h]" — different actual heads per rank.
        out_r_padded = torch.zeros(sq, num_heads, v_head_dim)
        lse_r_padded = torch.full((num_heads, sq), float("-inf"))
        out_r_padded[:, head_slice] = out_r
        lse_r_padded[head_slice] = lse_r
        partial_outs_broken.append(out_r_padded)
        partial_lses_broken.append(lse_r_padded)
    out_merged_broken, _ = _lse_weighted_merge(partial_outs_broken, partial_lses_broken)

    # With heads disjoint per rank, the merge does NOT equal single-rank FA
    # because each rank only saw 1/world_size of K positions for its heads
    # (not the full K). This is exactly the bug the TP=1 assertion guards.
    assert not torch.allclose(out_merged_broken, out_ref, atol=1e-3), (
        "expected broken-TP merge to diverge from reference, but it did not"
    )
