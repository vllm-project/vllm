# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-tensor tests for the DCP exact-merge global top-k in the sparse MLA
indexer (``sparse_indexer_mode == "exact"``).

Simulate N DCP ranks on a single CPU process: each rank picks a local top-k
over its interleaved KV shard, the candidates are concatenated (standing in for
the cross-rank all_gather), and every rank independently reselects the global
top-k. Asserts (1) the union of per-rank owned selections reconstructs the true
global top-k, and (2) ties are broken deterministically and identically across
ranks (the int64 total-order key), which is what prevents per-rank selection
drift -> garbled tokens. No GPU or process group required.
"""

import pytest
import torch

from vllm.model_executor.layers.sparse_attn_indexer import (
    _dcp_pack_local_candidates,
    _dcp_select_owned_global_topk,
)


def _simulate_local_topk(full_logits: torch.Tensor, world: int, topk: int):
    """Build each rank's (local_topk_indices, local_logits, local_valid) by
    taking its interleaved shard (CP_INTERLEAVE=1: rank r owns global positions
    {r, r+world, ...}) and its local top-k, mirroring top_k_per_row over the
    shard (padding short rows to ``topk`` with -1)."""
    T, L = full_logits.shape
    per_rank = []
    for r in range(world):
        owned_pos = torch.arange(r, L, world)
        local_len = int(owned_pos.numel())
        local_logits = full_logits[:, owned_pos]  # [T, local_len]
        k = min(topk, local_len)
        idx = torch.topk(local_logits, k, dim=1).indices
        local_topk = torch.full((T, topk), -1, dtype=torch.int32)
        local_topk[:, :k] = idx.to(torch.int32)
        local_valid = torch.full((T,), local_len, dtype=torch.int32)
        per_rank.append((local_topk, local_logits, local_valid))
    return per_rank


def _run_merge(full_logits: torch.Tensor, world: int, topk: int):
    """Run the full exact-merge across simulated ranks; return, per row, the set
    of selected GLOBAL positions (union of every rank's owned selection)."""
    per_rank = _simulate_local_topk(full_logits, world, topk)
    cands = [
        _dcp_pack_local_candidates(lt, ll, lv, r, world)
        for r, (lt, ll, lv) in enumerate(per_rank)
    ]
    cand_all = torch.cat(cands, dim=0)  # stands in for all_gather(dim=0)
    T = full_logits.shape[0]
    selected = [set() for _ in range(T)]
    for r in range(world):
        owned_local = _dcp_select_owned_global_topk(cand_all, world, topk, r)
        for t in range(T):
            for li in owned_local[t].tolist():
                if li >= 0:
                    selected[t].add(r + li * world)
    return selected


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("topk", [4, 16])
def test_exact_merge_reconstructs_global_topk(world, topk):
    """Top-k-of-top-k: every global top-k position lies in its owning rank's
    local top-k, so the merged selection equals the true global top-k."""
    torch.manual_seed(0)
    T, L = 5, 200
    # Distinct values (no ties) so the reference top-k set is unique.
    full_logits = torch.randn(T, L) + torch.arange(L) * 1e-4
    got = _run_merge(full_logits, world, topk)
    ref = torch.topk(full_logits, topk, dim=1).indices
    for t in range(T):
        assert got[t] == set(ref[t].tolist()), f"row {t}"


@pytest.mark.parametrize("world", [2, 4])
def test_exact_merge_tie_determinism(world):
    """With heavily-tied fp8-like scores, every rank must select the IDENTICAL
    global set (size topk), repeatably. Without the int64 total-order key a
    plain topk would break ties per-rank-nondeterministically and the ranks
    would diverge."""
    torch.manual_seed(0)
    T, L, topk = 4, 128, 16
    full_logits = torch.randint(0, 3, (T, L)).float()  # ties everywhere
    got1 = _run_merge(full_logits, world, topk)
    got2 = _run_merge(full_logits, world, topk)
    for t in range(T):
        # Candidates are globally distinct, so a full global top-k is exactly
        # topk distinct positions, partitioned across the owning ranks.
        assert len(got1[t]) == topk
        assert got1[t] == got2[t]


def test_exact_merge_tie_break_prefers_lowest_position():
    """Among equal scores the int64 key biases toward the lowest global
    position (~global_pos in the low bits), giving a well-defined tie-break."""
    world, topk = 2, 4
    T, L = 1, 16
    full_logits = torch.zeros(T, L)  # all tied
    got = _run_merge(full_logits, world, topk)
    # All scores equal -> the topk lowest global positions win.
    assert got[0] == set(range(topk))
