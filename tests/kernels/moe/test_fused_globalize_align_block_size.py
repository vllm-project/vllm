# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness of the fused globalize + align + sort decode metadata kernel.

The fused op replaces the DeepEP-v2 INDEXED cudagraph-decode chain
(_globalize_recv_topk_idx + moe_align_block_size + count_and_sort). It is
checked against that exact reference chain over a grid of shapes / EP ranks /
fill distributions.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.fused_globalize_align_block_size import (
    fused_globalize_align_block_size,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 import (
    _globalize_recv_topk_idx,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="fused globalize align is CUDA-only"
)


def _expert_map(global_num_experts, rank_expert_offset, local_num_experts, device):
    expert_map = torch.full((global_num_experts,), -1, device=device, dtype=torch.int32)
    rng = torch.arange(local_num_experts, device=device, dtype=torch.int32)
    expert_map[rank_expert_offset : rank_expert_offset + local_num_experts] = rng
    return expert_map


def _make_case(n, topk, ep_size, local_e, block_size, seed, frac_recv, rank, fill):
    torch.manual_seed(seed)
    device = "cuda"
    global_num_experts = local_e * ep_size
    rank_idx = {"first": 0, "mid": ep_size // 2, "last": ep_size - 1}[rank]
    rank_expert_offset = rank_idx * local_e

    if fill == "all_nonlocal":
        local_idx = torch.full((n, topk), -1, device=device, dtype=torch.int64)
    elif fill == "single":
        local_idx = torch.zeros((n, topk), device=device, dtype=torch.int64)
    elif fill == "all_local":
        local_idx = torch.randint(
            0, local_e, (n, topk), device=device, dtype=torch.int64
        )
    else:
        local_idx = torch.randint(
            -1, local_e, (n, topk), device=device, dtype=torch.int64
        )

    num_recv = max(0, min(n, int(round(n * frac_recv))))
    if num_recv < n:
        local_idx[num_recv:] = torch.randint(
            -1, local_e, (n - num_recv, topk), device=device, dtype=torch.int64
        )
    psum = torch.zeros(4, device=device, dtype=torch.int32)
    psum[-1] = num_recv
    return local_idx, psum, rank_expert_offset, global_num_experts


def _group_by_expert(sorted_ids, expert_ids, block_size, valid_len, total):
    groups: dict[int, list[int]] = {}
    for b in range(valid_len // block_size):
        eid = int(expert_ids[b].item())
        blk = sorted_ids[b * block_size : (b + 1) * block_size]
        groups.setdefault(eid, []).extend(blk[blk < total].tolist())
    return groups


@pytest.mark.parametrize("n", [1, 7, 64, 257])
# topk=16 with local_num_experts=28 is the Kimi K3 decode config. topk=6 and
# block_size=16, 128 are outside the old template dispatch table: they guard
# against a regression to fixed (topk, block_size) specialization.
@pytest.mark.parametrize("topk", [1, 6, 8, 16])
@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
@pytest.mark.parametrize("local_e", [28, 32])
@pytest.mark.parametrize("rank", ["first", "mid", "last"])
@pytest.mark.parametrize("fill", ["rand", "all_local", "all_nonlocal", "single"])
def test_fused_matches_reference(n, topk, block_size, local_e, rank, fill):
    ep_size = 4
    local_idx, psum, reo, gne = _make_case(
        n, topk, ep_size, local_e, block_size, seed=n * 131 + topk,
        frac_recv=0.75, rank=rank, fill=fill,
    )

    ref_global = _globalize_recv_topk_idx(local_idx.clone(), psum, reo, gne)
    ref_sorted, ref_expert_ids, ref_num = moe_align_block_size(
        topk_ids=ref_global,
        block_size=block_size,
        num_experts=gne,
        expert_map=_expert_map(gne, reo, local_e, local_idx.device),
        ignore_invalid_experts=True,
    )

    fused_global, fused_sorted, fused_expert_ids, fused_num = (
        fused_globalize_align_block_size(
            local_idx.clone(), psum, reo, gne, local_e, block_size
        )
    )

    total = n * topk
    assert torch.equal(fused_global, ref_global)
    assert torch.equal(fused_num, ref_num)

    valid = int(fused_num.item())
    nb = valid // block_size
    assert torch.equal(fused_expert_ids[:nb], ref_expert_ids[:nb])
    assert bool((fused_expert_ids[nb:] == -1).all())
    assert bool((fused_sorted[valid:] == total).all())

    ref_groups = _group_by_expert(ref_sorted, ref_expert_ids, block_size,
                                  int(ref_num.item()), total)
    fused_groups = _group_by_expert(fused_sorted, fused_expert_ids, block_size,
                                    valid, total)
    assert set(ref_groups) == set(fused_groups)
    for eid in ref_groups:
        assert sorted(ref_groups[eid]) == sorted(fused_groups[eid])
