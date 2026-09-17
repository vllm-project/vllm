# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two DCP ranks must reproduce what one rank computes over the whole cache.

This runs both ranks in one process against separately built caches, so it
tests the part that a collective cannot hide: whether a localized id still
addresses the key its global id named, and whether merging the two partial
softmaxes by their LSE reconstructs the single-rank answer.

The merge here is the arithmetic of correct_attn_out, transcribed. Running the
real collective needs two GPUs and adds nothing to what is being checked.
"""

import pytest
import torch

from vllm.platforms import current_platform

requires_gpu = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="QSA kernels need CUDA"
)

PAGE = 16
HEAD_DIM = 64
NUM_KV_HEADS = 2
NUM_Q_HEADS = 8


def _owner(pos, world, interleave):
    return (pos // interleave) % world


def _local_id(pos, world, interleave):
    return (pos // (world * interleave)) * interleave + (pos % interleave)


def _merge_by_lse(outs, lses):
    """correct_attn_out's arithmetic, base 2 (QSA scores are log2-scaled)."""
    lses = torch.where(torch.isnan(lses) | torch.isposinf(lses), -torch.inf, lses)
    lse_max = lses.max(dim=0).values
    lse_max = torch.where(torch.isneginf(lse_max), torch.zeros_like(lse_max), lse_max)
    global_lse = torch.log2(torch.exp2(lses - lse_max).sum(dim=0)) + lse_max

    total = torch.zeros_like(outs[0])
    for rank in range(outs.shape[0]):
        exponent = lses[rank] - global_lse
        exponent = torch.where(
            torch.isnan(exponent) | torch.isposinf(exponent), -torch.inf, exponent
        )
        factor = torch.exp2(exponent)
        corrected = outs[rank] * factor.unsqueeze(-1)
        corrected = torch.where(
            (factor == 0.0).unsqueeze(-1), torch.zeros_like(corrected), corrected
        )
        total = total + corrected
    return total, global_lse


def _pack(rows, width, device):
    packed = torch.full((len(rows), width + 1), -1, dtype=torch.int32, device=device)
    for i, row in enumerate(rows):
        for j, g in enumerate(row[:width]):
            packed[i, j] = g
        packed[i, width] = len(row[:width])
    return packed


@requires_gpu
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("interleave", [1, 4, 16])
@pytest.mark.parametrize("seq_len", [64, 200])
def test_sharded_ranks_reproduce_the_single_rank_result(world, interleave, seq_len):
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import (
        qsa_dcp_empty_owner_rows,
        qsa_localize_dcp_indices,
        qsa_neutralize_empty_owner_lse_,
    )

    torch.manual_seed(1234 + seq_len)
    device = "cuda"

    full_blocks = (seq_len + PAGE - 1) // PAGE
    shape = (full_blocks, PAGE, NUM_KV_HEADS, HEAD_DIM)
    k_full = torch.randn(shape, dtype=torch.bfloat16, device=device)
    v_full = torch.randn(shape, dtype=torch.bfloat16, device=device)

    # Scatter each global position to the rank that owns it, at its local id.
    local_len = seq_len // world + PAGE
    local_blocks = (local_len + PAGE - 1) // PAGE
    local_shape = (local_blocks, PAGE, NUM_KV_HEADS, HEAD_DIM)
    k_rank = [
        torch.zeros(local_shape, dtype=torch.bfloat16, device=device)
        for _ in range(world)
    ]
    v_rank = [
        torch.zeros(local_shape, dtype=torch.bfloat16, device=device)
        for _ in range(world)
    ]
    for pos in range(seq_len):
        r = _owner(pos, world, interleave)
        lid = _local_id(pos, world, interleave)
        k_rank[r][lid // PAGE, lid % PAGE] = k_full[pos // PAGE, pos % PAGE]
        v_rank[r][lid // PAGE, lid % PAGE] = v_full[pos // PAGE, pos % PAGE]

    # A selection per query row: strided, so ownership is spread unevenly.
    num_rows = 6
    selections = [
        list(range(row, seq_len, row + 2))[: seq_len // 2] for row in range(num_rows)
    ]
    width = max(len(s) for s in selections)
    global_sel = _pack(selections, width, device)

    q = torch.randn(
        (num_rows, NUM_Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    token_to_req = torch.zeros(num_rows, dtype=torch.int32, device=device)
    full_table = torch.arange(full_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )
    local_table = torch.arange(
        local_blocks, dtype=torch.int32, device=device
    ).unsqueeze(0)

    reference, reference_lse = qsa_sparse_paged_attention(
        q, k_full, v_full, global_sel, full_table, token_to_req, False, return_lse=True
    )

    outs, lses = [], []
    for rank in range(world):
        local_sel = qsa_localize_dcp_indices(
            global_sel, torch.empty_like(global_sel), world, rank, interleave
        )
        empty = qsa_dcp_empty_owner_rows(local_sel)
        out, lse = qsa_sparse_paged_attention(
            q,
            k_rank[rank],
            v_rank[rank],
            local_sel,
            local_table,
            token_to_req,
            False,
            return_lse=True,
        )
        qsa_neutralize_empty_owner_lse_(lse, empty)
        outs.append(out.float())
        lses.append(lse.float())

    merged, merged_lse = _merge_by_lse(torch.stack(outs), torch.stack(lses))

    torch.testing.assert_close(
        merged.to(reference.dtype), reference, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(merged_lse, reference_lse.float(), rtol=1e-3, atol=1e-3)


@requires_gpu
def test_a_rank_owning_none_of_the_selection_contributes_nothing():
    """The sparse-only case: a rank holds KV but none of what was selected."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import (
        qsa_dcp_empty_owner_rows,
        qsa_localize_dcp_indices,
        qsa_neutralize_empty_owner_lse_,
    )

    device, world, interleave = "cuda", 2, 1
    torch.manual_seed(7)
    blocks = 4
    shape = (blocks, PAGE, NUM_KV_HEADS, HEAD_DIM)
    k = torch.randn(shape, dtype=torch.bfloat16, device=device)
    v = torch.randn(shape, dtype=torch.bfloat16, device=device)

    # Every selected position is even, so rank 1 owns none of them.
    sel = _pack([[0, 2, 4, 6, 8]], 8, device)
    q = torch.randn((1, NUM_Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device)
    token_to_req = torch.zeros(1, dtype=torch.int32, device=device)
    table = torch.arange(blocks, dtype=torch.int32, device=device).unsqueeze(0)

    local = qsa_localize_dcp_indices(sel, torch.empty_like(sel), world, 1, interleave)
    empty = qsa_dcp_empty_owner_rows(local)
    assert bool(empty[0]), "rank 1 should own none of an all-even selection"

    out, lse = qsa_sparse_paged_attention(
        q, k, v, local, table, token_to_req, False, return_lse=True
    )
    qsa_neutralize_empty_owner_lse_(lse, empty)
    assert torch.isneginf(lse).all()

    # Merging it against a real partial must leave that partial untouched.
    good_out = torch.randn_like(out.float())
    good_lse = torch.zeros_like(lse.float())
    merged, _ = _merge_by_lse(
        torch.stack([out.float(), good_out]), torch.stack([lse.float(), good_lse])
    )
    torch.testing.assert_close(merged, good_out)
