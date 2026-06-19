# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import build_prefill_chunk_metadata
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.ops.common import CPTritonContext, correct_attn_out


def _local_count(length: int, rank: int, world: int, interleave: int) -> int:
    return sum(1 for pos in range(length) if (pos // interleave) % world == rank)


def _global_to_local_indices(
    global_indices: torch.Tensor,
    rank: int,
    world: int,
    interleave: int,
) -> torch.Tensor:
    valid = global_indices >= 0
    global_i64 = global_indices.to(torch.int64).clamp_min(0)
    owner = (global_i64 // interleave) % world
    local = (global_i64 // (world * interleave)) * interleave + global_i64 % interleave
    return torch.where(valid & (owner == rank), local, -1).to(torch.int64)


def _local_to_global_indices(
    local_indices: torch.Tensor,
    rank: int,
    world: int,
    interleave: int,
) -> torch.Tensor:
    valid = local_indices >= 0
    local = local_indices.to(torch.int64).clamp_min(0)
    global_indices = (
        (local // interleave) * (world * interleave)
        + rank * interleave
        + local % interleave
    )
    return torch.where(valid, global_indices, -1).to(torch.int64)


def _attention_from_indices(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = indices >= 0
    safe_indices = indices.clamp_min(0)
    selected_k = k[safe_indices]
    selected_v = v[safe_indices]
    scores = torch.einsum("td,tkd->tk", q, selected_k)
    scores = scores.masked_fill(~valid, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1).masked_fill(~valid, 0.0)
    out = torch.einsum("tk,tkd->td", probs, selected_v)
    empty_rows = ~valid.any(dim=-1)
    out[empty_rows] = 0
    lse[empty_rows] = float("-inf")
    return out, lse


def _dcp_lse_merge(
    local_outs: list[torch.Tensor],
    local_lses: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    outs = torch.stack(local_outs, dim=0)
    lses = torch.stack(local_lses, dim=0)
    merged_lse = torch.logsumexp(lses, dim=0)
    weights = torch.exp(lses - merged_lse.unsqueeze(0))
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    merged_out = (outs * weights.unsqueeze(-1)).sum(dim=0)
    return merged_out, merged_lse


class _FakeDCPGroup:
    """Single-process stand-in: ``all_gather`` returns the pre-built
    concatenation of every rank's packed ``(score, global_id)`` candidates,
    mirroring the one packed all-gather the merge issues."""

    def __init__(self, gathered_packed: torch.Tensor) -> None:
        self.gathered_packed = gathered_packed

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == 1
        return self.gathered_packed.clone()


def _run_decode_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    next_n: int,
    topk: int,
) -> torch.Tensor:
    indices = torch.empty(
        (logits.shape[0], topk), dtype=torch.int32, device=logits.device
    )
    torch.ops._C.top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        topk,
    )
    return indices


def _run_persistent_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int,
    max_seq_len: int,
) -> torch.Tensor:
    indices = torch.empty(
        (logits.shape[0], topk), dtype=torch.int32, device=logits.device
    )
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=logits.device)
    torch.ops._C.persistent_topk(
        logits,
        seq_lens,
        indices,
        workspace,
        topk,
        max_seq_len,
    )
    return indices


def _run_prefill_topk(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    indices = torch.empty(
        (logits.shape[0], topk), dtype=torch.int32, device=logits.device
    )
    torch.ops._C.top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        topk,
    )
    return indices


def _dcp_attention_from_local_topks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    local_topks: list[torch.Tensor],
    world: int,
    interleave: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_outs = []
    local_lses = []
    for rank, local_topk in enumerate(local_topks):
        owned = [
            pos for pos in range(k.shape[0]) if (pos // interleave) % world == rank
        ]
        local_out, local_lse = _attention_from_indices(
            q, k[owned], v[owned], local_topk.to(torch.int64)
        )
        local_outs.append(local_out)
        local_lses.append(local_lse)
    return _dcp_lse_merge(local_outs, local_lses)


def _merge_local_topks_global_with_fake_dcp(
    local_logits: list[torch.Tensor],
    local_topks: list[torch.Tensor],
    topk: int,
    world: int,
    interleave: int,
    row_starts: list[torch.Tensor | None] | None = None,
) -> list[torch.Tensor]:
    """Run ``_merge_dcp_topk_global`` for every rank against a faked
    all-gather and return each rank's global-index result (all ranks should
    agree). The fake pre-builds the packed candidate concatenation the merge's
    single ``all_gather`` would return."""
    packed_per_rank = []
    for rank, (logits, indices) in enumerate(zip(local_logits, local_topks)):
        score_indices = indices.clamp_min(0).to(torch.long)
        if row_starts is not None:
            rs = row_starts[rank]
            assert rs is not None
            score_indices = score_indices + rs.to(torch.long).view(-1, 1)
        scores = logits.gather(1, score_indices).masked_fill(indices < 0, float("-inf"))
        global_ids = sparse_indexer._local_dcp_indices_to_global(
            indices, rank, world, interleave
        )
        packed_per_rank.append(
            torch.stack((scores.float(), global_ids.to(torch.float32)), dim=-1)
        )

    fake_group = _FakeDCPGroup(torch.cat(packed_per_rank, dim=1).contiguous())
    original_get_dcp_group = sparse_indexer.get_dcp_group
    sparse_indexer.get_dcp_group = lambda: fake_group
    try:
        return [
            sparse_indexer._merge_dcp_topk_global(
                logits,
                indices,
                topk,
                rank,
                world,
                interleave,
                row_starts=None if row_starts is None else row_starts[rank],
            )
            for rank, (logits, indices) in enumerate(zip(local_logits, local_topks))
        ]
    finally:
        sparse_indexer.get_dcp_group = original_get_dcp_group


def _merge_local_topks_with_fake_dcp(
    local_logits: list[torch.Tensor],
    local_topks: list[torch.Tensor],
    topk: int,
    world: int,
    interleave: int,
    row_starts: list[torch.Tensor | None] | None = None,
) -> list[torch.Tensor]:
    """Like ``_merge_local_topks_global_with_fake_dcp`` but returns each rank's
    result as local (per-shard) indices for the partial-attention helpers."""
    merged_global = _merge_local_topks_global_with_fake_dcp(
        local_logits, local_topks, topk, world, interleave, row_starts=row_starts
    )
    return [
        _global_to_local_indices(g.to(torch.int64), rank, world, interleave)
        for rank, g in enumerate(merged_global)
    ]


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("interleave", [1, 2, 4])
def test_get_dcp_local_seq_lens_matches_naive(world: int, interleave: int):
    seq_lens = torch.arange(0, 33, dtype=torch.int32)

    for rank in range(world):
        actual = get_dcp_local_seq_lens(seq_lens, world, rank, interleave)
        expected = torch.tensor(
            [
                _local_count(int(seq_len), rank, world, interleave)
                for seq_len in seq_lens
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual, expected)


def test_get_dcp_local_seq_lens_can_localize_per_token_bounds():
    seq_lens = torch.tensor([0, 1, 2, 3, 4, 7, 8, 17], dtype=torch.int32)
    world = 4
    interleave = 2

    for rank in range(world):
        actual = get_dcp_local_seq_lens(seq_lens, world, rank, interleave)
        expected = torch.tensor(
            [
                _local_count(int(seq_len), rank, world, interleave)
                for seq_len in seq_lens
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual, expected)


def test_get_dcp_local_seq_lens_preserves_mtp_bounds_shape():
    seq_lens = torch.tensor([[8, 9, 10], [11, 12, 13]], dtype=torch.int32)
    world = 2
    rank = 1
    interleave = 1

    actual = get_dcp_local_seq_lens(seq_lens, world, rank, interleave)
    expected = torch.tensor(
        [
            [_local_count(int(seq_len), rank, world, interleave) for seq_len in row]
            for row in seq_lens
        ],
        dtype=torch.int32,
    )

    assert actual.shape == seq_lens.shape
    torch.testing.assert_close(actual, expected)


def test_get_dcp_local_seq_lens_must_run_after_decode_expansion():
    world = 2
    rank = 1
    interleave = 1
    expanded_bounds = torch.tensor([8, 9, 10], dtype=torch.int32)

    localized_after_expansion = get_dcp_local_seq_lens(
        expanded_bounds, world, rank, interleave
    )
    localized_request_len_minus_offsets = get_dcp_local_seq_lens(
        torch.tensor([10], dtype=torch.int32), world, rank
    ) - torch.tensor([2, 1, 0], dtype=torch.int32)

    assert not torch.equal(
        localized_after_expansion, localized_request_len_minus_offsets
    )
    torch.testing.assert_close(
        localized_after_expansion, torch.tensor([4, 4, 5], dtype=torch.int32)
    )


@pytest.mark.parametrize("interleave", [1, 2])
def test_sparse_dcp_attention_matches_global_topk_attention(interleave: int):
    torch.manual_seed(0)
    world = 2
    topk = 3
    num_queries = 4
    max_seq_len = 13
    head_dim = 8

    q = torch.randn(num_queries, head_dim)
    k = torch.randn(max_seq_len, head_dim)
    v = torch.randn(max_seq_len, head_dim)
    seq_lens = torch.tensor([6, 8, 11, 13], dtype=torch.int64)

    scores = q @ k.T
    global_topk = torch.full((num_queries, topk), -1, dtype=torch.int64)
    for row, seq_len in enumerate(seq_lens.tolist()):
        global_topk[row, :topk] = scores[row, :seq_len].topk(topk).indices

    ref_out, ref_lse = _attention_from_indices(q, k, v, global_topk)

    local_outs = []
    local_lses = []
    local_topks = []
    for rank in range(world):
        owned = [
            pos for pos in range(max_seq_len) if (pos // interleave) % world == rank
        ]
        k_local = k[owned]
        v_local = v[owned]
        local_topk = _global_to_local_indices(global_topk, rank, world, interleave)
        local_topks.append(local_topk)
        local_out, local_lse = _attention_from_indices(q, k_local, v_local, local_topk)
        local_outs.append(local_out)
        local_lses.append(local_lse)

    dcp_out, dcp_lse = _dcp_lse_merge(local_outs, local_lses)

    torch.testing.assert_close(dcp_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dcp_lse, ref_lse, atol=1e-5, rtol=1e-5)

    gathered_global = torch.cat(
        [
            _local_to_global_indices(local_topks[rank], rank, world, interleave)
            for rank in range(world)
        ],
        dim=1,
    )
    assert set(gathered_global[gathered_global >= 0].tolist()) == set(
        global_topk.flatten().tolist()
    )


def test_local_topk_union_is_not_equivalent_to_global_topk_attention():
    world = 2
    interleave = 1
    topk = 2
    q = torch.tensor([[1.0]])
    k = torch.tensor(
        [
            [1.00],
            [0.90],
            [0.95],
            [0.85],
        ]
    )
    v = torch.tensor([[0.0], [1000.0], [0.0], [1000.0]])

    scores = q @ k.T
    global_topk = scores.topk(topk, dim=-1).indices
    ref_out, ref_lse = _attention_from_indices(q, k, v, global_topk)

    local_topks = []
    for rank in range(world):
        owned = [
            pos for pos in range(k.shape[0]) if (pos // interleave) % world == rank
        ]
        local_topks.append(scores[:, owned].topk(topk, dim=-1).indices)

    local_union_out, local_union_lse = _dcp_attention_from_local_topks(
        q, k, v, local_topks, world, interleave
    )

    assert not torch.allclose(local_union_out, ref_out)
    assert not torch.allclose(local_union_lse, ref_lse)


def _stable_topk_local(
    logits: torch.Tensor, seq_lens: torch.Tensor, k: int
) -> torch.Tensor:
    """Reference stable top-K (score desc, then lowest id) over each score row."""
    width = logits.shape[1]
    ids = torch.arange(width, device=logits.device, dtype=torch.float64)
    mask = ids.unsqueeze(0) >= seq_lens.unsqueeze(1).to(torch.float64)
    composite = logits.double() * 1.0e9 - ids.unsqueeze(0)
    composite = composite.masked_fill(mask, float("-inf"))
    return composite.topk(k, dim=-1).indices.to(torch.int32)


def test_dcp_merge_topk_matches_global_stable_under_ties():
    """The two-stage candidate merge must equal a single-pass global stable
    top-K, bit-for-bit and identically on every rank, even with many tied
    scores -- the case where plain ``torch.topk`` could diverge across ranks
    and silently corrupt the LSE merge."""
    torch.manual_seed(11)
    world = 4
    interleave = 1
    topk = 6
    num_rows = 3
    max_seq_len = 40
    # Coarse integer scores => many exact ties to stress tie-breaking.
    scores = torch.randint(0, 4, (num_rows, max_seq_len)).float()
    seq_lens = torch.full((num_rows,), max_seq_len, dtype=torch.int32)

    ref_global = _stable_topk_local(scores, seq_lens, topk)

    local_logits = []
    local_topks = []
    for rank in range(world):
        owned = [p for p in range(max_seq_len) if (p // interleave) % world == rank]
        rank_logits = scores[:, owned].contiguous()
        rank_seq = torch.tensor([len(owned)] * num_rows, dtype=torch.int32)
        local_logits.append(rank_logits)
        local_topks.append(_stable_topk_local(rank_logits, rank_seq, topk))

    merged = _merge_local_topks_global_with_fake_dcp(
        local_logits, local_topks, topk, world, interleave
    )
    # Determinism: every rank selects the identical global top-K.
    for rank_topk in merged[1:]:
        torch.testing.assert_close(rank_topk, merged[0])
    # Equivalence: candidate merge == single-pass global stable top-K.
    torch.testing.assert_close(merged[0], ref_global)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("interleave", [1, 2])
def test_sparse_decode_dcp_attention_matches_non_dcp(interleave: int):
    torch.manual_seed(0)
    device = torch.device("cuda")
    world = 2
    topk = 4
    next_n = 3
    batch_size = 2
    num_rows = batch_size * next_n
    max_seq_len = 17
    head_dim = 16

    q = torch.randn(num_rows, head_dim, device=device)
    k = torch.randn(max_seq_len, head_dim, device=device)
    v = torch.randn(max_seq_len, head_dim, device=device)
    logits = q @ k.T
    seq_lens = torch.tensor(
        [[9, 10, 11], [15, 16, 17]], dtype=torch.int32, device=device
    )

    non_dcp_topk = _run_decode_topk(logits, seq_lens, next_n, topk)
    ref_out, ref_lse = _attention_from_indices(q, k, v, non_dcp_topk.to(torch.int64))

    local_logits = []
    local_topks = []
    for rank in range(world):
        owned = [
            pos for pos in range(max_seq_len) if (pos // interleave) % world == rank
        ]
        rank_logits = logits[:, owned].contiguous()
        rank_seq_lens = get_dcp_local_seq_lens(
            seq_lens, world, rank, interleave
        ).contiguous()
        local_logits.append(rank_logits)
        local_topks.append(_run_decode_topk(rank_logits, rank_seq_lens, next_n, topk))

    merged_topks = _merge_local_topks_with_fake_dcp(
        local_logits, local_topks, topk, world, interleave
    )
    dcp_out, dcp_lse = _dcp_attention_from_local_topks(
        q, k, v, merged_topks, world, interleave
    )

    torch.testing.assert_close(dcp_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dcp_lse, ref_lse, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_sparse_decode_dcp_persistent_topk_matches_non_dcp():
    torch.manual_seed(3)
    device = torch.device("cuda")
    world = 2
    interleave = 1
    topk = 512
    num_rows = 2
    max_seq_len = 1025
    head_dim = 16

    q = torch.randn(num_rows, head_dim, device=device)
    k = torch.randn(max_seq_len, head_dim, device=device)
    v = torch.randn(max_seq_len, head_dim, device=device)
    logits = q @ k.T
    seq_lens = torch.tensor([[1024], [1025]], dtype=torch.int32, device=device)

    non_dcp_topk = torch.empty((num_rows, topk), dtype=torch.int64, device=device)
    for row, seq_len in enumerate(seq_lens.flatten().tolist()):
        non_dcp_topk[row] = logits[row, :seq_len].topk(topk).indices
    ref_out, ref_lse = _attention_from_indices(q, k, v, non_dcp_topk)

    local_logits = []
    local_topks = []
    for rank in range(world):
        owned = [
            pos for pos in range(max_seq_len) if (pos // interleave) % world == rank
        ]
        rank_logits = logits[:, owned].contiguous()
        rank_seq_lens = get_dcp_local_seq_lens(
            seq_lens, world, rank, interleave
        ).contiguous()
        local_logits.append(rank_logits)
        local_topks.append(
            _run_persistent_topk(
                rank_logits,
                rank_seq_lens,
                topk,
                max_seq_len=rank_logits.shape[1],
            )
        )

    merged_global_topks = _merge_local_topks_global_with_fake_dcp(
        local_logits, local_topks, topk, world, interleave
    )
    # The radix top-K kernel selects a deterministic SET but writes it in
    # nondeterministic (atomicAdd) order; the production path is permutation-
    # invariant (compaction + softmax), so all ranks must agree on the set, not
    # the array order. (The fp64 fallback happens to return sorted order.)
    ref_topk = merged_global_topks[0]
    for rank_topk in merged_global_topks[1:]:
        for row in range(rank_topk.shape[0]):
            assert set(rank_topk[row].tolist()) == set(ref_topk[row].tolist())

    local_outs = []
    local_lses = []
    for rank, global_topk in enumerate(merged_global_topks):
        owned = [
            pos for pos in range(max_seq_len) if (pos // interleave) % world == rank
        ]
        local_topk = _global_to_local_indices(
            global_topk.to(torch.int64),
            rank,
            world,
            interleave,
        )
        local_out, local_lse = _attention_from_indices(
            q, k[owned], v[owned], local_topk
        )
        local_outs.append(local_out)
        local_lses.append(local_lse)

    dcp_out, dcp_lse = _dcp_lse_merge(local_outs, local_lses)
    torch.testing.assert_close(dcp_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dcp_lse, ref_lse, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("interleave", [1, 2])
def test_sparse_prefill_dcp_attention_matches_non_dcp(interleave: int):
    torch.manual_seed(1)
    device = torch.device("cuda")
    world = 2
    topk = 4
    num_rows = 5
    max_seq_len = 19
    head_dim = 16

    q = torch.randn(num_rows, head_dim, device=device)
    k = torch.randn(max_seq_len, head_dim, device=device)
    v = torch.randn(max_seq_len, head_dim, device=device)
    logits = q @ k.T
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device=device)
    row_ends = torch.tensor([7, 9, 11, 16, 19], dtype=torch.int32, device=device)

    non_dcp_topk = _run_prefill_topk(logits, row_starts, row_ends, topk)
    ref_out, ref_lse = _attention_from_indices(q, k, v, non_dcp_topk.to(torch.int64))

    local_logits = []
    local_topks = []
    local_row_starts = []
    for rank in range(world):
        owned = [
            pos for pos in range(max_seq_len) if (pos // interleave) % world == rank
        ]
        rank_logits = logits[:, owned].contiguous()
        rank_starts = torch.zeros_like(row_starts)
        rank_ends = get_dcp_local_seq_lens(
            row_ends, world, rank, interleave
        ).contiguous()
        local_logits.append(rank_logits)
        local_row_starts.append(rank_starts)
        local_topks.append(_run_prefill_topk(rank_logits, rank_starts, rank_ends, topk))

    merged_topks = _merge_local_topks_with_fake_dcp(
        local_logits,
        local_topks,
        topk,
        world,
        interleave,
        row_starts=local_row_starts,
    )
    dcp_out, dcp_lse = _dcp_attention_from_local_topks(
        q, k, v, merged_topks, world, interleave
    )

    torch.testing.assert_close(dcp_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dcp_lse, ref_lse, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_sparse_prefill_dcp_metadata_uses_global_causal_bounds():
    device = torch.device("cuda")
    seq_len = 8
    dcp_world_size = 4

    query_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    query_start_loc_cpu = torch.tensor([0, seq_len], dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int32)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)

    chunk = build_prefill_chunk_metadata(
        start_idx=0,
        end_idx=1,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        uncompressed_seq_lens=seq_lens,
        compressed_seq_lens=seq_lens,
        compressed_seq_lens_cpu=seq_lens_cpu,
        block_table=block_table,
        compress_ratio=1,
        dcp_rank=0,
        dcp_world_size=dcp_world_size,
        cp_kv_cache_interleave_size=1,
    )
    assert chunk is not None
    torch.accelerator.synchronize()

    torch.testing.assert_close(
        chunk.local_cu_seq_lens.cpu(),
        torch.tensor([0, 2], dtype=torch.int32),
    )
    torch.testing.assert_close(
        chunk.cu_seqlen_ks.cpu(),
        torch.zeros(seq_len, dtype=torch.int32),
    )
    torch.testing.assert_close(
        chunk.cu_seqlen_ke.cpu(),
        torch.arange(1, seq_len + 1, dtype=torch.int32),
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_dcp_filter_compacts_valid_slots_for_sparse_kernel():
    block_size = 4
    num_topk = 128
    dcp_size = 2
    req_id = torch.zeros(1, dtype=torch.int32, device="cuda")
    token_indices = torch.full((1, num_topk), -1, dtype=torch.int32, device="cuda")
    token_indices[0, :8] = torch.arange(8, dtype=torch.int32, device="cuda")
    block_table = torch.tensor([[10]], dtype=torch.int32, device="cuda")

    out, valid_counts = triton_filter_and_convert_dcp_index(
        req_id,
        block_table,
        token_indices,
        dcp_size=dcp_size,
        dcp_rank=0,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk,
        return_valid_counts=True,
    )

    valid = int(valid_counts.item())
    assert valid == 4
    assert (out[0, :valid] >= 0).all()
    assert (out[0, valid:] == -1).all()
    torch.testing.assert_close(
        out[0, :valid].cpu(),
        torch.tensor([40, 41, 42, 43], dtype=torch.int32),
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_dcp_global_topk_physical_attention_matches_non_dcp():
    torch.manual_seed(2)
    device = torch.device("cuda")
    dcp_size = 2
    block_size = 4
    num_topk = 128
    selected_k = 8
    seq_len = 16
    head_dim = 16
    num_queries = 3

    q = torch.randn(num_queries, head_dim, device=device)
    k_global = torch.randn(seq_len, head_dim, device=device)
    v_global = torch.randn(seq_len, head_dim, device=device)
    scores = q @ k_global.T
    global_topk = torch.full(
        (num_queries, num_topk), -1, dtype=torch.int32, device=device
    )
    global_topk[:, :selected_k] = scores.topk(selected_k, dim=-1).indices.to(
        torch.int32
    )

    ref_out, ref_lse = _attention_from_indices(
        q, k_global, v_global, global_topk[:, :selected_k].to(torch.int64)
    )

    local_outs = []
    local_lses = []
    for rank in range(dcp_size):
        block_ids = torch.tensor(
            [[rank * 10 + 1, rank * 10 + 2]], dtype=torch.int32, device=device
        )
        num_slots = int((block_ids.max().item() + 1) * block_size)
        k_cache = torch.zeros(num_slots, head_dim, device=device)
        v_cache = torch.zeros(num_slots, head_dim, device=device)
        for global_idx in range(seq_len):
            if global_idx % dcp_size != rank:
                continue
            local_idx = global_idx // dcp_size
            block = local_idx // block_size
            offset = local_idx % block_size
            slot = int(block_ids[0, block].item()) * block_size + offset
            k_cache[slot] = k_global[global_idx]
            v_cache[slot] = v_global[global_idx]

        slots, valid_counts = triton_filter_and_convert_dcp_index(
            torch.zeros(num_queries, dtype=torch.int32, device=device),
            block_ids,
            global_topk,
            dcp_size=dcp_size,
            dcp_rank=rank,
            BLOCK_SIZE=block_size,
            NUM_TOPK_TOKENS=num_topk,
            return_valid_counts=True,
        )
        row_ids = torch.arange(num_queries, device=device)
        assert (slots[row_ids, valid_counts] == -1).all()
        local_out, local_lse = _attention_from_indices(
            q, k_cache, v_cache, slots.to(torch.int64)
        )
        local_outs.append(local_out)
        local_lses.append(local_lse)

    dcp_out, dcp_lse = _dcp_lse_merge(local_outs, local_lses)
    torch.testing.assert_close(dcp_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dcp_lse, ref_lse, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("is_lse_base_on_e", [True, False])
def test_correct_attn_out_zeroes_empty_nan_partial(is_lse_base_on_e: bool):
    out = torch.full((1, 1, 4), float("nan"), device="cuda")
    lses = torch.tensor(
        [[[0.0]], [[float("-inf")]]],
        dtype=torch.float32,
        device="cuda",
    )

    corrected, final_lse = correct_attn_out(
        out,
        lses,
        cp_rank=1,
        ctx=CPTritonContext(),
        is_lse_base_on_e=is_lse_base_on_e,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(corrected, torch.zeros_like(corrected))
    torch.testing.assert_close(final_lse, torch.zeros_like(final_lse))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_radix_stable_topk_matches_fp64_set():
    """The CUDA radix stable top-K must select the same SET as the fp64
    composite fallback -- including many exact ties and -inf/-1 padding. Both
    implement the same total order (score desc, then lowest global id), so the
    selected sets are identical; only the array order may differ."""
    if not hasattr(torch.ops._C, "stable_top_k_from_candidates"):
        pytest.skip("radix kernel not built")
    torch.manual_seed(7)
    device = torch.device("cuda")
    B, C, k = 3, 4096, 2048
    # Coarse integer scores => many exact ties at the selection boundary.
    scores = torch.randint(0, 16, (B, C), device=device).float()
    ids = torch.arange(C, device=device, dtype=torch.int32).unsqueeze(0).repeat(B, 1)
    invalid = torch.rand(B, C, device=device) < 0.3
    scores = scores.masked_fill(invalid, float("-inf"))
    ids = torch.where(invalid, torch.full_like(ids, -1), ids)

    radix = sparse_indexer._stable_topk_from_candidates_radix(scores.float(), ids, k)
    fp64 = sparse_indexer._stable_topk_from_candidates_fp64(scores, ids, k)
    for b in range(B):
        assert set(radix[b].tolist()) == set(fp64[b].tolist())


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_decode_topk_pads_surplus_with_negative_one():
    """When a row's valid length < topk the top-k kernel must pad the surplus
    slots with -1: the DCP merge (`topk_indices >= 0`) and
    `triton_filter_and_convert_dcp_index` (`tok < 0`) both treat <0 as invalid,
    so a non-(-1) pad would be silently attended. This is the common case under
    DCP, where each rank's local seq_len = global / world is usually << topk.
    Checked on the *set* of valid indices (the kernel may order them by score,
    not position)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    topk = 8
    next_n = 1
    seq_lens_list = [0, 3, 5, 8]
    num_rows = len(seq_lens_list)
    logits = torch.randn(num_rows, 16, device=device)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device).view(
        num_rows, next_n
    )
    idx = _run_decode_topk(logits, seq_lens, next_n, topk)
    for r, sl in enumerate(seq_lens_list):
        valid = idx[r][idx[r] >= 0]
        # seq_len <= topk, so top-k selects exactly the whole valid range.
        assert valid.numel() == min(sl, topk)
        assert set(valid.tolist()) == set(range(sl))
        assert (idx[r] == -1).sum().item() == topk - min(sl, topk)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_persistent_topk_pads_surplus_with_negative_one():
    """Same surplus=-1 invariant for the persistent_topk kernel (k>=512)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    topk = 512  # persistent_topk requires k in {512, 1024, 2048}
    seq_lens_list = [100, 300, 512]
    num_rows = len(seq_lens_list)
    max_seq_len = 600
    logits = torch.randn(num_rows, max_seq_len, device=device)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device).view(
        num_rows, 1
    )
    idx = _run_persistent_topk(logits, seq_lens, topk, max_seq_len)
    for r, sl in enumerate(seq_lens_list):
        valid = idx[r][idx[r] >= 0]
        assert valid.numel() == min(sl, topk)
        assert set(valid.tolist()) == set(range(sl))
        assert (idx[r] == -1).sum().item() == topk - min(sl, topk)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_dcp_interleave_source_index_reconstructs_global_k():
    """Prefill K-gather: each DCP rank holds its round-robin slice of every
    request's K in a [dcp_size, max_local_total] buffer, and the all-gathered
    buffer is rank-major. ``_dcp_interleave_source_index`` must map each global
    (chunk-order) position back to (rank, local_offset) so that
    ``gathered[src_idx]`` reconstructs the original global-ordered K."""
    dcp_size = 3
    req_lens = [7, 4, 1]  # uneven -> exercises round-robin remainders
    num_reqs = len(req_lens)
    global_total = sum(req_lens)
    dim = 5

    global_cu = torch.zeros(num_reqs + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(req_lens, dtype=torch.int32), dim=0, out=global_cu[1:])
    global_k = torch.arange(global_total * dim, dtype=torch.float32).view(
        global_total, dim
    )

    # Round-robin (interleave=1) per-rank, per-request local token counts.
    local_counts = torch.zeros(dcp_size, num_reqs, dtype=torch.int64)
    for r, length in enumerate(req_lens):
        for t in range(length):
            local_counts[t % dcp_size, r] += 1
    max_local_total = int(local_counts.sum(dim=1).max().item())
    local_cu = torch.zeros(dcp_size, num_reqs + 1, dtype=torch.int32)
    torch.cumsum(local_counts, dim=1, out=local_cu[:, 1:])

    # Rank-major buffer that all_gather(local_k, dim=0) would produce.
    gathered = torch.zeros(dcp_size, max_local_total, dim)
    for r, length in enumerate(req_lens):
        for t in range(length):
            rank = t % dcp_size
            slot = int(local_cu[rank, r]) + t // dcp_size
            gathered[rank, slot] = global_k[int(global_cu[r]) + t]

    src_idx = sparse_indexer._dcp_interleave_source_index(
        global_cu, local_cu, max_local_total, dcp_size, global_total
    )
    recon = gathered.view(-1, dim)[src_idx]
    torch.testing.assert_close(recon, global_k)


def test_dcp_interleave_source_index_empty_chunk():
    src_idx = sparse_indexer._dcp_interleave_source_index(
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(2, 1, dtype=torch.int32),
        0,
        2,
        0,
    )
    assert src_idx.numel() == 0


def test_sparse_decode_dcp_short_context_matches_non_dcp():
    """End-to-end DCP decode where the global seq_len < topk (so every rank's
    local top-k is surplus-padded). Exercises the kernel surplus -> merge mask
    -> global top-k -> physical localize -> LSE merge chain for the common
    short-context decode case, vs the non-DCP reference."""
    torch.manual_seed(4)
    device = torch.device("cuda")
    world = 2
    interleave = 1
    topk = 512
    num_rows = 2
    max_seq_len = 300  # < topk -> surplus everywhere
    head_dim = 16

    q = torch.randn(num_rows, head_dim, device=device)
    k = torch.randn(max_seq_len, head_dim, device=device)
    v = torch.randn(max_seq_len, head_dim, device=device)
    logits = q @ k.T
    seq_lens = torch.tensor([[250], [300]], dtype=torch.int32, device=device)

    non_dcp_topk = torch.empty((num_rows, topk), dtype=torch.int64, device=device)
    for row, seq_len in enumerate(seq_lens.flatten().tolist()):
        sel = logits[row, :seq_len].topk(min(topk, seq_len)).indices
        non_dcp_topk[row, : sel.numel()] = sel
        non_dcp_topk[row, sel.numel() :] = -1
    ref_out, ref_lse = _attention_from_indices(q, k, v, non_dcp_topk)

    local_logits = []
    local_topks = []
    for rank in range(world):
        owned = [p for p in range(max_seq_len) if (p // interleave) % world == rank]
        rank_logits = logits[:, owned].contiguous()
        rank_seq_lens = get_dcp_local_seq_lens(seq_lens, world, rank, interleave)
        local_logits.append(rank_logits)
        local_topks.append(
            _run_persistent_topk(
                rank_logits,
                rank_seq_lens.contiguous(),
                topk,
                max_seq_len=rank_logits.shape[1],
            )
        )

    merged_global_topks = _merge_local_topks_global_with_fake_dcp(
        local_logits, local_topks, topk, world, interleave
    )
    # The radix top-K kernel selects a deterministic SET but writes it in
    # nondeterministic (atomicAdd) order; the production path is permutation-
    # invariant (compaction + softmax), so all ranks must agree on the set, not
    # the array order. (The fp64 fallback happens to return sorted order.)
    ref_topk = merged_global_topks[0]
    for rank_topk in merged_global_topks[1:]:
        for row in range(rank_topk.shape[0]):
            assert set(rank_topk[row].tolist()) == set(ref_topk[row].tolist())

    local_outs = []
    local_lses = []
    for rank, global_topk in enumerate(merged_global_topks):
        owned = [p for p in range(max_seq_len) if (p // interleave) % world == rank]
        local_topk = _global_to_local_indices(
            global_topk.to(torch.int64), rank, world, interleave
        )
        local_out, local_lse = _attention_from_indices(
            q, k[owned], v[owned], local_topk
        )
        local_outs.append(local_out)
        local_lses.append(local_lse)

    dcp_out, dcp_lse = _dcp_lse_merge(local_outs, local_lses)
    torch.testing.assert_close(dcp_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dcp_lse, ref_lse, atol=1e-5, rtol=1e-5)
