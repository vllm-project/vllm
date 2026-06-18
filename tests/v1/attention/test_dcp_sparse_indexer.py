# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DCP-aware sparse indexer.

Validates the global<->local position remap math and the global top-k
reconstruction: for interleaved DCP sharding, the *union* over ranks of the
per-rank `_dcp_global_topk_remap` output (mapped back to global positions)
must equal the true global top-k of the concatenated full logits.

These tests exercise the pure tensor logic in
`vllm/model_executor/layers/sparse_attn_indexer.py` (no GPU, no real
distributed communicator): `get_dcp_group` is monkeypatched to a fake that
emulates all_gather by concatenating all ranks' pre-computed contributions.
"""

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as idx_mod
from vllm.model_executor.layers.sparse_attn_indexer import (
    _global_to_local_position,
    _local_to_global_position,
)


class _FakeDCPGroup:
    """Single-process stand-in for the DCP group coordinator.

    `set_all` pre-stashes every rank's (global_pos, score) candidate tensors
    (computed by the caller for each rank). `all_gather` then concatenates them
    in rank order, emulating a real all-gather.
    """

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank_in_group = rank
        self._pos: list[torch.Tensor] | None = None
        self._scores: list[torch.Tensor] | None = None

    def set_all(self, pos_list, scores_list):
        self._pos = list(pos_list)
        self._scores = list(scores_list)

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        # `tensor` is this rank's contribution (already in self._* at its rank).
        stash = self._pos if tensor.dtype != torch.float32 else self._scores
        assert stash is not None
        return torch.cat(stash, dim=dim)


def _local_to_global(local_idx, rank, world_size, interleave):
    return _local_to_global_position(local_idx, rank, world_size, interleave)


@pytest.mark.parametrize("interleave", [1, 2, 4])
@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_position_roundtrip(interleave, world_size):
    total = 3 * interleave * world_size
    g = torch.arange(total, dtype=torch.int32)
    owner = (g // interleave) % world_size
    for rank in range(world_size):
        present = g[owner == rank]
        for local_idx in range(present.numel()):
            gg = _local_to_global(
                torch.tensor([local_idx]), rank, world_size, interleave
            )
            # this global position should map back to local_idx on rank `rank`
            assert (
                _global_to_local_position(gg, interleave, world_size).item()
                == local_idx
            )
        # cross-check the owner implied by local->global matches `rank`
        gg_all = _local_to_global(
            torch.arange(present.numel(), dtype=torch.int32),
            rank,
            world_size,
            interleave,
        )
        assert (((gg_all // interleave) % world_size) == rank).all()


@pytest.mark.parametrize("interleave", [1, 2, 3])
@pytest.mark.parametrize("world_size", [2, 4])
def test_global_topk_reconstructs_reference(interleave, world_size, monkeypatch):
    torch.manual_seed(0)
    topk_tokens = 8
    num_rows = 4
    total_len = 6 * interleave * world_size
    full_logits = torch.randn(num_rows, total_len)

    # Shard full_logits to each rank per the interleave scheme. The shard's
    # column order is chosen so local index l maps to global via the formula.
    g = torch.arange(total_len)
    owner = (g // interleave) % world_size
    per_rank_logits = [
        full_logits[:, owner == rank].contiguous() for rank in range(world_size)
    ]

    # Pre-pass: compute each rank's (global_pos, score) candidates by running the
    # same local-topk + local->global logic the helper uses internally. When a
    # shard is shorter than topk_tokens, the real kernel returns only as many
    # valid indices as the shard holds and pads the rest with -1; emulate that.
    pos_list, scores_list = [], []
    for rank in range(world_size):
        lg = per_rank_logits[rank]
        k = min(topk_tokens, lg.shape[1])
        local_topk = lg.topk(k, dim=1).indices.to(torch.int32)
        pad = torch.full((local_topk.shape[0], topk_tokens - k), -1, dtype=torch.int32)
        local_topk = torch.cat([local_topk, pad], dim=1)
        invalid = local_topk < 0
        idx_safe = torch.clamp(local_topk, min=0)
        scores = lg.gather(1, idx_safe.long()).masked_fill(invalid, float("-inf"))
        gp = _local_to_global(idx_safe, rank, world_size, interleave)
        gp = torch.where(invalid, gp.new_full((), -1), gp)
        pos_list.append(gp.contiguous())
        scores_list.append(scores.contiguous())

    # Run the real helper once per rank with a fake group that returns the full
    # concatenation, capturing each rank's final local-index buffer.
    per_rank_final = []
    for rank in range(world_size):
        fake = _FakeDCPGroup(world_size, rank)
        fake.set_all(pos_list, scores_list)
        monkeypatch.setattr(idx_mod, "get_dcp_group", lambda f=fake: f)

        topk_buf = torch.full((num_rows, topk_tokens), -1, dtype=torch.int32)
        # Seed the buffer with rank's local top-k (what the kernel would write).
        lg = per_rank_logits[rank]
        k = min(topk_tokens, lg.shape[1])
        seed = lg.topk(k, dim=1).indices.to(torch.int32)
        topk_buf[:, :k] = seed
        idx_mod._dcp_global_topk_remap(topk_buf, lg, topk_tokens, interleave)
        per_rank_final.append(topk_buf)

    # Reference: global top-k of the FULL logits, as sets per row.
    ref = full_logits.topk(topk_tokens, dim=1).indices
    for row in range(num_rows):
        union = set()
        for rank in range(world_size):
            for local_idx in per_rank_final[rank][row].tolist():
                if local_idx < 0:
                    continue
                union.add(
                    _local_to_global(
                        torch.tensor([local_idx]), rank, world_size, interleave
                    ).item()
                )
        assert union == set(ref[row].tolist()), (
            row,
            sorted(union),
            sorted(ref[row].tolist()),
        )


def test_global_topk_ownership_partition(monkeypatch):
    """Every selected global position is owned by exactly one rank's output."""
    interleave, world_size, topk_tokens = 2, 2, 6
    num_rows = 2
    total_len = 4 * interleave * world_size
    full_logits = torch.randn(num_rows, total_len)
    g = torch.arange(total_len)
    owner = (g // interleave) % world_size
    per_rank_logits = [
        full_logits[:, owner == r].contiguous() for r in range(world_size)
    ]
    pos_list, scores_list = [], []
    for rank in range(world_size):
        lg = per_rank_logits[rank]
        k = min(topk_tokens, lg.shape[1])
        local_topk = lg.topk(k, dim=1).indices.to(torch.int32)
        pad = torch.full((local_topk.shape[0], topk_tokens - k), -1, dtype=torch.int32)
        local_topk = torch.cat([local_topk, pad], dim=1)
        idx_safe = torch.clamp(local_topk, min=0)
        scores = lg.gather(1, idx_safe.long()).masked_fill(
            local_topk < 0, float("-inf")
        )
        gp = _local_to_global(idx_safe, rank, world_size, interleave)
        gp = torch.where(local_topk < 0, gp.new_full((), -1), gp)
        pos_list.append(gp.contiguous())
        scores_list.append(scores.contiguous())
    for rank in range(world_size):
        fake = _FakeDCPGroup(world_size, rank)
        fake.set_all(pos_list, scores_list)
        monkeypatch.setattr(idx_mod, "get_dcp_group", lambda f=fake: f)
        lg = per_rank_logits[rank]
        k = min(topk_tokens, lg.shape[1])
        buf = torch.full((num_rows, topk_tokens), -1, dtype=torch.int32)
        buf[:, :k] = lg.topk(k, dim=1).indices.to(torch.int32)
        idx_mod._dcp_global_topk_remap(buf, lg, topk_tokens, interleave)
        # owned positions get a real (>=0) local idx; others are -1
        for local_idx in buf.flatten().tolist():
            if local_idx >= 0:
                gg = _local_to_global(
                    torch.tensor([local_idx]), rank, world_size, interleave
                ).item()
                assert (gg // interleave) % world_size == rank


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
