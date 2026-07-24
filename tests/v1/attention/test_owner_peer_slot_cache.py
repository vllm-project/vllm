# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.mla import owner_peer_slot_cache as cache_module
from vllm.v1.attention.backends.mla.owner_peer_slot_cache import (
    OwnerPeerSlotCache,
)


def test_owner_peer_slot_cache_refreshes_once_and_reuses(monkeypatch):
    calls = 0

    def fake_convert(
        req_id,
        block_table,
        token_indices,
        *,
        return_valid_counts,
        out,
        valid_counts_out,
        **_,
    ):
        nonlocal calls
        calls += 1
        assert return_valid_counts
        assert out.shape == token_indices.shape
        assert valid_counts_out.shape == (token_indices.shape[0],)
        out.copy_(token_indices + 100)
        valid_counts_out.fill_(token_indices.shape[1])
        return out, valid_counts_out

    monkeypatch.setattr(
        cache_module,
        "convert_global_indices_to_dcp_peer_slots",
        fake_convert,
    )
    cache = OwnerPeerSlotCache(
        torch.empty((8, 4), dtype=torch.int32),
        torch.empty(8, dtype=torch.int32),
    )
    req_id = torch.tensor([0, 1], dtype=torch.int32)
    block_table = torch.tensor([[3, 7], [5, 1]], dtype=torch.int32)
    logical = torch.arange(8, dtype=torch.int32).view(2, 4)

    cache.refresh(
        req_id,
        block_table,
        logical,
        dcp_size=2,
        blocks_per_peer=16,
        cp_kv_cache_interleave_size=4,
        block_size=4,
    )
    first_slots, first_counts = cache.get(
        2,
        block_table,
        dcp_size=2,
        blocks_per_peer=16,
        cp_kv_cache_interleave_size=4,
        block_size=4,
    )
    second_slots, second_counts = cache.get(
        2,
        block_table,
        dcp_size=2,
        blocks_per_peer=16,
        cp_kv_cache_interleave_size=4,
        block_size=4,
    )

    assert calls == 1
    assert cache.generation == 1
    assert first_slots.data_ptr() == second_slots.data_ptr()
    assert first_counts.data_ptr() == second_counts.data_ptr()
    torch.testing.assert_close(first_slots, logical + 100)
    torch.testing.assert_close(first_counts, torch.full((2,), 4, dtype=torch.int32))


def test_owner_peer_slot_cache_invalidates_and_rejects_layout_changes(monkeypatch):
    def fake_convert(
        req_id,
        block_table,
        token_indices,
        *,
        out,
        valid_counts_out,
        **_,
    ):
        out.copy_(token_indices)
        valid_counts_out.fill_(token_indices.shape[1])
        return out, valid_counts_out

    monkeypatch.setattr(
        cache_module,
        "convert_global_indices_to_dcp_peer_slots",
        fake_convert,
    )
    cache = OwnerPeerSlotCache(
        torch.empty((4, 2), dtype=torch.int32),
        torch.empty(4, dtype=torch.int32),
    )
    req_id = torch.tensor([0], dtype=torch.int32)
    block_table = torch.tensor([[3]], dtype=torch.int32)
    logical = torch.tensor([[0, 1]], dtype=torch.int32)
    kwargs = {
        "dcp_size": 2,
        "blocks_per_peer": 16,
        "cp_kv_cache_interleave_size": 4,
        "block_size": 4,
    }
    cache.refresh(req_id, block_table, logical, **kwargs)

    with pytest.raises(RuntimeError, match="layout does not match"):
        cache.get(1, block_table.clone(), **kwargs)

    cache.invalidate()
    with pytest.raises(RuntimeError, match="before an Indexer refresh"):
        cache.get(1, block_table, **kwargs)


def test_owner_local_slots_route_once_per_generation_and_rebuild(monkeypatch):
    def fake_convert(
        req_id,
        block_table,
        token_indices,
        *,
        out,
        valid_counts_out,
        **_,
    ):
        out.copy_(token_indices + 10)
        valid_counts_out.fill_(token_indices.shape[1])
        return out, valid_counts_out

    monkeypatch.setattr(
        cache_module,
        "convert_global_indices_to_dcp_peer_slots",
        fake_convert,
    )
    cache = OwnerPeerSlotCache(
        torch.empty((8, 3), dtype=torch.int32),
        torch.empty(8, dtype=torch.int32),
    )
    req_id = torch.tensor([0, 1], dtype=torch.int32)
    block_table = torch.tensor([[3, 7], [5, 1]], dtype=torch.int32)
    logical = torch.arange(6, dtype=torch.int32).view(2, 3)
    refresh_kwargs = {
        "dcp_size": 2,
        "blocks_per_peer": 16,
        "cp_kv_cache_interleave_size": 4,
        "block_size": 4,
    }
    route_kwargs = {
        "source_stride": 4,
        "owner_rank": 1,
        "dcp_world_size": 2,
        "blocks_per_peer": 16,
        "cp_kv_cache_interleave_size": 4,
        "block_size": 4,
    }
    route_calls: list[torch.Tensor] = []

    def route_and_filter(
        padded_peer_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        route_calls.append(padded_peer_slots.clone())
        routed = torch.cat([padded_peer_slots, padded_peer_slots + 100], dim=0)
        counts = torch.full(
            (routed.shape[0],),
            routed.shape[1],
            dtype=torch.int32,
        )
        return routed, counts

    cache.refresh(req_id, block_table, logical, **refresh_kwargs)
    first = cache.get_or_build_owner_local(
        2,
        block_table,
        build=route_and_filter,
        **route_kwargs,
    )
    second = cache.get_or_build_owner_local(
        2,
        block_table,
        build=route_and_filter,
        **route_kwargs,
    )

    assert len(route_calls) == 1
    assert first[0].data_ptr() == second[0].data_ptr()
    assert first[1].data_ptr() == second[1].data_ptr()
    expected_padded = torch.full((4, 3), -1, dtype=torch.int32)
    expected_padded[:2] = logical + 10
    torch.testing.assert_close(route_calls[0], expected_padded)

    metadata_calls: list[object] = []

    def build_metadata() -> object:
        value = object()
        metadata_calls.append(value)
        return value

    first_metadata = cache.get_or_build_owner_local_metadata(
        ("kernel", 8, 3),
        build_metadata,
    )
    second_metadata = cache.get_or_build_owner_local_metadata(
        ("kernel", 8, 3),
        build_metadata,
    )
    assert first_metadata is second_metadata
    assert len(metadata_calls) == 1
    different_shape_metadata = cache.get_or_build_owner_local_metadata(
        ("kernel", 10, 3),
        build_metadata,
    )
    assert different_shape_metadata is not first_metadata
    assert len(metadata_calls) == 2

    # A new Indexer refresh is a new epoch even when tensor layouts are stable.
    cache.refresh(req_id, block_table, logical + 1, **refresh_kwargs)
    cache.get_or_build_owner_local(
        2,
        block_table,
        build=route_and_filter,
        **route_kwargs,
    )
    assert len(route_calls) == 2
    refreshed_metadata = cache.get_or_build_owner_local_metadata(
        ("kernel", 8, 3),
        build_metadata,
    )
    assert refreshed_metadata is not first_metadata
    assert len(metadata_calls) == 3

    # Fixed-shape route changes cannot reuse an older collective result.
    cache.get_or_build_owner_local(
        2,
        block_table,
        build=route_and_filter,
        **(route_kwargs | {"source_stride": 5}),
    )
    assert len(route_calls) == 3
    assert route_calls[-1].shape == (5, 3)


def test_owner_local_slot_cache_zero_rows_and_shape_guards(monkeypatch):
    def fake_convert(
        req_id,
        block_table,
        token_indices,
        *,
        out,
        valid_counts_out,
        **_,
    ):
        assert token_indices.shape[0] == 0
        return out, valid_counts_out

    monkeypatch.setattr(
        cache_module,
        "convert_global_indices_to_dcp_peer_slots",
        fake_convert,
    )
    cache = OwnerPeerSlotCache(
        torch.empty((4, 2), dtype=torch.int32),
        torch.empty(4, dtype=torch.int32),
    )
    block_table = torch.tensor([[3]], dtype=torch.int32)
    refresh_kwargs = {
        "dcp_size": 2,
        "blocks_per_peer": 16,
        "cp_kv_cache_interleave_size": 4,
        "block_size": 4,
    }
    cache.refresh(
        torch.empty(0, dtype=torch.int32),
        block_table,
        torch.empty((0, 2), dtype=torch.int32),
        **refresh_kwargs,
    )

    def build(padded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert torch.all(padded == -1)
        return (
            torch.full((6, 2), -1, dtype=torch.int32),
            torch.zeros(6, dtype=torch.int32),
        )

    slots, counts = cache.get_or_build_owner_local(
        0,
        block_table,
        source_stride=3,
        owner_rank=0,
        dcp_world_size=2,
        blocks_per_peer=16,
        cp_kv_cache_interleave_size=4,
        block_size=4,
        build=build,
    )
    assert slots.shape == (6, 2)
    assert torch.count_nonzero(counts) == 0

    with pytest.raises(RuntimeError, match="smaller than its active rows"):
        cache.get_or_build_owner_local(
            0,
            block_table,
            source_stride=0,
            owner_rank=0,
            dcp_world_size=2,
            blocks_per_peer=16,
            cp_kv_cache_interleave_size=4,
            block_size=4,
            build=build,
        )
