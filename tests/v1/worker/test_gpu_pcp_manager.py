# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
from vllm.v1.worker.gpu.pcp_manager import PCPManager


def _copy_to_cpu(value, out=None, device=None):
    tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    if out is not None:
        return out.copy_(tensor)
    return tensor


def test_replicated_decode_piecewise_graph_padding(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)

    segments_by_rank, per_rank_num_tokens = manager._build_batch_layout(
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        num_computed_tokens=np.full(3, 16, dtype=np.int32),
        is_prefilling=np.zeros(3, dtype=np.bool_),
        query_start_loc_np=np.arange(4, dtype=np.int32),
        padded_num_tokens=4,
    )

    assert per_rank_num_tokens == [3, 3]
    request_indices = [
        [segment.global_batch_req_idx for segment in rank] for rank in segments_by_rank
    ]
    assert request_indices == [[0, 1, 2], [0, 1, 2]]
    assert torch.equal(manager._hidden_restore_idx, torch.tensor([0, 1, 2]))
    assert torch.equal(
        manager._padded_gather_idx,
        torch.tensor([0, 1, 2, 0, 0, 1, 2, 0]),
    )
    assert torch.equal(
        manager._gathered_kv_write_mask,
        torch.tensor([True, True, True, False, False, False, False, False]),
    )


def test_input_buffers_are_exposed_for_cudagraph_capture():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=8,
    )

    assert manager.input_buffers is manager._input_buffers
    assert manager.input_buffers.input_ids.shape == (8,)
    assert manager.input_buffers.positions.shape == (8,)
    assert manager.input_buffers.is_padding.shape == (8,)


@pytest.mark.parametrize(
    ("pcp_world_size", "num_scheduled_tokens", "is_prefilling", "expected"),
    [
        (2, [8], [True], 4),
        (2, [7], [True], 4),
        (2, [3], [False], 3),
        (2, [3, 8], [False, True], 7),
        (4, [2, 9], [False, True], 5),
    ],
)
def test_num_tokens_for_dispatch_uses_largest_pcp_rank(
    pcp_world_size, num_scheduled_tokens, is_prefilling, expected
):
    manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=0,
        device=torch.device("cpu"),
    )

    _, actual, _ = manager.get_dispatch_batch_shape(
        np.asarray(num_scheduled_tokens, dtype=np.int32),
        np.asarray(is_prefilling, dtype=np.bool_),
    )

    assert actual == expected


def test_graph_padding_cannot_be_smaller_than_largest_pcp_rank(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)

    with pytest.raises(ValueError, match="smaller than the largest rank-local batch"):
        manager._build_batch_layout(
            num_scheduled_tokens=np.ones(3, dtype=np.int32),
            num_computed_tokens=np.full(3, 16, dtype=np.int32),
            is_prefilling=np.zeros(3, dtype=np.bool_),
            query_start_loc_np=np.arange(4, dtype=np.int32),
            padded_num_tokens=2,
        )


def _schedule_for_rank(pcp_rank: int, pcp_world_size: int, monkeypatch):
    """Run one PCP rank's _build_batch_layout and return its schedule."""
    manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=pcp_rank,
        device=torch.device("cpu"),
        dcp_world_size=8,
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)
    # Two prefills of 32 tokens each, one with 100 tokens already computed.
    num_scheduled_tokens = np.full(2, 32, dtype=np.int32)
    manager._build_batch_layout(
        num_scheduled_tokens=num_scheduled_tokens,
        num_computed_tokens=np.array([0, 100], dtype=np.int32),
        is_prefilling=np.ones(2, dtype=np.bool_),
        query_start_loc_np=np.array([0, 32, 64], dtype=np.int32),
    )
    return manager.schedule_seq_lens_np


def test_schedule_context_lens_are_identical_across_pcp_ranks(monkeypatch):
    """The DCP gather schedule must not depend on which PCP rank builds it."""
    pcp_world_size = 4
    schedules = [
        _schedule_for_rank(rank, pcp_world_size, monkeypatch)
        for rank in range(pcp_world_size)
    ]
    for rank, schedule in enumerate(schedules[1:], start=1):
        assert np.array_equal(schedules[0], schedule), (
            f"rank {rank} schedule {schedule} != rank 0 schedule {schedules[0]}"
        )


def test_schedule_context_lens_bound_every_rank_local_context(monkeypatch):
    """The schedule must bound every rank's local extent, and no more."""
    pcp_world_size = 4
    schedule = _schedule_for_rank(0, pcp_world_size, monkeypatch)
    num_scheduled_tokens = np.full(2, 32, dtype=np.int32)
    num_computed = np.array([0, 100], dtype=np.int32)
    is_prefilling = np.ones(2, dtype=np.bool_)

    for req_idx in range(2):
        assert schedule[req_idx] == num_computed[req_idx] + 32

    reached = {req_idx: 0 for req_idx in range(2)}
    for rank in range(pcp_world_size):
        manager = PCPManager(
            pcp_world_size=pcp_world_size, pcp_rank=rank, device=torch.device("cpu")
        )
        for req_idx, chunk_offset, chunk_len in manager._iter_rank_chunks(
            rank, num_scheduled_tokens, is_prefilling
        ):
            local_extent = num_computed[req_idx] + chunk_offset + chunk_len
            assert local_extent <= schedule[req_idx]
            reached[req_idx] = max(reached[req_idx], local_extent)
    assert reached == {0: int(schedule[0]), 1: int(schedule[1])}


def _published_schedule_for_rank(
    pcp_rank: int,
    pcp_world_size: int,
    num_scheduled_tokens: np.ndarray,
    num_computed_tokens: np.ndarray,
    is_prefilling: np.ndarray,
    monkeypatch,
    padded_num_tokens: int | None = None,
):
    """Build one PCP rank's layout and return the schedule it publishes."""
    manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=pcp_rank,
        device=torch.device("cpu"),
        dcp_world_size=2,
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)
    query_start_loc_np = np.concatenate([[0], np.cumsum(num_scheduled_tokens)]).astype(
        np.int32
    )
    manager._build_batch_layout(
        num_scheduled_tokens=num_scheduled_tokens,
        num_computed_tokens=num_computed_tokens,
        is_prefilling=is_prefilling,
        query_start_loc_np=query_start_loc_np,
        padded_num_tokens=padded_num_tokens,
    )
    return pcp_manager_module.get_current_pcp_schedule()


@pytest.mark.parametrize(
    ("num_scheduled_tokens", "num_computed_tokens"),
    [
        ([32, 32], [0, 0]),
        ([32, 32, 32, 32], [0, 0, 0, 0]),
        ([32, 32], [100, 0]),  # one continued, one fresh
        ([64, 32, 48], [0, 512, 0]),  # ragged lengths and mixed contexts
        # A replicated prefill BEHIND a split one. Sorting replicated rows first
        # put request 1 ahead of request 0.
        ([64, 3], [0, 0]),
        ([64, 3, 64], [0, 0, 0]),
    ],
)
@pytest.mark.parametrize("pcp_world_size", [2, 4])
def test_published_row_order_is_identical_on_every_pcp_rank(
    num_scheduled_tokens, num_computed_tokens, pcp_world_size, monkeypatch
):
    """Every rank must map its rows to the same global requests, in the same order."""
    num_scheduled_tokens = np.array(num_scheduled_tokens, dtype=np.int32)
    num_computed_tokens = np.array(num_computed_tokens, dtype=np.int32)
    is_prefilling = np.ones(len(num_scheduled_tokens), dtype=np.bool_)

    orders = [
        _published_schedule_for_rank(
            rank,
            pcp_world_size,
            num_scheduled_tokens,
            num_computed_tokens,
            is_prefilling,
            monkeypatch,
        ).local_to_global_req_idx_np
        for rank in range(pcp_world_size)
    ]
    for rank, order in enumerate(orders[1:], start=1):
        assert np.array_equal(orders[0], order), (
            f"rank {rank} rows map to {order.tolist()}, rank 0 to {orders[0].tolist()}"
        )
    # Grouped by request and ascending: what the indexer plan indexes by.
    assert np.all(np.diff(orders[0]) >= 0)


def test_published_row_order_puts_every_decode_before_every_prefill(monkeypatch):
    """split_decodes_and_prefills takes the FIRST prefilling row as the boundary."""
    # req 0 is a continued prefill, req 1 is a decode.
    schedule = _published_schedule_for_rank(
        pcp_rank=0,
        pcp_world_size=2,
        num_scheduled_tokens=np.array([32, 1], dtype=np.int32),
        num_computed_tokens=np.array([100, 20], dtype=np.int32),
        is_prefilling=np.array([True, False], dtype=np.bool_),
        monkeypatch=monkeypatch,
    )
    # The decode (request 1) must be row 0.
    assert schedule.local_to_global_req_idx_np[0] == 1


@pytest.mark.parametrize("query_len", [1, 2, 3, 5, 6, 9])
def test_dcp_replicates_prefills_too_short_to_split(query_len, monkeypatch):
    """A prefill that cannot fill 2*pcp chunks is replicated, not split."""
    pcp_world_size = 2
    num_scheduled_tokens = np.array([query_len], dtype=np.int32)
    is_prefilling = np.ones(1, dtype=np.bool_)

    rows_per_rank = []
    for rank in range(pcp_world_size):
        manager = PCPManager(
            pcp_world_size=pcp_world_size,
            pcp_rank=rank,
            device=torch.device("cpu"),
            dcp_world_size=2,
        )
        assert manager.replicated_requests(num_scheduled_tokens, is_prefilling)[0]
        rows = list(
            manager._iter_rank_chunks(rank, num_scheduled_tokens, is_prefilling)
        )
        assert rows == [(0, 0, query_len)]
        rows_per_rank.append(rows)
    assert rows_per_rank[0] == rows_per_rank[1]
