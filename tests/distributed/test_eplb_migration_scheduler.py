# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.config.parallel import EPLBConfig
from vllm.distributed.eplb.eplb_communicator import EplbCommunicator
from vllm.distributed.eplb.migration_scheduler import (
    MigrationFlow,
    schedule_migration_batches,
)
from vllm.distributed.eplb.rebalance_execute import move_from_buffer, move_to_buffer


class _MockEplbCommunicator(EplbCommunicator):
    def __init__(self) -> None:
        self.send_calls: list[tuple[list[torch.Tensor], int, int]] = []
        self.recv_calls: list[tuple[list[torch.Tensor], int, int]] = []
        self.context_calls = 0
        self.execute_count = 0

    def add_send(
        self, tensors: list[torch.Tensor], dst_rank: int, expert_id: int
    ) -> None:
        self.send_calls.append((tensors, dst_rank, expert_id))

    def add_recv(
        self, tensors: list[torch.Tensor], src_rank: int, expert_id: int
    ) -> None:
        self.recv_calls.append((tensors, src_rank, expert_id))

    def set_transfer_context(self, old_indices: np.ndarray, layer_idx: int) -> None:
        self.context_calls += 1

    def execute(self) -> None:
        self.execute_count += 1


def test_migration_batching_is_enabled_by_default_and_can_be_disabled() -> None:
    assert EPLBConfig().enable_migration_batching
    assert not EPLBConfig(enable_migration_batching=False).enable_migration_batching


def test_schedule_migration_batches_is_deterministic() -> None:
    transfers = [
        (1, 3, 0),
        (2, 4, 1),
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 5),
    ]
    num_local_experts, old_indices, new_indices = _placements_for_transfers(
        5, transfers
    )

    batches = schedule_migration_batches(num_local_experts, old_indices, new_indices)

    assert [_endpoints(batch) for batch in batches] == [
        [(1, 3), (2, 4)],
        [(0, 1)],
        [(0, 2)],
        [(0, 3)],
        [(0, 4)],
    ]
    for batch in batches:
        _assert_no_endpoint_conflict(batch)


def test_schedule_migration_batches_coalesces_rank_pair() -> None:
    transfers = [
        (0, 1, 0),
        (0, 1, 1),
        (2, 3, 2),
    ]
    num_local_experts, old_indices, new_indices = _placements_for_transfers(
        4, transfers
    )

    assert schedule_migration_batches(num_local_experts, old_indices, new_indices) == [
        [
            MigrationFlow(0, 1, expert_ids=(0, 1)),
            MigrationFlow(2, 3, expert_ids=(2,)),
        ]
    ]


def test_schedule_migration_batches_covers_random_instructions() -> None:
    rng = np.random.default_rng(42)
    transfers = []
    for expert_id in range(80):
        src_rank, dst_rank = rng.integers(0, 8, size=2).tolist()
        if src_rank != dst_rank:
            transfers.append((src_rank, dst_rank, expert_id))
    num_local_experts, old_indices, new_indices = _placements_for_transfers(
        8, transfers
    )

    batches = schedule_migration_batches(num_local_experts, old_indices, new_indices)

    scheduled = _flatten_transfers(batches)
    assert len(scheduled) == len(transfers)
    assert set(scheduled) == set(transfers)
    for batch in batches:
        _assert_no_endpoint_conflict(batch)


def test_schedule_migration_batches_excludes_local_copies() -> None:
    # Old: rank 0 [0, 1], rank 1 [1, 2]. New: rank 1 needs expert 0.
    old_indices = np.array([0, 1, 1, 2], dtype=np.int64)
    new_indices = np.array([0, 1, 0, 2], dtype=np.int64)

    assert schedule_migration_batches(2, old_indices, new_indices) == [
        [MigrationFlow(0, 1, expert_ids=(0,))]
    ]


def test_schedule_migration_batches_requires_matching_shapes() -> None:
    old_indices = np.array([0, 1], dtype=np.int64)
    new_indices = np.array([0], dtype=np.int64)

    with pytest.raises(AssertionError):
        schedule_migration_batches(1, old_indices, new_indices)


def test_schedule_migration_batches_balances_replicas() -> None:
    # Rank 0 and rank 1 hold expert 0; ranks 2, 3, and 4 need it.
    old_indices = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    new_indices = np.array([0, 0, 0, 0, 0], dtype=np.int64)

    batches = schedule_migration_batches(1, old_indices, new_indices)

    assert set(_flatten_transfers(batches)) == {
        (0, 2, 0),
        (0, 4, 0),
        (1, 3, 0),
    }


def test_move_to_buffer_uses_multiple_batches() -> None:
    old_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    new_indices = np.array([1, 2, 3, 0], dtype=np.int64)
    communicator = _MockEplbCommunicator()

    move_to_buffer(
        num_local_experts=1,
        old_indices=old_indices,
        new_indices=new_indices,
        expert_weights=[torch.zeros(1, 1)],
        expert_weights_buffers=[torch.zeros(1, 1)],
        cuda_stream=None,
        ep_rank=0,
        communicator=communicator,
        layer_idx=7,
        enable_migration_batching=True,
    )

    assert communicator.context_calls == 2
    assert communicator.execute_count == 2
    assert [(dst, expert) for _, dst, expert in communicator.send_calls] == [(3, 0)]
    assert [(src, expert) for _, src, expert in communicator.recv_calls] == [(1, 1)]


def test_move_to_buffer_uses_primary_duplicate_destination() -> None:
    old_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    new_indices = np.array([2, 2, 0, 3], dtype=np.int64)
    expert_weights = [torch.zeros(2, 1)]
    recv_buffer = torch.zeros(2, 1)
    communicator = _MockEplbCommunicator()

    transfer_metadata = move_to_buffer(
        num_local_experts=2,
        old_indices=old_indices,
        new_indices=new_indices,
        expert_weights=expert_weights,
        expert_weights_buffers=[recv_buffer],
        cuda_stream=None,
        ep_rank=0,
        communicator=communicator,
        enable_migration_batching=True,
    )

    assert len(communicator.recv_calls) == 1
    recv_tensors, src_rank, expert_id = communicator.recv_calls[0]
    assert (src_rank, expert_id) == (1, 2)
    assert recv_tensors[0].data_ptr() == recv_buffer[0].data_ptr()

    recv_buffer[0].fill_(42)
    move_from_buffer(
        expert_weights=expert_weights,
        expert_weights_buffers=[recv_buffer],
        transfer_metadata=transfer_metadata,
        new_indices=new_indices,
        ep_rank=0,
    )
    torch.testing.assert_close(expert_weights[0], torch.full((2, 1), 42.0))


def _endpoints(
    flows: list[MigrationFlow],
) -> list[tuple[int, int]]:
    return [(item.src_rank, item.dst_rank) for item in flows]


def _assert_no_endpoint_conflict(batch: list[MigrationFlow]) -> None:
    endpoints: set[int] = set()
    for flow in batch:
        assert flow.src_rank not in endpoints
        assert flow.dst_rank not in endpoints
        endpoints.update((flow.src_rank, flow.dst_rank))


def _flatten_transfers(
    batches: list[list[MigrationFlow]],
) -> list[tuple[int, int, int]]:
    return [
        (flow.src_rank, flow.dst_rank, expert_id)
        for batch in batches
        for flow in batch
        for expert_id in flow.expert_ids
    ]


def _placements_for_transfers(
    num_ranks: int,
    transfers: list[tuple[int, int, int]],
) -> tuple[int, np.ndarray, np.ndarray]:
    old_counts = [0] * num_ranks
    new_counts = [0] * num_ranks
    for src_rank, dst_rank, _ in transfers:
        old_counts[src_rank] += 1
        new_counts[dst_rank] += 1
    num_local_experts = max(1, *old_counts, *new_counts)
    old_indices = np.full(num_ranks * num_local_experts, -1, dtype=np.int64)
    new_indices = np.full_like(old_indices, -1)
    old_counts = [0] * num_ranks
    new_counts = [0] * num_ranks
    for src_rank, dst_rank, expert_id in transfers:
        old_offset = src_rank * num_local_experts + old_counts[src_rank]
        new_offset = dst_rank * num_local_experts + new_counts[dst_rank]
        old_indices[old_offset] = expert_id
        new_indices[new_offset] = expert_id
        old_counts[src_rank] += 1
        new_counts[dst_rank] += 1
    return num_local_experts, old_indices, new_indices
