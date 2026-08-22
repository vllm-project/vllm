# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.config.parallel import EPLBConfig
from vllm.distributed.eplb.eplb_communicator import EplbCommunicator
from vllm.distributed.eplb.migration_scheduler import (
    MigrationInstruction,
    build_migration_instructions,
    schedule_migration_batches,
)
from vllm.distributed.eplb.rebalance_execute import move_to_buffer


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


def test_migration_batching_is_opt_in() -> None:
    assert not EPLBConfig().enable_migration_batching
    assert EPLBConfig(enable_migration_batching=True).enable_migration_batching


def test_schedule_migration_batches_is_deterministic() -> None:
    instructions = [
        MigrationInstruction(1, 3, expert_id=0),
        MigrationInstruction(2, 4, expert_id=1),
        MigrationInstruction(0, 1, expert_id=2),
        MigrationInstruction(0, 2, expert_id=3),
        MigrationInstruction(0, 3, expert_id=4),
        MigrationInstruction(0, 4, expert_id=5),
    ]

    batches = schedule_migration_batches(instructions)

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
    instructions = [
        MigrationInstruction(0, 1, expert_id=0),
        MigrationInstruction(0, 1, expert_id=1),
        MigrationInstruction(2, 3, expert_id=2),
    ]

    assert schedule_migration_batches(instructions) == [instructions]


def test_schedule_migration_batches_covers_random_instructions() -> None:
    rng = np.random.default_rng(42)
    instructions = []
    for expert_id in range(80):
        src_rank, dst_rank = rng.integers(0, 8, size=2).tolist()
        if src_rank != dst_rank:
            instructions.append(MigrationInstruction(src_rank, dst_rank, expert_id))

    batches = schedule_migration_batches(instructions)

    scheduled = [item for batch in batches for item in batch]
    assert len(scheduled) == len(instructions)
    assert set(scheduled) == set(instructions)
    for batch in batches:
        _assert_no_endpoint_conflict(batch)


def test_build_migration_instructions_excludes_local_copies() -> None:
    # Old: rank 0 [0, 1], rank 1 [1, 2]. New: rank 1 needs expert 0.
    old_indices = np.array([0, 1, 1, 2], dtype=np.int64)
    new_indices = np.array([0, 1, 0, 2], dtype=np.int64)

    assert build_migration_instructions(2, old_indices, new_indices) == [
        MigrationInstruction(0, 1, expert_id=0)
    ]


def test_build_migration_instructions_balances_replicas() -> None:
    # Rank 0 and rank 1 hold expert 0; ranks 2, 3, and 4 need it.
    old_indices = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    new_indices = np.array([0, 0, 0, 0, 0], dtype=np.int64)

    assert build_migration_instructions(1, old_indices, new_indices) == [
        MigrationInstruction(0, 2, expert_id=0),
        MigrationInstruction(0, 4, expert_id=0),
        MigrationInstruction(1, 3, expert_id=0),
    ]


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
    recv_buffer = torch.zeros(2, 1)
    communicator = _MockEplbCommunicator()

    move_to_buffer(
        num_local_experts=2,
        old_indices=old_indices,
        new_indices=new_indices,
        expert_weights=[torch.zeros(2, 1)],
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


def _endpoints(
    instructions: list[MigrationInstruction],
) -> list[tuple[int, int]]:
    return [(item.src_rank, item.dst_rank) for item in instructions]


def _assert_no_endpoint_conflict(batch: list[MigrationInstruction]) -> None:
    endpoints: set[int] = set()
    rank_pairs: set[tuple[int, int]] = set()
    for instruction in batch:
        rank_pair = (instruction.src_rank, instruction.dst_rank)
        if rank_pair not in rank_pairs:
            assert instruction.src_rank not in endpoints
            assert instruction.dst_rank not in endpoints
            endpoints.update(rank_pair)
            rank_pairs.add(rank_pair)
