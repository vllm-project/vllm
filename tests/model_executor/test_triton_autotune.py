# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    get_world_group,
    init_distributed_environment,
)
from vllm.model_executor.warmup import triton_autotune
from vllm.model_executor.warmup.triton_autotune import (
    TunableConfigTable,
    TuningItem,
    run_config_tuning,
)

SINGLE_RANK = SimpleNamespace(rank_in_group=0, world_size=1, cpu_group=None)


class FakeTable(TunableConfigTable):
    name = "fake"

    def __init__(self, buckets, failing=()):
        self.buckets = buckets
        self.failing = set(failing)
        self.tuned = []
        self.committed = None

    def pending_items(self, worker):
        return [TuningItem(self.name, ("headdim", 64), b) for b in self.buckets]

    def tune(self, item):
        self.tuned.append(item.bucket)
        if item.bucket in self.failing:
            return None
        return {"BLOCK_SIZE_M": item.bucket}

    def commit(self, results):
        self.committed = {item.bucket: cfg for item, cfg in results.items()}


def test_single_rank_tunes_every_pending_item():
    table = FakeTable([1, 8, 16])
    run_config_tuning([table], None, SINGLE_RANK)
    assert sorted(table.tuned) == [1, 8, 16]
    assert table.committed == {b: {"BLOCK_SIZE_M": b} for b in (1, 8, 16)}


def test_nothing_pending_means_no_work():
    table = FakeTable([])
    assert run_config_tuning([table], None, SINGLE_RANK) == {}
    assert table.tuned == []
    assert table.committed is None


def test_failed_items_are_not_committed():
    table = FakeTable([1, 8], failing={8})
    run_config_tuning([table], None, SINGLE_RANK)
    assert table.committed == {1: {"BLOCK_SIZE_M": 1}}


def test_flag_off_does_nothing(monkeypatch):
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_triton_autotune=False)
        )
    )
    monkeypatch.setattr(
        triton_autotune, "_tables", lambda: pytest.fail("tables were built")
    )
    triton_autotune.triton_autotune(worker)


def _two_rank_worker(rank, init_method, queue):
    try:
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            backend="gloo",
            distributed_init_method=init_method,
        )
        # Rank 1 needs an extra bucket, like a pipeline stage with other layers.
        table = FakeTable([1, 8, 16] + ([32] if rank == 1 else []))
        run_config_tuning([table], None, get_world_group())
        queue.put((rank, table.tuned, table.committed))
    finally:
        destroy_distributed_environment()


def test_two_ranks_split_the_work_and_share_results(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_DISTRIBUTED_USE_SPLIT_GROUP", "0")
    queue = torch.multiprocessing.get_context("spawn").SimpleQueue()
    torch.multiprocessing.spawn(
        _two_rank_worker,
        args=(f"file://{tmp_path / 'store'}", queue),
        nprocs=2,
        join=True,
    )
    by_rank = {rank: (tuned, committed) for rank, tuned, committed in
               (queue.get() for _ in range(2))}
    assert sorted(by_rank[0][0] + by_rank[1][0]) == [1, 8, 16, 32]
    expected = {b: {"BLOCK_SIZE_M": b} for b in (1, 8, 16, 32)}
    assert by_rank[0][1] == by_rank[1][1] == expected