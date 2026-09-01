# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import (
    DPSyncCoordinator,
    DPSyncState,
    dispatch_cg_and_sync_dp,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _Work:
    def __init__(
        self,
        tensor: torch.Tensor,
        remote_tokens: int,
        remote_reqs: int = 2,
        generation_delta: int = 0,
        parent_generation_delta: int = 0,
        remote_need_eager: bool = False,
    ) -> None:
        self.tensor = tensor
        self.remote_tokens = remote_tokens
        self.remote_reqs = remote_reqs
        self.generation_delta = generation_delta
        self.parent_generation_delta = parent_generation_delta
        self.remote_need_eager = remote_need_eager
        self.wait_calls = 0

    def wait(self) -> None:
        self.wait_calls += 1
        self.tensor[0, 1] = self.remote_tokens
        self.tensor[1, 1] = CUDAGraphMode.FULL.value
        self.tensor[2, 1] = 1
        self.tensor[3, 1] = 1
        self.tensor[4, 1] = self.remote_reqs
        self.tensor[5, 1] = self.tensor[5, 0] + self.generation_delta
        self.tensor[6, 1] = self.tensor[6, 0] + self.parent_generation_delta
        self.tensor[7, 1] = int(self.remote_need_eager)


def _graph_manager() -> Mock:
    manager = Mock(spec=CudaGraphManager)
    manager.dispatch.side_effect = lambda num_reqs, num_tokens, *args, **kwargs: (
        BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            uniform_token_count=1,
        )
    )
    return manager


def test_async_dp_sync_waits_on_result_and_reuses_buffer(monkeypatch):
    tensors = []
    works = []

    def all_reduce(tensor, group, async_op):
        assert async_op
        tensors.append(tensor)
        work = _Work(tensor, remote_tokens=4)
        works.append(work)
        return work

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    coordinator = DPSyncCoordinator(2, 0, group=Mock())
    manager = _graph_manager()

    future = coordinator.start(manager, 2, 2, uniform_token_count=1)

    assert works[0].wait_calls == 0
    with pytest.raises(RuntimeError, match="already in flight"):
        coordinator.start(manager, 2, 2, uniform_token_count=1)

    batch_desc, sync = future.result(manager)

    assert works[0].wait_calls == 1
    assert batch_desc.num_tokens == 4
    assert sync is not None
    assert sync.num_tokens_across_dp.tolist() == [4, 4]
    assert sync.generation == 0
    assert sync.live_num_tokens_across_dp == (2, 4)
    assert sync.live_num_reqs_across_dp == (2, 2)
    assert future.result(manager) == (batch_desc, sync)
    assert works[0].wait_calls == 1

    future.release()
    future.release()
    next_future = coordinator.start(manager, 2, 2, uniform_token_count=1)
    assert tensors[1] is tensors[0]
    next_future.release()


def test_async_dp_sync_release_waits_for_unconsumed_work(monkeypatch):
    works = []

    def all_reduce(tensor, group, async_op):
        work = _Work(tensor, remote_tokens=2)
        works.append(work)
        return work

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    coordinator = DPSyncCoordinator(2, 0, group=Mock())
    manager = _graph_manager()

    future = coordinator.start(manager, 2, 2, uniform_token_count=1)
    future.release()

    assert works[0].wait_calls == 1
    replacement = coordinator.start(manager, 2, 2, uniform_token_count=1)
    replacement.release()


def test_async_dp_sync_does_not_wait_twice_after_resolution_error(monkeypatch):
    work = None

    def all_reduce(tensor, group, async_op):
        nonlocal work
        work = _Work(tensor, remote_tokens=2)
        return work

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    coordinator = DPSyncCoordinator(2, 0, group=Mock())
    manager = _graph_manager()
    manager.dispatch.side_effect = [
        BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=2,
            num_reqs=2,
            uniform_token_count=1,
        ),
        RuntimeError("dispatch failed"),
    ]

    future = coordinator.start(manager, 2, 2, uniform_token_count=1)
    with pytest.raises(RuntimeError, match="dispatch failed"):
        future.result(manager)
    assert work is not None
    assert work.wait_calls == 1

    future.release()
    assert work.wait_calls == 1


def test_execution_contract_ignores_inactive_rank_graph_mode(monkeypatch):
    def all_reduce(tensor, group, async_op):
        return _Work(tensor, remote_tokens=3, remote_reqs=2)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    manager = Mock(spec=CudaGraphManager)

    def dispatch(num_reqs, num_tokens, *args, **kwargs):
        if num_tokens == 0:
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=0,
                num_reqs=0,
            )
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=4,
            num_reqs=4,
            uniform_token_count=1,
            max_query_len=1,
        )

    manager.dispatch.side_effect = dispatch
    coordinator = DPSyncCoordinator(2, 0, group=Mock(), execution_contract=True)

    future = coordinator.start(manager, 0, 0, uniform_token_count=None)
    batch_desc, sync = future.result(manager)

    assert batch_desc.cg_mode == CUDAGraphMode.FULL
    assert batch_desc.num_tokens == 4
    assert batch_desc.num_reqs == 4
    assert sync is not None
    assert sync.num_tokens_across_dp.tolist() == [4, 4]
    assert sync.live_num_tokens_across_dp == (0, 3)
    assert sync.live_num_reqs_across_dp == (0, 2)
    future.release()


def test_async_dp_sync_rejects_generation_mismatch(monkeypatch):
    work = None

    def all_reduce(tensor, group, async_op):
        nonlocal work
        work = _Work(tensor, remote_tokens=2, generation_delta=1)
        return work

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    coordinator = DPSyncCoordinator(2, 0, group=Mock())
    manager = _graph_manager()

    future = coordinator.start(manager, 2, 2, uniform_token_count=1)
    with pytest.raises(RuntimeError, match="generation mismatch"):
        future.result(manager)

    future.release()
    assert work is not None
    assert work.wait_calls == 1


def test_async_dp_sync_rejects_parent_generation_mismatch(monkeypatch):
    def all_reduce(tensor, group, async_op):
        return _Work(
            tensor,
            remote_tokens=2,
            parent_generation_delta=1,
        )

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    coordinator = DPSyncCoordinator(2, 0, group=Mock(), lane="speculator")
    manager = _graph_manager()

    future = coordinator.start(
        manager,
        2,
        2,
        uniform_token_count=1,
        parent_generation=9,
    )
    with pytest.raises(RuntimeError, match="parent generation mismatch"):
        future.result(manager)
    future.release()


def test_execution_contract_selects_global_eager_fallback(monkeypatch):
    def all_reduce(tensor, group, async_op):
        return _Work(
            tensor,
            remote_tokens=3,
            remote_reqs=2,
            remote_need_eager=True,
        )

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    coordinator = DPSyncCoordinator(2, 0, group=Mock(), execution_contract=True)

    future = coordinator.start(_graph_manager(), 0, 0, uniform_token_count=None)
    batch_desc, sync = future.result(_graph_manager())

    assert batch_desc.cg_mode == CUDAGraphMode.NONE
    assert batch_desc.num_tokens == 3
    assert batch_desc.num_reqs == 2
    assert sync is not None and sync.eager
    future.release()


def test_execution_contract_returns_empty_when_all_ranks_are_idle(monkeypatch):
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, group, async_op: _Work(tensor, remote_tokens=0, remote_reqs=0),
    )
    coordinator = DPSyncCoordinator(2, 0, group=Mock(), execution_contract=True)

    future = coordinator.start(_graph_manager(), 0, 0, uniform_token_count=None)
    batch_desc, sync = future.result(_graph_manager())

    assert batch_desc.num_tokens == 0
    assert batch_desc.num_reqs == 0
    assert sync is None
    future.release()


def test_target_and_speculator_use_distinct_collective_lanes(monkeypatch):
    base_group = Mock()
    speculator_group = Mock()
    groups = []

    monkeypatch.setattr(
        dp_utils,
        "get_dp_group",
        lambda: SimpleNamespace(cpu_group=base_group),
    )
    monkeypatch.setattr(
        torch.distributed,
        "get_process_group_ranks",
        lambda group: [0, 1],
    )
    new_group = Mock(return_value=speculator_group)
    monkeypatch.setattr(torch.distributed, "new_group", new_group)

    work = Mock()
    work.wait.return_value = None

    def all_reduce(tensor, group, async_op):
        groups.append(group)
        return work

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    dp_utils._DP_SYNC_GROUPS.clear()
    try:
        target = DPSyncCoordinator(2, 0, lane="target")
        target.start(_graph_manager(), 1, 1, uniform_token_count=1).release()
        speculator = DPSyncCoordinator(2, 0, lane="speculator")
        speculator.start(_graph_manager(), 1, 1, uniform_token_count=1).release()
    finally:
        dp_utils._DP_SYNC_GROUPS.clear()

    assert groups == [base_group, speculator_group]
    new_group.assert_called_once_with(
        ranks=[0, 1],
        backend="gloo",
        use_local_synchronization=True,
    )


def test_reused_execution_contract_selects_global_request_capacity():
    manager = _graph_manager()
    sync = DPSyncState(
        num_tokens_across_dp=torch.tensor([4, 4]),
        uniform_token_count=1,
        eager=False,
        execution_num_reqs=4,
    )

    batch_desc, reused = dispatch_cg_and_sync_dp(
        manager,
        num_reqs=1,
        num_tokens=4,
        uniform_token_count=1,
        dp_size=2,
        dp_rank=0,
        dp_sync=sync,
    )

    assert batch_desc.num_reqs == 4
    assert reused is sync
    assert manager.dispatch.call_args.args[:2] == (4, 4)
