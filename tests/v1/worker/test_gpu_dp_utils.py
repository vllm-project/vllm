# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncCoordinator

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _Work:
    def __init__(self, tensor: torch.Tensor, remote_tokens: int) -> None:
        self.tensor = tensor
        self.remote_tokens = remote_tokens
        self.wait_calls = 0

    def wait(self) -> None:
        self.wait_calls += 1
        self.tensor[:, 1] = torch.tensor(
            [self.remote_tokens, CUDAGraphMode.FULL.value, 1, -1],
            dtype=torch.int32,
        )


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
    assert future.result(manager) == (batch_desc, sync)
    assert works[0].wait_calls == 1

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
