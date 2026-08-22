# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
import time
from multiprocessing.process import BaseProcess
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import vllm.distributed.elastic_ep.elastic_execute as elastic_execute
from tests.utils import multi_gpu_test
from vllm.compilation.cuda_graph import CUDAGraphEntry, CUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor
from vllm.forward_context import BatchDescriptor
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


def _make_executor(cudagraph_mode: CUDAGraphMode) -> ElasticEPScalingExecutor:
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode)
        ),
    )
    executor = object.__new__(ElasticEPScalingExecutor)
    executor.worker_ref = lambda: worker
    return executor


@pytest.mark.parametrize(
    ("is_rocm", "cudagraph_mode", "expected"),
    [
        (True, CUDAGraphMode.NONE, False),
        (True, CUDAGraphMode.PIECEWISE, True),
        (True, CUDAGraphMode.FULL, True),
        (True, CUDAGraphMode.FULL_DECODE_ONLY, True),
        (True, CUDAGraphMode.FULL_AND_PIECEWISE, True),
        (False, CUDAGraphMode.NONE, False),
        (False, CUDAGraphMode.FULL_AND_PIECEWISE, False),
    ],
)
def test_should_defer_target_group_warmup(
    monkeypatch: pytest.MonkeyPatch,
    is_rocm: bool,
    cudagraph_mode: CUDAGraphMode,
    expected: bool,
) -> None:
    platform = SimpleNamespace(is_rocm=lambda: is_rocm)
    monkeypatch.setattr("vllm.platforms.current_platform", platform)
    executor = _make_executor(cudagraph_mode)

    assert executor._should_defer_target_group_warmup() is expected


@pytest.mark.parametrize("defer", [False, True])
def test_maybe_warm_target_groups_during_prepare(defer: bool) -> None:
    executor = _make_executor(CUDAGraphMode.FULL_AND_PIECEWISE)
    executor._should_defer_target_group_warmup = Mock(return_value=defer)
    executor._warm_dp_ep_device_groups = Mock()
    dp_group = object()
    ep_group = object()

    executor._maybe_warm_target_groups_during_prepare(dp_group, ep_group)

    if defer:
        executor._warm_dp_ep_device_groups.assert_not_called()
    else:
        executor._warm_dp_ep_device_groups.assert_called_once_with(dp_group, ep_group)


@pytest.mark.parametrize("defer", [False, True])
def test_deferred_target_group_warmup_runs_between_graph_release_and_recapture(
    monkeypatch: pytest.MonkeyPatch, defer: bool
) -> None:
    events: list[object] = []

    class FakeMultiBlockTable:
        block_tables: list[object] = []

        def clear(self) -> None:
            events.append("clear")

    runner = SimpleNamespace(
        input_batch=SimpleNamespace(block_table=FakeMultiBlockTable()),
        max_num_tokens=32,
    )
    runner._dummy_run = lambda *args, **kwargs: events.append("dummy")
    worker = SimpleNamespace(model_runner=runner)
    worker.compile_or_warm_up_model = lambda: events.append("compile")

    executor = object.__new__(ElasticEPScalingExecutor)
    executor.worker_ref = lambda: worker
    executor._release_cuda_graphs = lambda: events.append("release")
    executor._should_defer_target_group_warmup = lambda: defer
    executor._warm_dp_ep_device_groups = lambda dp, ep: events.append(("warm", dp, ep))

    dp_group = object()
    ep_group = object()
    monkeypatch.setattr(elastic_execute, "get_dp_group", lambda: dp_group)
    monkeypatch.setattr(elastic_execute, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(
        elastic_execute, "unlock_workspace", lambda: events.append("unlock")
    )
    monkeypatch.setattr(
        elastic_execute, "lock_workspace", lambda: events.append("lock")
    )

    executor.warm_and_capture()

    expected: list[object] = ["clear", "release"]
    if defer:
        expected.append(("warm", dp_group, ep_group))
    expected.extend(["unlock", "dummy", "compile", "lock"])
    assert events == expected


@pytest.mark.parametrize("is_existing_worker", [False, True])
def test_commit_scale_up_warms_every_target_member(
    is_existing_worker: bool,
) -> None:
    executor = _make_executor(CUDAGraphMode.FULL_AND_PIECEWISE)
    executor.worker.model_runner = SimpleNamespace(setup_eplb_from_mapping=Mock())
    retired_groups = object()
    mapping = object()
    executor.broadcast_expert_mapping = Mock()
    executor.switch_and_prepare = Mock(return_value=retired_groups)
    executor.receive_expert_mapping = Mock(return_value=mapping)
    executor.warm_and_capture = Mock()
    executor._perform_eplb_reshuffle = Mock()
    executor._start_group_cleanup = Mock()

    executor.commit_scale_up(is_existing_worker)

    executor.warm_and_capture.assert_called_once_with()
    executor._perform_eplb_reshuffle.assert_called_once_with(async_op=True)
    if is_existing_worker:
        executor.broadcast_expert_mapping.assert_called_once_with()
        executor.switch_and_prepare.assert_called_once_with()
        executor._start_group_cleanup.assert_called_once_with(retired_groups)
        executor.receive_expert_mapping.assert_not_called()
        executor.worker.model_runner.setup_eplb_from_mapping.assert_not_called()
    else:
        executor.broadcast_expert_mapping.assert_not_called()
        executor.switch_and_prepare.assert_not_called()
        executor.receive_expert_mapping.assert_called_once_with()
        executor.worker.model_runner.setup_eplb_from_mapping.assert_called_once_with(
            mapping
        )
        executor._start_group_cleanup.assert_not_called()


@pytest.mark.parametrize(("removing", "expected_warmups"), [(False, 1), (True, 0)])
def test_commit_scale_down_warms_only_surviving_target_members(
    removing: bool, expected_warmups: int
) -> None:
    executor = _make_executor(CUDAGraphMode.FULL_AND_PIECEWISE)
    retired_groups = object()
    executor.perform_scale_down_eplb_reshuffle = Mock()
    executor.switch_and_remove = Mock()
    executor.switch_and_prepare = Mock(return_value=retired_groups)
    executor.warm_and_capture = Mock()
    executor._start_group_cleanup = Mock()

    executor.commit_scale_down(new_dp_size=2, removing=removing)

    executor.perform_scale_down_eplb_reshuffle.assert_called_once_with(2)
    assert executor.warm_and_capture.call_count == expected_warmups
    if removing:
        executor.switch_and_remove.assert_called_once_with()
        executor.switch_and_prepare.assert_not_called()
        executor._start_group_cleanup.assert_not_called()
    else:
        executor.switch_and_remove.assert_not_called()
        executor.switch_and_prepare.assert_called_once_with()
        executor._start_group_cleanup.assert_called_once_with(retired_groups)


def _deferred_warmup_worker(rank: int, world_size: int, port: int) -> None:
    active_group = None
    dp_device_group = None
    ep_device_group = None
    executor = None
    wrapper = None
    graph = None
    captured_graph = None
    try:
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(rank)
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
        )
        ranks = list(range(world_size))
        active_group = dist.new_group(ranks, backend="nccl")

        active_tensor = torch.tensor([rank + 1.0], device=device)
        dist.all_reduce(active_tensor, group=active_group)
        torch.accelerator.synchronize()

        active_tensor.fill_(rank + 1.0)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dist.all_reduce(active_tensor, group=active_group)
        torch.accelerator.synchronize()
        dist.barrier()

        config = VllmConfig()
        config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
        wrapper = CUDAGraphWrapper(nn.Identity(), config, CUDAGraphMode.FULL)
        descriptor = BatchDescriptor(num_tokens=1)
        wrapper.concrete_cudagraph_entries[descriptor] = CUDAGraphEntry(
            batch_descriptor=descriptor,
            cudagraph=graph,
            output=active_tensor,
        )
        graph = None

        # Creating fresh target groups after capture mirrors the problematic
        # ordering while leaving their RCCL communicators lazily initialized.
        dp_device_group = dist.new_group(ranks, backend="nccl")
        ep_device_group = dist.new_group(ranks, backend="nccl")
        dp_group = SimpleNamespace(device=device, device_group=dp_device_group)
        ep_group = SimpleNamespace(device=device, device_group=ep_device_group)

        block_table = SimpleNamespace(block_tables=[])
        block_table.clear = block_table.block_tables.clear

        class Runner:
            def __init__(self) -> None:
                self.model = wrapper
                self.input_batch = SimpleNamespace(block_table=block_table)
                self.max_num_tokens = 1
                self.dummy_ran = False

            def get_model(self):
                return self.model

            def _dummy_run(self, *args, **kwargs) -> None:
                assert executor is not None and executor.warm_calls == 1
                assert descriptor not in wrapper.concrete_cudagraph_entries
                self.dummy_ran = True

        class Worker:
            def __init__(self) -> None:
                self.vllm_config = config
                self.model_runner = Runner()
                self.device = device
                self.rank = rank

            def compile_or_warm_up_model(self) -> None:
                assert self.model_runner.dummy_ran

        class RecordingExecutor(ElasticEPScalingExecutor):
            def __init__(
                self,
                worker,
                graph_wrapper: CUDAGraphWrapper,
                batch_descriptor: BatchDescriptor,
            ) -> None:
                super().__init__(worker)
                self.graph_wrapper = graph_wrapper
                self.batch_descriptor = batch_descriptor
                self.warm_calls = 0

            def _warm_dp_ep_device_groups(self, dp_group, ep_group) -> None:
                assert (
                    self.batch_descriptor
                    not in self.graph_wrapper.concrete_cudagraph_entries
                )
                super()._warm_dp_ep_device_groups(dp_group, ep_group)
                self.warm_calls += 1

        worker = Worker()
        executor = RecordingExecutor(worker, wrapper, descriptor)

        executor._maybe_warm_target_groups_during_prepare(dp_group, ep_group)
        assert executor.warm_calls == 0

        active_tensor.fill_(rank + 3.0)
        dist.barrier()
        captured_graph = wrapper.concrete_cudagraph_entries[descriptor].cudagraph
        assert captured_graph is not None
        captured_graph.replay()
        torch.accelerator.synchronize()
        assert active_tensor.item() == 7.0
        captured_graph = None

        with (
            patch.object(elastic_execute, "get_dp_group", return_value=dp_group),
            patch.object(elastic_execute, "get_ep_group", return_value=ep_group),
            patch.object(elastic_execute, "unlock_workspace"),
            patch.object(elastic_execute, "lock_workspace"),
        ):
            executor.warm_and_capture()
        assert descriptor not in wrapper.concrete_cudagraph_entries
        assert executor.warm_calls == 1
    finally:
        captured_graph = None
        graph = None
        if wrapper is not None:
            wrapper.clear_graphs()
            torch.accelerator.synchronize()
        if executor is not None:
            executor.shutdown()
        for group in (ep_device_group, dp_device_group):
            if group is not None:
                dist.destroy_process_group(group)
        if active_group is not None:
            dist.destroy_process_group(active_group)
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only RCCL test")
@multi_gpu_test(num_gpus=2)
def test_deferred_warmup_uses_real_rccl_after_graph_release() -> None:
    world_size = 2
    port = get_open_port()
    ctx = mp.get_context("spawn")
    candidates: list[BaseProcess] = [
        ctx.Process(
            target=_deferred_warmup_worker,
            args=(rank, world_size, port),
            name=f"rccl-warmup-rank-{rank}",
        )
        for rank in range(world_size)
    ]
    processes: list[BaseProcess] = []
    try:
        for candidate in candidates:
            candidate.start()
            processes.append(candidate)

        deadline = time.monotonic() + 120
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        timed_out = [process for process in processes if process.is_alive()]
        assert not timed_out, (
            "RCCL warmup workers timed out: "
            f"{[(process.name, process.pid) for process in timed_out]}"
        )
        for process in processes:
            assert process.exitcode == 0, (
                f"{process.name} exited with code {process.exitcode}"
            )
    finally:
        alive = [process for process in processes if process.is_alive()]
        for process in alive:
            process.terminate()
        for process in alive:
            process.join(timeout=5)
        survivors = [process for process in alive if process.is_alive()]
        for process in survivors:
            process.kill()
        for process in survivors:
            process.join(timeout=10)
