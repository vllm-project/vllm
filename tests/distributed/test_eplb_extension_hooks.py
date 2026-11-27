# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import os
import socket
import sys
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import timedelta

import pytest
import torch
from transformers import LlamaConfig

import vllm.config.parallel as parallel_config_module
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.config.parallel import EPLBConfig
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.distributed.eplb.async_worker import (
    _transfer_cycle,
    run_rebalance_experts,
    transfer_run_periodically,
)
from vllm.distributed.eplb.eplb_communicator import TorchDistGlooStagedEplbCommunicator
from vllm.distributed.eplb.eplb_state import (
    EplbModelState,
    EplbState,
    EplbStats,
    compute_logical_maps,
)
from vllm.distributed.eplb.eplb_utils import CpuGpuEvent
from vllm.distributed.eplb.policy import DefaultEplbPolicy
from vllm.distributed.eplb.policy_process import (
    EplbPolicyProcess,
    EplbPolicyProcessError,
)
from vllm.distributed.eplb.rebalance_execute import AsyncEplbLayerResult, transfer_layer
from vllm.distributed.parallel_state import get_ep_group, get_eplb_group
from vllm.platforms import current_platform
from vllm.platforms.interface import Platform, PlatformEnum


class _Platform(Platform):
    _enum = PlatformEnum.OOT

    @classmethod
    def supports_eplb(cls):
        return True

    @classmethod
    def check_and_update_eplb_config(cls, parallel_config):
        config = parallel_config.eplb_config
        if config.policy is None:
            config.policy = "custom"
        if config.policy != "custom" or config.communicator not in (
            None,
            "custom_comm",
        ):
            raise ValueError("unsupported custom EPLB configuration")
        config.communicator = "custom_comm"


@pytest.mark.parametrize(
    "policy, communicator", [(None, None), ("custom", None), ("custom", "custom_comm")]
)
def test_platform_can_validate_out_of_tree_names(monkeypatch, policy, communicator):
    monkeypatch.setattr(parallel_config_module, "current_platform", _Platform())
    config = ParallelConfig(
        enable_eplb=True,
        enable_expert_parallel=True,
        tensor_parallel_size=2,
        distributed_executor_backend="uni",
        eplb_config=EPLBConfig(policy=policy, communicator=communicator),
    )
    assert config.eplb_config.policy == "custom"
    assert config.eplb_config.communicator == "custom_comm"


@pytest.mark.parametrize("field_name", ["policy", "communicator"])
def test_builtin_platform_rejects_unknown_names(field_name):
    config = ParallelConfig(eplb_config=EPLBConfig(**{field_name: "unknown"}))
    with pytest.raises(ValueError, match=f"Unknown EPLB {field_name}"):
        Platform.check_and_update_eplb_config(config)


def test_platform_can_reject_eplb_even_on_cuda(monkeypatch):
    class UnsupportedCuda(Platform):
        _enum = PlatformEnum.CUDA

        @classmethod
        def supports_eplb(cls):
            return False

    monkeypatch.setattr(parallel_config_module, "current_platform", UnsupportedCuda())
    with pytest.raises(ValueError, match="not supported on this platform"):
        ParallelConfig(
            enable_eplb=True,
            tensor_parallel_size=2,
            distributed_executor_backend="uni",
        )


def test_xpu_preserves_eplb_and_xccl_auto_selection():
    class XpuPlatform(Platform):
        _enum = PlatformEnum.XPU

    config = ParallelConfig()
    assert XpuPlatform.supports_eplb()
    XpuPlatform.check_and_update_eplb_config(config)
    assert config.eplb_config.communicator == "torch_xccl"
    assert config.eplb_config.policy == "default"


@pytest.mark.parametrize(
    "communicator, use_async",
    [
        ("torch_gloo", True),
        ("torch_xccl", True),
        ("nixl", True),
        ("torch_nccl", False),
        ("pynccl", False),
    ],
)
def test_explicit_builtin_communicator_is_preserved(communicator, use_async):
    config = ParallelConfig(
        eplb_config=EPLBConfig(communicator=communicator, use_async=use_async)
    )
    Platform.check_and_update_eplb_config(config)
    assert config.eplb_config.communicator == communicator


@pytest.mark.parametrize("communicator", ["torch_nccl", "pynccl"])
def test_async_nccl_restrictions_remain(communicator):
    config = ParallelConfig(eplb_config=EPLBConfig(communicator=communicator))
    with pytest.raises(ValueError, match="incompatible with async EPLB"):
        Platform.check_and_update_eplb_config(config)


def test_llm_rejects_async_nccl_communicator(vllm_runner, monkeypatch, tmp_path):
    """Reject an invalid async communicator before loading model weights."""

    class BuiltinPlatform(Platform):
        _enum = PlatformEnum.CUDA

    monkeypatch.setattr(parallel_config_module, "current_platform", BuiltinPlatform())
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    ).save_pretrained(tmp_path)
    eplb_config = EPLBConfig(communicator="torch_nccl")
    with pytest.raises(ValueError, match="incompatible with async EPLB"):
        vllm_runner(
            str(tmp_path),
            max_model_len=32,
            skip_tokenizer_init=True,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            enable_eplb=True,
            distributed_executor_backend="uni",
            eplb_config=eplb_config,
        )


@pytest.mark.parametrize("interval", [0, -1])
def test_balancedness_interval_must_be_positive(interval):
    with pytest.raises(ValueError, match="log_balancedness_interval"):
        EPLBConfig(log_balancedness_interval=interval)


@pytest.fixture
def cpu_group(tmp_path):
    config = VllmConfig()
    config.parallel_config.enable_eplb = True
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{tmp_path}/store",
            backend="gloo",
        )
        initialize_model_parallel(backend="gloo")
        try:
            yield get_ep_group().cpu_group
        finally:
            cleanup_dist_env_and_memory()


class _ExpertModel(torch.nn.Module):
    """Small real expert weight owner for control-plane and transfer checks."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(16).reshape(4, 2, 2).float(), False
        )
        self.expert_weights = [[self.weight]]
        self.num_moe_layers = 1


class _ModelState(EplbModelState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_commit = False
        self.committed_layers = []
        self.committed_maps = []

    def _commit_eplb_maps_for_layer(self, target, layer):
        super()._commit_eplb_maps_for_layer(target, layer)
        assert self.pending_result is not None
        assert not self.pending_result.consumed_event._recorded.is_set()
        if self.fail_commit:
            raise RuntimeError("device commit failed")
        self.committed_layers.append(layer)
        self.committed_maps.append(self.physical_to_logical_map.clone())

    def _commit_eplb_maps(self, target):
        super()._commit_eplb_maps(target)
        self.committed_layers.extend(range(self.model.num_moe_layers))
        self.committed_maps.append(self.physical_to_logical_map.clone())


class _State(EplbState):
    model_state_cls = _ModelState

    def __init__(self, policy=None):
        self.selected_policy = policy or DefaultEplbPolicy()
        super().__init__(ParallelConfig(), torch.device("cpu"))

    def create_policy(self):
        return self.selected_policy


def _make_model_state(cpu_group, mapping):
    model = _ExpertModel()
    logical, replicas = compute_logical_maps(mapping, 4)
    loads = torch.zeros(1, 4, dtype=torch.int64)
    return _State.model_state_cls(
        physical_to_logical_map=mapping,
        physical_to_logical_map_buffer=mapping,
        logical_to_physical_map=logical,
        logical_replica_count=replicas,
        expert_load_pass=loads,
        expert_load_pass_buffer=loads,
        expert_load_window=torch.zeros(4, 1, 4, dtype=torch.int64),
        last_expert_load=torch.ones(1, 4, dtype=torch.int64),
        model_name="expert-test",
        model=model,
        expert_buffer=[torch.empty_like(model.weight)],
        rebalanced=False,
        eplb_stats=None,
        device_index=None,
        communicator=TorchDistGlooStagedEplbCommunicator(cpu_group),
    )


@pytest.fixture
def model_state(cpu_group):
    return _make_model_state(cpu_group, torch.arange(4).reshape(1, 4))


@pytest.mark.parametrize("use_async", [False, True])
def test_policy_receives_extended_stats_without_losing_inputs(model_state, use_async):
    """Custom inputs survive direct calls and async subprocess planning."""
    worker_pid = os.getpid()

    @dataclass
    class Stats(EplbStats):
        metadata: tuple[int, ...] = ()

    class Policy(DefaultEplbPolicy):
        def rebalance_experts(self, stats, old_global_expert_indices=None):
            assert (os.getpid() != worker_pid) is use_async
            assert isinstance(stats, Stats)
            assert stats.metadata == (3, 7)
            assert stats.global_expert_load_window.device.type == "cpu"
            torch.testing.assert_close(stats.global_expert_load_window, loads)
            return super().rebalance_experts(stats, old_global_expert_indices)

    loads = torch.tensor([[13, 3, 4, 3]])
    current = torch.tensor([[0, 1, 2, 3, 0, 1]])
    device = "cpu"
    if use_async and (current_platform.is_cuda_alike() or current_platform.is_xpu()):
        device = current_platform.device_type
    published_stats = Stats(loads.to(device), 6, 1, 1, 2, metadata=(3, 7))
    state = _State(Policy())
    if use_async:
        model_state.eplb_stats = published_stats
        with closing(EplbPolicyProcess(state.policy)) as state.policy_process:
            result = run_rebalance_experts(model_state, state, current, None)
    else:
        result = state.policy.rebalance_experts(published_stats, current)
    default_state = EplbState(ParallelConfig(), torch.device("cpu"))
    assert isinstance(default_state.policy, DefaultEplbPolicy)
    expected = default_state.policy.rebalance_experts(published_stats, current)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("fail_commit", [False, True])
def test_async_commit_updates_maps_before_buffer_acknowledgement(
    model_state, fail_commit
):
    model_state.fail_commit = fail_commit
    target = torch.tensor([[1, 0, 3, 2]])
    original = model_state.model.weight.detach().clone()
    metadata = transfer_layer(
        old_layer_indices=model_state.physical_to_logical_map[0],
        new_layer_indices=target[0],
        expert_weights=model_state.model.expert_weights[0],
        expert_weights_buffer=model_state.expert_buffer,
        communicator=model_state.communicator,
        ep_group=get_ep_group().device_group,
        stream=None,
        is_profile=False,
    )
    result = AsyncEplbLayerResult(0, target[0], metadata, CpuGpuEvent())
    model_state.pending_result = result
    model_state.rebalanced = True
    if fail_commit:
        with pytest.raises(RuntimeError, match="commit failed"):
            model_state._move_to_workspace(0)
    else:
        model_state._move_to_workspace(0)
    torch.testing.assert_close(model_state.model.weight, original[[1, 0, 3, 2]])
    torch.testing.assert_close(model_state.physical_to_logical_map, target)
    if fail_commit:
        assert model_state.committed_layers == []
        assert model_state.pending_result is result
        assert not result.consumed_event._recorded.is_set()
    else:
        assert model_state.committed_layers == [0]
        torch.testing.assert_close(model_state.committed_maps[0], target)
        assert model_state.pending_result is None
        assert result.consumed_event._recorded.is_set()


def test_sync_commit_updates_maps_before_device_work(model_state):
    target = torch.tensor([[1, 0, 3, 2]])
    model_state._commit_eplb_maps(target)
    assert model_state.committed_layers == [0]
    torch.testing.assert_close(model_state.committed_maps[0], target)
    torch.testing.assert_close(
        model_state.logical_to_physical_map[0, :, 0], torch.tensor([1, 0, 3, 2])
    )


class _GatedPolicy(DefaultEplbPolicy):
    def __init__(self, endpoint, fail):
        self.endpoint = endpoint
        self.fail = fail

    def rebalance_experts(self, stats, old_global_expert_indices=None):
        if self.endpoint is not None:
            with socket.create_connection(self.endpoint, timeout=20) as connection:
                connection.sendall(b"r")
                assert connection.recv(1) == b"g"
        if self.fail:
            raise ValueError("test planner failed")
        return super().rebalance_experts(stats, old_global_expert_indices)


def _run_coordinated_stop(rank, store, reason, entered_collective, reports):
    torch.set_num_threads(1)
    parallel = ParallelConfig(cpu_distributed_timeout_seconds=10)
    config = VllmConfig(parallel_config=parallel)
    # Create the real control-plane groups without requiring accelerator transfers.
    parallel.enable_eplb = True
    parallel.tensor_parallel_size = 2
    with set_current_vllm_config(config), socket.socket() as listener:
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"file://{store}",
            backend="gloo",
            timeout=timedelta(seconds=30),
        )
        initialize_model_parallel(tensor_model_parallel_size=2, backend="gloo")
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(20)
        policy = _GatedPolicy(
            listener.getsockname() if rank == 1 else None,
            fail=rank == 1 and reason == "planner_error",
        )
        state = _State(policy)
        state.is_async = True
        mapping = torch.arange(4).repeat(2).reshape(1, 8)
        model = _make_model_state(get_eplb_group().cpu_group, mapping)
        model.rebalanced = True
        model.eplb_stats = EplbStats(torch.tensor([[13, 3, 4, 3]]), 8, 1, 1, 2)
        state.model_states["test"] = model
        if reason == "reset":
            draft = _make_model_state(get_eplb_group().cpu_group, mapping)
            draft.rebalanced = True
            draft.eplb_stats = model.eplb_stats
            state.model_states["draft"] = draft
        original_mapping = model.physical_to_logical_map.clone()
        weights = model.model.weight.detach().clone()
        completed = threading.Event()
        errors = []

        def transfer():
            try:
                _transfer_cycle(state, None, False)
            except Exception as error:
                errors.append(error)
            finally:
                completed.set()

        try:
            with closing(EplbPolicyProcess(policy)) as state.policy_process:
                torch.distributed.barrier(group=get_eplb_group().cpu_group)
                worker = threading.Thread(target=transfer, daemon=True)
                worker.start()
                if rank == 0:
                    deadline = time.monotonic() + 20
                    while not entered_collective.is_set():
                        frame = sys._current_frames().get(worker.ident)
                        while frame is not None:
                            if frame.f_code.co_name == "all_reduce":
                                entered_collective.set()
                                break
                            frame = frame.f_back
                        if entered_collective.is_set():
                            break
                        assert not completed.wait(0.001)
                        assert time.monotonic() < deadline
                else:
                    with listener.accept()[0] as connection:
                        assert connection.recv(1) == b"r"
                        assert entered_collective.wait(20)
                        if reason == "reset":
                            state.ft_reset_epoch += 1
                            for cancelled_model in state.model_states.values():
                                cancelled_model.rebalanced = False
                            connection.sendall(b"g")
                        elif reason == "shutdown":
                            state.async_worker_stop.set()
                            state.policy_process.close()
                        else:
                            connection.sendall(b"g")
                assert completed.wait(20)
                worker.join(timeout=1)
                assert not worker.is_alive()
                torch.testing.assert_close(
                    model.physical_to_logical_map, original_mapping
                )
                torch.testing.assert_close(model.model.weight, weights)
                assert model.pending_result is None
                assert model.num_committed_layers == 0
                if reason == "planner_error":
                    assert len(errors) == 1, errors
                    assert isinstance(errors[0], EplbPolicyProcessError), errors
                    assert model.rebalanced
                    state.async_worker_error = errors[0]
                    state.expert_rearrangement_step_interval = 3000
                    with pytest.raises(RuntimeError, match="Async EPLB worker failed"):
                        state.step(is_dummy=True)
                else:
                    assert not errors, errors
                    assert not model.rebalanced
                    assert all(
                        not cancelled_model.rebalanced
                        for cancelled_model in state.model_states.values()
                    )
                reports.put(rank)
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.parametrize("reason", ["reset", "planner_error", "shutdown"])
def test_async_policy_cancellation_is_coordinated_across_ranks(
    tmp_path, monkeypatch, reason
):
    """A slow or failed planner still joins a peer already waiting in Gloo."""
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "cpu")
    context = multiprocessing.get_context("spawn")
    entered_collective = context.Event()
    reports = context.Queue()
    workers = [
        context.Process(
            target=_run_coordinated_stop,
            args=(rank, str(tmp_path / "store"), reason, entered_collective, reports),
        )
        for rank in range(2)
    ]
    for worker in workers:
        worker.start()
    try:
        for worker in workers:
            worker.join(timeout=60)
            assert worker.exitcode == 0
        assert sorted(reports.get(timeout=5) for _ in workers) == [0, 1]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)


@pytest.mark.parametrize("in_flight", [False, True])
def test_shutdown_cancels_inflight_and_idle_rebalance(cpu_group, in_flight):
    """Stopping an in-process worker cancels further transfers and reaps the child."""
    state = _State()
    state.is_async = True
    model = _make_model_state(cpu_group, torch.arange(4).reshape(1, 4))
    state.model_states["test"] = model
    process = EplbPolicyProcess(state.policy)
    state.policy_process = process
    worker = threading.Thread(target=transfer_run_periodically, args=(state, None))
    state.async_worker = worker
    worker.start()
    try:
        if in_flight:
            model.rebalanced = True
            model.eplb_stats = EplbStats(torch.tensor([[9, 1, 2, 1]]), 4, 1, 1, 1)
            state.rearrange_event.record()
            deadline = time.monotonic() + 10
            while model.pending_result is None:
                assert time.monotonic() < deadline
                time.sleep(0.005)
        state.stop_async_loop(timeout=5)
        state.stop_async_loop(timeout=5)
        assert not worker.is_alive()
        assert process._process.poll() is not None
        assert state.async_worker is None
        assert state.policy_process is None
        assert not model.rebalanced
        assert model.pending_result is None
    finally:
        state.async_worker_stop.set()
        process.close()
        worker.join(timeout=5)


def test_ft_reset_skips_subsequent_model_planning_and_retired_group(
    cpu_group, monkeypatch
):
    """An epoch bump on model 1 skips model 2 IPC and retired Gloo collectives."""
    calls: list[int] = []

    class Policy(DefaultEplbPolicy):
        def rebalance_experts(self, stats, old_global_expert_indices=None):
            calls.append(int(stats.global_expert_load_window.sum().item()))
            return super().rebalance_experts(stats, old_global_expert_indices)

    state = _State(Policy())
    state.is_async = True
    draft = _make_model_state(cpu_group, torch.arange(4).reshape(1, 4))
    target = _make_model_state(cpu_group, torch.arange(4).reshape(1, 4))
    draft.rebalanced = True
    target.rebalanced = True
    draft.eplb_stats = EplbStats(torch.tensor([[5, 1, 1, 1]]), 4, 1, 1, 1)
    target.eplb_stats = EplbStats(torch.tensor([[9, 2, 2, 2]]), 4, 1, 1, 1)
    state.model_states["draft"] = draft
    state.model_states["target"] = target

    original_transfer = draft.communicator.execute

    def reset_mid_transfer():
        original_transfer()
        state.ft_reset_epoch += 1
        draft.rebalanced = False
        target.rebalanced = False
        monkeypatch.setattr(get_eplb_group(), "cpu_group", object())

    draft.communicator.execute = reset_mid_transfer
    _transfer_cycle(state, None, False)
    assert calls == [8]
    assert draft.pending_result is None
    assert target.pending_result is None
