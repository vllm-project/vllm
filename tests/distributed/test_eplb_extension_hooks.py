# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from vllm.distributed.eplb.async_worker import run_rebalance_experts
from vllm.distributed.eplb.eplb_communicator import TorchDistGlooStagedEplbCommunicator
from vllm.distributed.eplb.eplb_state import (
    EplbModelState,
    EplbState,
    EplbStats,
    _move_to_workspace,
    compute_logical_maps,
)
from vllm.distributed.eplb.eplb_utils import CpuGpuEvent
from vllm.distributed.eplb.policy import DefaultEplbPolicy
from vllm.distributed.eplb.rebalance_execute import AsyncEplbLayerResult, transfer_layer
from vllm.distributed.parallel_state import get_ep_group
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
    with set_current_vllm_config(VllmConfig()):
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


class _State(EplbState):
    def __init__(self, policy=None, fail_commit=False):
        self.selected_policy = policy or DefaultEplbPolicy()
        self.fail_commit = fail_commit
        self.committed_layers = []
        self.committed_maps = []
        super().__init__(ParallelConfig(), torch.device("cpu"))

    def create_policy(self):
        return self.selected_policy

    def _commit_rebalance(self, model_state, target, layer_idx=None):
        super()._commit_rebalance(model_state, target, layer_idx)
        if model_state.pending_result is not None:
            assert not model_state.pending_result.consumed_event._recorded.is_set()
        if self.fail_commit:
            raise RuntimeError("device commit failed")
        self.committed_layers.extend(
            range(model_state.model.num_moe_layers)
            if layer_idx is None
            else [layer_idx]
        )
        self.committed_maps.append(model_state.physical_to_logical_map.clone())


@pytest.fixture
def model_state(cpu_group):
    model = _ExpertModel()
    mapping = torch.arange(4).reshape(1, 4)
    logical, replicas = compute_logical_maps(mapping, 4)
    loads = torch.zeros(1, 4, dtype=torch.int64)
    return EplbModelState(
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


@pytest.mark.parametrize("use_async", [False, True])
def test_planning_hook_preserves_existing_policy_result(model_state, use_async):
    loads = torch.tensor([[13, 3, 4, 3]])
    current = torch.tensor([[0, 1, 2, 3, 0, 1]])
    observed = []

    class State(_State):
        def rebalance_experts(self, model, weight, *args):
            observed.append(model)
            assert weight.ndim == 2
            return super().rebalance_experts(model, weight, *args)

    state = State()
    if use_async:
        model_state.eplb_stats = EplbStats(loads, 6, 1, 1, 2)
        result = run_rebalance_experts(model_state, state, current, None)
    else:
        result = state.rebalance_experts(model_state, loads, 6, 1, 1, 2, current)
    expected = DefaultEplbPolicy.rebalance_experts(loads, 6, 1, 1, 2, current)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert len(observed) == 1 and observed[0] is model_state


@pytest.mark.parametrize("fail_commit", [False, True])
def test_async_commit_updates_maps_before_buffer_acknowledgement(
    model_state, fail_commit
):
    state = _State(fail_commit=fail_commit)
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
            _move_to_workspace(model_state, 0, state._commit_rebalance)
    else:
        _move_to_workspace(model_state, 0, state._commit_rebalance)
    torch.testing.assert_close(model_state.model.weight, original[[1, 0, 3, 2]])
    torch.testing.assert_close(model_state.physical_to_logical_map, target)
    if fail_commit:
        assert state.committed_layers == []
        assert model_state.pending_result is result
        assert not result.consumed_event._recorded.is_set()
    else:
        assert state.committed_layers == [0]
        torch.testing.assert_close(state.committed_maps[0], target)
        assert model_state.pending_result is None
        assert result.consumed_event._recorded.is_set()


def test_sync_commit_updates_maps_before_device_work(model_state):
    state = _State()
    target = torch.tensor([[1, 0, 3, 2]])
    state._commit_rebalance(model_state, target)
    assert state.committed_layers == [0]
    torch.testing.assert_close(state.committed_maps[0], target)
    torch.testing.assert_close(
        model_state.logical_to_physical_map[0, :, 0], torch.tensor([1, 0, 3, 2])
    )
