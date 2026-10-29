# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from dataclasses import dataclass, field

import pytest
import torch

import vllm.config.parallel as parallel_config_module
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.config.parallel import EPLBConfig
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.distributed.eplb.eplb_communicator import TorchDistGlooStagedEplbCommunicator
from vllm.distributed.eplb.eplb_state import (
    EplbLayerState,
    EplbModelState,
    EplbState,
    _move_to_workspace,
    compute_logical_maps,
)
from vllm.distributed.eplb.eplb_utils import CpuGpuEvent
from vllm.distributed.eplb.policy import (
    DefaultEplbPolicy,
    EplbPlan,
    EplbPolicyState,
    EplbRebalanceContext,
    EplbTopology,
)
from vllm.distributed.eplb.rebalance_execute import AsyncEplbLayerResult, transfer_layer
from vllm.distributed.parallel_state import get_ep_group
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.platforms.interface import Platform, PlatformEnum


class _Platform(Platform):
    _enum = PlatformEnum.OOT

    @classmethod
    def supports_eplb(cls):
        return True

    @classmethod
    def check_and_update_eplb_config(cls, parallel_config):
        config = parallel_config.eplb_config
        if config.policy != "custom" or config.communicator not in (
            None,
            "custom_comm",
        ):
            raise ValueError("unsupported custom EPLB configuration")
        config.communicator = "custom_comm"


@pytest.mark.parametrize("communicator", [None, "custom_comm"])
def test_platform_can_validate_out_of_tree_names(monkeypatch, communicator):
    monkeypatch.setattr(parallel_config_module, "current_platform", _Platform())
    config = ParallelConfig(
        enable_eplb=True,
        enable_expert_parallel=True,
        tensor_parallel_size=2,
        distributed_executor_backend="uni",
        eplb_config=EPLBConfig(policy="custom", communicator=communicator),
    )
    assert config.eplb_config.policy == "custom"
    assert config.eplb_config.communicator == "custom_comm"


@pytest.mark.parametrize("field_name", ["policy", "communicator"])
def test_builtin_platform_rejects_unknown_names(field_name):
    config = ParallelConfig(eplb_config=EPLBConfig(**{field_name: "unknown"}))
    with pytest.raises(ValueError, match=f"Unknown EPLB {field_name}"):
        Platform.check_and_update_eplb_config(config)


@pytest.mark.parametrize(
    "platform_enum", [PlatformEnum.CUDA, PlatformEnum.ROCM, PlatformEnum.XPU]
)
def test_builtin_accelerator_eplb_capability(platform_enum):
    class BuiltinPlatform(Platform):
        _enum = platform_enum

    assert BuiltinPlatform.supports_eplb()


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


@pytest.mark.parametrize("communicator", ["torch_gloo", "torch_xccl", "nixl"])
def test_explicit_builtin_communicator_is_preserved(communicator):
    config = ParallelConfig(eplb_config=EPLBConfig(communicator=communicator))
    Platform.check_and_update_eplb_config(config)
    assert config.eplb_config.communicator == communicator


@pytest.mark.parametrize("communicator", ["torch_nccl", "pynccl"])
def test_async_nccl_restrictions_remain(communicator):
    with pytest.raises(ValueError, match="incompatible with async EPLB"):
        EPLBConfig(communicator=communicator)


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

    def forward(self, x, expert_id):
        return x @ self.weight[expert_id]


@dataclass
class _PolicyState(EplbPolicyState):
    committed_layers: list[int] = field(default_factory=list)


class _Policy(DefaultEplbPolicy):
    def __init__(self, target=None, fail_commit=False):
        self.target = target
        self.fail_commit = fail_commit

    def create_state(self, num_moe_layers):
        return _PolicyState()

    def rebalance_experts(self, context, policy_state):
        if self.target is not None:
            return EplbPlan(self.target)
        return super().rebalance_experts(context, policy_state)

    def on_layer_committed(self, policy_state, plan, layer_idx):
        if self.fail_commit:
            raise RuntimeError("policy commit failed")
        policy_state.committed_layers.append(layer_idx)


class _State(EplbState):
    def __init__(self, policy=None, fail_commit=False):
        self.selected_policy = policy or _Policy()
        self.fail_commit = fail_commit
        self.committed_layers = []
        super().__init__(ParallelConfig(), torch.device("cpu"))

    def create_policy(self):
        return self.selected_policy

    def on_layer_committed(self, model_state, layer_idx):
        if self.fail_commit:
            raise RuntimeError("device commit failed")
        self.committed_layers.append(layer_idx)


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
        policy_state=_PolicyState(),
    )


def test_default_policy_preserves_aggregate_result(cpu_group):
    context = EplbRebalanceContext(
        load_window_cpu=torch.tensor([[[9, 1, 3, 2]], [[4, 2, 1, 1]]]),
        physical_to_logical_map_cpu=torch.tensor([[0, 1, 2, 3, 0, 1]]),
        topology=EplbTopology(num_groups=1, num_nodes=1, num_ranks=2),
        num_replicas=6,
        cpu_group=cpu_group,
    )
    policy = DefaultEplbPolicy()
    result = policy.rebalance_experts(context, policy.create_state(1))
    expected = policy._rebalance_from_weights(
        torch.tensor([[13, 3, 4, 3]]),
        6,
        1,
        1,
        2,
        context.physical_to_logical_map_cpu,
    )
    torch.testing.assert_close(result.physical_to_logical_map, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_policy_target_requires_cpu_integer(model_state, cpu_group, dtype):
    target = torch.zeros(1, 4, dtype=dtype)
    state = _State(_Policy(target))
    context = EplbRebalanceContext(
        torch.ones(1, 1, 4),
        torch.arange(4).reshape(1, 4),
        EplbTopology(1, 1, 1),
        4,
        cpu_group,
    )
    if dtype == torch.float32:
        with pytest.raises(ValueError, match="CPU int32 or int64"):
            state.rebalance_experts(model_state, context)
    else:
        assert (
            state.rebalance_experts(model_state, context).physical_to_logical_map
            is target
        )


def test_shared_policy_keeps_model_state_independent(model_state, cpu_group):
    policy = _Policy()
    first, second = policy.create_state(1), policy.create_state(1)
    plan = EplbPlan(torch.arange(4).reshape(1, 4))
    policy.on_layer_committed(first, plan, 0)
    assert first.committed_layers == [0]
    assert second.committed_layers == []
    state = _State(policy)
    assert state.policy is policy


def test_aggregate_allreduce_keeps_original_payload_shape(model_state):
    state = _State()
    first = torch.arange(8).reshape(2, 4)
    second = torch.arange(12).reshape(3, 4)
    reduced = state._allreduce_list([first, second])
    assert [tensor.shape for tensor in reduced] == [(2, 4), (3, 4)]
    torch.testing.assert_close(reduced[0], first, rtol=0, atol=0)
    torch.testing.assert_close(reduced[1], second, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["device", "policy"])
def test_commit_failure_keeps_buffer_unacknowledged(model_state, failure):
    state = _State(_Policy(fail_commit=failure == "policy"), failure == "device")
    plan = EplbPlan(torch.tensor([[1, 0, 3, 2]]))
    original = model_state.model.weight.detach().clone()
    metadata = transfer_layer(
        old_layer_indices=model_state.physical_to_logical_map[0],
        new_layer_indices=plan.physical_to_logical_map[0],
        expert_weights=model_state.model.expert_weights[0],
        expert_weights_buffer=model_state.expert_buffer,
        communicator=model_state.communicator,
        ep_group=get_ep_group().device_group,
        stream=None,
        is_profile=False,
    )
    result = AsyncEplbLayerResult(0, plan, metadata, CpuGpuEvent())
    model_state.pending_result = result
    model_state.rebalanced = True
    with pytest.raises(RuntimeError, match="commit failed"):
        _move_to_workspace(model_state, 0, state._commit_rebalance)
    torch.testing.assert_close(model_state.model.weight, original[[1, 0, 3, 2]])
    torch.testing.assert_close(
        model_state.physical_to_logical_map, plan.physical_to_logical_map
    )
    assert model_state.policy_state.committed_layers == []
    assert model_state.pending_result is result
    assert not result.consumed_event.wait(timeout=0)


def test_sync_commit_updates_maps_then_policy_state(model_state):
    state = _State()
    plan = EplbPlan(torch.tensor([[1, 0, 3, 2]]))
    state._commit_rebalance(model_state, plan)
    assert state.committed_layers == [0]
    assert model_state.policy_state.committed_layers == [0]
    torch.testing.assert_close(
        model_state.logical_to_physical_map[0, :, 0], torch.tensor([1, 0, 3, 2])
    )


def test_router_calls_layer_mapping_with_live_padding_metadata():
    observed = []

    def map_and_record(ids, num_tokens):
        observed.append(num_tokens)
        return ids + 1

    layer = EplbLayerState(
        expert_load_view=torch.zeros(2),
        logical_to_physical_map=torch.arange(2).unsqueeze(-1),
        logical_replica_count=torch.ones(2),
        should_record_tensor=torch.ones((), dtype=torch.bool),
        num_unpadded_tokens_tensors=[torch.tensor(1)],
        map_and_record=map_and_record,
    )
    router = FusedTopKRouter(top_k=1, global_num_experts=2, eplb_state=layer)
    torch.testing.assert_close(
        router._apply_eplb_mapping(torch.tensor([[0]])), torch.tensor([[1]])
    )
    assert observed == layer.num_unpadded_tokens_tensors


def test_idle_wait_can_stop_without_fabricating_device_event():
    state = _State()
    entered = threading.Event()
    stopped = threading.Event()

    def wait():
        entered.set()
        assert not state.wait_for_rearrangement(None)
        stopped.set()

    state.async_worker = threading.Thread(target=wait, daemon=True)
    state.async_worker.start()
    assert entered.wait(timeout=5)
    state.stop_async_loop(timeout=5)
    assert stopped.is_set()
    assert state.async_worker is None
    assert not state.rearrange_event.wait(timeout=0)
    state.stop_async_loop()


def test_stop_rejects_unconsumed_transfers(model_state):
    state = _State()
    entered = threading.Event()

    def wait():
        entered.set()
        state.wait_for_rearrangement(None)

    state.async_worker = threading.Thread(target=wait, daemon=True)
    state.model_states["model"] = model_state
    model_state.rebalanced = True
    state.async_worker.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(RuntimeError, match="Drain outstanding"):
            state.stop_async_loop()
        assert state.async_worker.is_alive()
    finally:
        model_state.rebalanced = False
        state.stop_async_loop(timeout=5)


def test_stop_timeout_keeps_worker_until_it_exits():
    state = _State()
    entered = threading.Event()
    release = threading.Event()

    def blocked_worker():
        entered.set()
        release.wait()

    worker = threading.Thread(target=blocked_worker, daemon=True)
    state.async_worker = worker
    worker.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(TimeoutError, match="did not stop"):
            state.stop_async_loop(timeout=0)
        assert state.async_worker is worker
    finally:
        release.set()
        state.stop_async_loop(timeout=5)
    assert state.async_worker is None
