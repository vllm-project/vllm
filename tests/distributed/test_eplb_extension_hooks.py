# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

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
    EplbLayerState,
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


@pytest.mark.parametrize("log_balancedness", [False, True])
@pytest.mark.parametrize("interval", [0, -1])
def test_balancedness_interval_must_be_positive(log_balancedness, interval):
    with pytest.raises(ValueError, match="log_balancedness_interval"):
        EPLBConfig(
            log_balancedness=log_balancedness,
            log_balancedness_interval=interval,
        )


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


class _State(EplbState):
    def __init__(self, policy=None, fail_commit=False):
        self.selected_policy = policy or DefaultEplbPolicy()
        self.fail_commit = fail_commit
        self.committed_layers = []
        self.committed_maps = []
        super().__init__(ParallelConfig(), torch.device("cpu"))

    def create_policy(self):
        return self.selected_policy

    def on_layer_committed(self, model_state, layer_idx):
        if model_state.pending_result is not None:
            assert not model_state.pending_result.consumed_event.wait(timeout=0)
        if self.fail_commit:
            raise RuntimeError("device commit failed")
        self.committed_layers.append(layer_idx)
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_policy_target_requires_cpu_integer(model_state, dtype):
    target = torch.zeros(1, 4, dtype=dtype)

    class Policy(DefaultEplbPolicy):
        @classmethod
        def rebalance_experts(cls, *args, **kwargs):
            return target

    state = _State(Policy())
    if dtype == torch.float32:
        with pytest.raises(ValueError, match="CPU int32 or int64"):
            state.rebalance_experts(model_state, torch.ones(1, 4), 4, 1, 1, 1)
    else:
        assert (
            state.rebalance_experts(model_state, torch.ones(1, 4), 4, 1, 1, 1) is target
        )


def test_aggregate_allreduce_keeps_original_payload_shape(model_state):
    state = _State()
    first = torch.arange(8).reshape(2, 4)
    second = torch.arange(12).reshape(3, 4)
    reduced = state._allreduce_list([first, second])
    assert [tensor.shape for tensor in reduced] == [(2, 4), (3, 4)]
    torch.testing.assert_close(reduced[0], first, rtol=0, atol=0)
    torch.testing.assert_close(reduced[1], second, rtol=0, atol=0)


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
        assert not result.consumed_event.wait(timeout=0)
    else:
        assert state.committed_layers == [0]
        torch.testing.assert_close(state.committed_maps[0], target)
        assert model_state.pending_result is None
        assert result.consumed_event.wait(timeout=0)


def test_sync_commit_updates_maps_before_callback(model_state):
    state = _State()
    target = torch.tensor([[1, 0, 3, 2]])
    state._commit_rebalance(model_state, target)
    assert state.committed_layers == [0]
    torch.testing.assert_close(state.committed_maps[0], target)
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
