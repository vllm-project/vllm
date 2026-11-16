# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass

import pytest
import torch
import torch.multiprocessing as mp
from transformers import LlamaConfig

import vllm.config.parallel as parallel_config_module
from vllm.config import ModelConfig, ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.config.parallel import EPLBConfig
from vllm.config.utils import hash_config_cached
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.distributed.eplb.async_worker import run_rebalance_experts
from vllm.distributed.eplb.eplb_communicator import (
    TorchDistGlooStagedEplbCommunicator,
    TorchDistNcclEplbCommunicator,
)
from vllm.distributed.eplb.eplb_state import (
    EplbModelState,
    EplbState,
    EplbStats,
    compute_logical_maps,
)
from vllm.distributed.eplb.eplb_utils import CpuGpuEvent
from vllm.distributed.eplb.placement import write_placement_snapshot
from vllm.distributed.eplb.policy import DefaultEplbPolicy
from vllm.distributed.eplb.rebalance_execute import (
    AsyncEplbLayerResult,
    rearrange_expert_weights_inplace,
    transfer_layer,
)
from vllm.distributed.parallel_state import get_ep_group
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

    def __init__(self, num_local_experts=4, device="cpu", expert_offset=0):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(
                expert_offset * 4,
                (expert_offset + num_local_experts) * 4,
                device=device,
            )
            .reshape(num_local_experts, 2, 2)
            .float(),
            False,
        )
        self.expert_weights = [[self.weight]]
        self.num_moe_layers = 1
        self.num_logical_experts = 4
        self.num_redundant_experts = 0


class _ModelState(EplbModelState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_commit = False
        self.committed_layers = []
        self.committed_maps = []

    def _commit_eplb_maps_for_layer(self, target, layer):
        super()._commit_eplb_maps_for_layer(target, layer)
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


def _create_model_state(cpu_group, device="cpu"):
    local_experts = 4 // cpu_group.size()
    model = _ExpertModel(local_experts, device, cpu_group.rank() * local_experts)
    mapping = torch.arange(4, device=device).reshape(1, 4)
    logical, replicas = compute_logical_maps(mapping, 4)
    loads = torch.zeros(1, 4, dtype=torch.int64, device=device)
    return _State.model_state_cls(
        physical_to_logical_map=mapping,
        physical_to_logical_map_buffer=mapping,
        logical_to_physical_map=logical,
        logical_replica_count=replicas,
        expert_load_pass=loads,
        expert_load_pass_buffer=loads,
        expert_load_window=torch.zeros(4, 1, 4, dtype=torch.int64, device=device),
        last_expert_load=torch.ones(1, 4, dtype=torch.int64, device=device),
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
    return _create_model_state(cpu_group)


def test_placement_export_waits_for_all_models(model_state, cpu_group, tmp_path):
    """A partial target/draft cycle must not replace the complete snapshot."""
    path = tmp_path / "placement.json"
    write_placement_snapshot(str(path), {"previous": True})
    state = _State()
    state.enable_placement_export(str(path))
    draft = _create_model_state(cpu_group)
    state.model_states.update(target=model_state, draft=draft)
    model_state.rebalanced = draft.rebalanced = True
    target_map = torch.tensor([[1, 0, 3, 2]])
    model_state._commit_eplb_maps(target_map)
    model_state.rebalanced = False

    state.export_committed_placement()

    assert json.loads(path.read_text()) == {"previous": True}
    draft_map = torch.tensor([[3, 2, 1, 0]])
    draft._commit_eplb_maps(draft_map)
    draft.rebalanced = False
    state.export_committed_placement()
    snapshot = json.loads(path.read_text())
    assert snapshot["version"] == 1
    assert snapshot["ep_size"] == 1
    assert set(snapshot["models"]) == {"target", "draft"}
    assert (
        snapshot["models"]["target"]["physical_to_logical_map"] == target_map.tolist()
    )
    assert snapshot["models"]["draft"]["physical_to_logical_map"] == draft_map.tolist()
    assert snapshot["models"]["target"]["num_logical_experts"] == 4


def test_placement_snapshot_precedes_later_map_commits(cpu_group, tmp_path):
    """The queued host snapshot retains the layout at its commit boundary."""
    device = (
        current_platform.device_type
        if current_platform.is_cuda_alike() or current_platform.is_xpu()
        else "cpu"
    )
    model = _create_model_state(cpu_group, device)
    state = EplbState(ParallelConfig(), torch.device(device))
    path = tmp_path / "placement.json"
    state.enable_placement_export(str(path))
    state.model_states["target"] = model
    model.model_id = "checkpoint"
    model.model_revision = "revision"
    first = torch.tensor([[1, 0, 3, 2]])
    model._commit_eplb_maps(first)

    state.export_committed_placement()
    model._commit_eplb_maps(torch.tensor([[3, 2, 1, 0]]))
    state.stop_async_loop()

    exported = json.loads(path.read_text())["models"]["target"]
    assert exported["physical_to_logical_map"] == first.tolist()
    assert exported["model"] == "checkpoint"
    assert exported["revision"] == "revision"


def test_placement_export_rejects_late_registration(model_state, tmp_path):
    state = _State()
    state.enable_placement_export(str(tmp_path / "placement.json"))
    state.model_states["target"] = model_state
    state.export_committed_placement()
    config_path = tmp_path / "model-config"
    LlamaConfig(architectures=["LlamaForCausalLM"]).save_pretrained(config_path)
    model_config = ModelConfig(model=str(config_path), skip_tokenizer_init=True)

    with pytest.raises(RuntimeError, match="after placement export starts"):
        state.add_model(model_state.model, model_config)


def test_placement_export_rejects_duplicate_model_hash(model_state, tmp_path):
    LlamaConfig(architectures=["LlamaForCausalLM"]).save_pretrained(tmp_path)
    model_config = ModelConfig(model=str(tmp_path), skip_tokenizer_init=True)
    state = _State()
    state.enable_placement_export(str(tmp_path / "placement.json"))
    state.model_states[hash_config_cached(model_config)] = model_state

    with pytest.raises(ValueError, match="distinct model configuration hashes"):
        state.add_model(model_state.model, model_config)


def test_atomic_placement_write_preserves_previous_snapshot(tmp_path):
    path = tmp_path / "nested" / "placement.json"
    write_placement_snapshot(str(path), {"version": 1, "placement": [0, 1]})
    previous = path.read_bytes()

    with pytest.raises(TypeError):
        write_placement_snapshot(str(path), {"unserializable": {1, 2}})

    assert path.read_bytes() == previous
    assert list(path.parent.iterdir()) == [path]


def test_placement_write_failure_preserves_snapshot_without_retrying(
    model_state, tmp_path, monkeypatch
):
    state = _State()
    path = tmp_path / "exports" / "placement.json"
    state.enable_placement_export(str(path))
    state.model_states["target"] = model_state
    write_placement_snapshot(str(path), {"previous": True})

    replace_calls = 0

    def replace_fails(self, target):
        nonlocal replace_calls
        replace_calls += 1
        raise OSError("filesystem unavailable")

    monkeypatch.setattr(type(path), "replace", replace_fails)
    state.export_committed_placement()
    state.stop_async_loop()

    assert json.loads(path.read_text()) == {"previous": True}
    assert replace_calls == 1
    assert list(path.parent.iterdir()) == [path]


def test_placement_export_requires_eplb(vllm_runner, tmp_path):
    """The serving entry point rejects export when EPLB is disabled."""
    LlamaConfig(architectures=["LlamaForCausalLM"]).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="requires enable_eplb"):
        vllm_runner(
            str(tmp_path),
            skip_tokenizer_init=True,
            eplb_config=EPLBConfig(placement_export_path=str(tmp_path / "map.json")),
        )


@pytest.mark.parametrize("path", ["", " "])
def test_placement_export_rejects_empty_path(path):
    with pytest.raises(ValueError, match="nonempty file path"):
        EPLBConfig(placement_export_path=path)


@pytest.mark.parametrize(
    "parallel_kwargs", [{"pipeline_parallel_size": 2}, {"enable_fault_tolerance": True}]
)
def test_placement_export_rejects_unsupported_lifecycles(parallel_kwargs, tmp_path):
    config = ParallelConfig(distributed_executor_backend="uni", **parallel_kwargs)
    state = EplbState(config, torch.device("cpu"))
    with pytest.raises(ValueError, match="fault tolerance disabled"):
        state.enable_placement_export(str(tmp_path / "placement.json"))


def _placement_export_worker(rank, directory, tp_size, pcp_size):
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"file://{directory}/distributed-store",
            backend="gloo",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            prefill_context_model_parallel_size=pcp_size,
            backend="gloo",
        )
        try:
            state = _State()
            path = directory / f"rank-{rank}.json"
            if tp_size * pcp_size != 2:
                with pytest.raises(ValueError, match="one EP group"):
                    state.enable_placement_export(str(path))
                assert not path.exists()
                return
            state.enable_placement_export(str(path))
            model = _create_model_state(get_ep_group().cpu_group)
            model.communicator = TorchDistNcclEplbCommunicator(get_ep_group().cpu_group)
            state.model_states["target"] = model
            target = torch.tensor([[2, 1, 0, 3]])
            rearrange_expert_weights_inplace(
                model.physical_to_logical_map,
                target,
                model.model.expert_weights,
                model.expert_buffer,
                get_ep_group().device_group,
                model.communicator,
            )
            model._commit_eplb_maps(target)
            expected_weights = torch.arange(16).reshape(4, 2, 2).float()
            torch.testing.assert_close(
                model.model.weight,
                expected_weights[target[0, rank * 2 : rank * 2 + 2]],
                rtol=0,
                atol=0,
            )
            state.export_committed_placement()
            state.stop_async_loop()
            if rank == 0:
                exported = json.loads(path.read_text())
                assert exported["ep_size"] == 2
                assert exported["models"]["target"]["physical_to_logical_map"] == (
                    target.tolist()
                )
            else:
                assert not path.exists()
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.cpu_test
@pytest.mark.parametrize("tp_size,pcp_size", [(2, 1), (1, 2), (1, 1)])
def test_placement_export_uses_one_global_writer(tmp_path, tp_size, pcp_size):
    """Cover TP/PCP EP groups and reject multiple independent EP groups."""
    mp.spawn(
        _placement_export_worker,
        args=(tmp_path, tp_size, pcp_size),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("use_async", [False, True])
def test_policy_receives_extended_stats_without_losing_inputs(model_state, use_async):
    @dataclass
    class Stats(EplbStats):
        metadata: tuple[int, ...] = ()

    class Policy(DefaultEplbPolicy):
        def rebalance_experts(self, stats, old_global_expert_indices=None):
            assert stats is published_stats
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
