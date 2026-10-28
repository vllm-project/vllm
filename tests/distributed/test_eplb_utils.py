# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.config.parallel as parallel_config_module
import vllm.distributed.eplb.eplb_state as eplb_state_module
from vllm.config.parallel import EPLBConfig, ParallelConfig
from vllm.distributed.eplb.eplb_state import (
    EplbState,
    _commit_eplb_maps,
    _commit_eplb_maps_for_layer,
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.eplb_utils import draft_model_supports_eplb


@pytest.mark.parametrize(
    ("model_type", "is_rocm", "has_moe", "expected"),
    [
        ("deepseek_v4", False, True, True),
        ("deepseek_v41", False, True, True),
        ("deepseek_v41", True, True, False),
        ("deepseek_v41", False, False, False),
    ],
)
def test_draft_model_supports_dspark_eplb(
    monkeypatch, model_type, is_rocm, has_moe, expected
):
    speculative_config = SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type)
        ),
    )
    monkeypatch.setattr(current_platform, "is_rocm", lambda: is_rocm)

    assert (
        draft_model_supports_eplb(
            speculative_config,
            object() if has_moe else None,
        )
        is expected
    )


def _make_model_state(
    phy2log: torch.Tensor,
    log2phy: torch.Tensor,
    logcnt: torch.Tensor,
    phy2log_storage: torch.Tensor | None = None,
) -> MagicMock:
    """Build a minimal EplbModelState mock with its map tensors."""
    state = MagicMock()
    state.physical_to_logical_map = phy2log
    state.physical_to_logical_map_buffer = (
        phy2log if phy2log_storage is None else phy2log_storage
    )
    state.logical_to_physical_map = log2phy
    state.logical_replica_count = logcnt
    return state


def test_commit_eplb_maps_shape_change():
    """When the number of physical experts changes, resize the active map within its
    preallocated storage.
    """
    num_layers, num_logical, num_physical = 2, 4, 6
    max_replicas = 3

    # Build current state tensors
    storage = torch.full((num_layers, num_physical + 2), -1, dtype=torch.long)
    model_state = _make_model_state(
        phy2log=storage[:, :num_physical],
        phy2log_storage=storage,
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    # The new map has two more physical experts. These new physical experts will
    # automatically map to the first two logical experts
    new_phy2log_larger = (
        (torch.arange(num_physical + 2, dtype=torch.long) % num_logical)
        .unsqueeze(0)
        .expand(num_layers, -1)
    )
    _commit_eplb_maps(model_state, new_phy2log_larger)

    # Check that the number of physical experts has been updated and that the values
    # match
    assert model_state.physical_to_logical_map.shape[1] == num_physical + 2
    assert model_state.physical_to_logical_map.data_ptr() == storage.data_ptr()
    assert torch.equal(model_state.physical_to_logical_map, new_phy2log_larger)


def test_commit_eplb_maps_for_layer_logical_padding():
    """Test that logical_to_physical_map is padded with -1 to fill the
    pre-allocated slots when the new map has fewer replicas than the max.
    """
    num_layers, num_logical, num_physical = 2, 4, 6
    max_replicas = 3

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = (
        (torch.arange(num_physical, dtype=torch.long) % num_logical)
        .unsqueeze(0)
        .expand(num_layers, -1)
        .contiguous()
    )
    layer = 0
    _commit_eplb_maps_for_layer(model_state, new_phy2log[layer], layer)

    assert torch.all(model_state.logical_to_physical_map[layer, :, 2] == -1)


def test_commit_eplb_maps_for_layer_shape_assert():
    """Test that a mismatched number of physical experts triggers an assertion error."""
    num_layers, num_logical, num_physical = 2, 4, 6

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full((num_layers, num_logical, 2), -1, dtype=torch.long),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )
    bad_phy2log = torch.zeros(num_layers, num_physical + 1, dtype=torch.long)
    with pytest.raises(AssertionError):
        _commit_eplb_maps_for_layer(model_state, bad_phy2log, layer=0)


def test_commit_eplb_maps():
    """Test that all values are copied correctly into model_state."""
    num_layers, num_logical, num_physical, max_replicas = 2, 3, 4, 2

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)
    new_log2phy = torch.tensor(
        [[[0, 3], [1, -1], [2, -1]], [[2, -1], [0, 3], [1, -1]]], dtype=torch.long
    )
    new_logcnt = torch.tensor([[2, 1, 1], [1, 2, 1]], dtype=torch.long)

    _commit_eplb_maps(model_state, new_phy2log)

    assert torch.equal(model_state.physical_to_logical_map, new_phy2log)
    assert torch.equal(model_state.logical_to_physical_map, new_log2phy)
    assert torch.equal(model_state.logical_replica_count, new_logcnt)


def test_commit_eplb_maps_for_layer():
    """Test that only the target layer is updated"""
    num_layers, num_logical, max_replicas = 2, 3, 2

    original_phy2log = torch.tensor([[9, 9, 9, 9], [8, 8, 8, 8]], dtype=torch.long)
    model_state = _make_model_state(
        phy2log=original_phy2log.clone(),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)
    new_log2phy = torch.tensor(
        [[[0, 3], [1, -1], [2, -1]], [[2, -1], [0, 3], [1, -1]]], dtype=torch.long
    )
    new_logcnt = torch.tensor([[2, 1, 1], [1, 2, 1]], dtype=torch.long)

    _commit_eplb_maps_for_layer(model_state, new_phy2log[0], layer=0)

    # Layer 0 updated
    assert torch.equal(model_state.physical_to_logical_map[0], new_phy2log[0])
    assert torch.equal(model_state.logical_to_physical_map[0], new_log2phy[0])
    assert torch.equal(model_state.logical_replica_count[0], new_logcnt[0])

    # Layer 1 untouched
    assert torch.equal(model_state.physical_to_logical_map[1], original_phy2log[1])


def _make_eplb_state() -> EplbState:
    parallel_config = MagicMock()
    parallel_config.enable_eplb = True
    parallel_config.eplb_config = MagicMock(
        use_async=False,
        window_size=4,
        step_interval=8,
        num_redundant_experts=0,
        policy="default",
        communicator="torch_gloo",
    )
    parallel_config.enable_elastic_ep = False
    parallel_config.num_ubatches = 1
    return EplbState(parallel_config, torch.device("cpu"))


class _EplbTestModel:
    def __init__(self, num_experts: int, num_redundant: int):
        self.num_routed_experts = num_experts
        self.num_logical_experts = num_experts
        self.num_redundant_experts = num_redundant
        self.num_physical_experts = num_experts + num_redundant
        self.num_local_physical_experts = self.num_physical_experts // 2
        self.num_expert_groups = num_experts
        self.num_moe_layers = 1
        self.num_shared_experts = 0
        self.moe_layers = ()
        self.expert_weights = [[torch.zeros(self.num_local_physical_experts, 1)]]

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        pass


class _EplbTestModelConfig:
    def __init__(self, name: str):
        self.model = name

    def compute_hash(self) -> str:
        return self.model


class _TopologyEncodingPolicy:
    @classmethod
    def rebalance_experts(
        cls,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mapping = (torch.arange(num_replicas, dtype=torch.long) + 1).remainder(
            num_groups
        )
        return mapping.unsqueeze(0).expand(weight.shape[0], -1).clone()


class _EplbTestGroup:
    def __init__(self):
        self.device_group = self

    def rank(self) -> int:
        return 1

    def size(self) -> int:
        return 2


class _EplbTestEvent:
    def record(self, stream: torch.Stream | None = None) -> None:
        pass


@pytest.mark.parametrize("use_async", [False, True])
def test_rearrange_handles_heterogeneous_model_topologies(monkeypatch, use_async):
    """Rearrangement keeps each model's physical expert and group counts."""
    monkeypatch.setattr(
        parallel_config_module.current_platform, "is_cuda_alike", lambda: True
    )
    parallel_config = ParallelConfig(
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config=EPLBConfig(
            window_size=4,
            step_interval=8,
            use_async=use_async,
            communicator="torch_gloo",
        ),
    )
    monkeypatch.setattr(eplb_state_module, "CpuGpuEvent", _EplbTestEvent)
    state = EplbState(parallel_config, torch.device("cpu"))
    group = _EplbTestGroup()
    monkeypatch.setattr(eplb_state_module, "get_ep_group", lambda: group)
    monkeypatch.setattr(eplb_state_module, "get_eplb_group", lambda: group)
    monkeypatch.setattr(eplb_state_module, "get_node_count", lambda: 1)
    monkeypatch.setattr(eplb_state_module, "all_reduce", lambda tensor, group: None)
    monkeypatch.setattr(
        eplb_state_module, "create_eplb_communicator", lambda **kwargs: None
    )
    monkeypatch.setattr(
        eplb_state_module,
        "rearrange_expert_weights_inplace",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(eplb_state_module, "PIN_MEMORY", False)
    models = (
        (_EplbTestModel(num_experts=4, num_redundant=2), "target"),
        (_EplbTestModel(num_experts=6, num_redundant=2), "draft"),
    )
    for model, name in models:
        state.add_model(model, _EplbTestModelConfig(name))
    state.policy = _TopologyEncodingPolicy

    state.rearrange()

    target_state = state.model_states["target"]
    draft_state = state.model_states["draft"]
    if use_async:
        assert target_state.eplb_stats is not None
        assert (
            target_state.eplb_stats.num_replicas,
            target_state.eplb_stats.num_groups,
        ) == (6, 4)
        assert draft_state.eplb_stats is not None
        assert (
            draft_state.eplb_stats.num_replicas,
            draft_state.eplb_stats.num_groups,
        ) == (8, 6)
        assert target_state.rebalanced and draft_state.rebalanced
    else:
        assert torch.equal(
            target_state.physical_to_logical_map,
            torch.tensor([[1, 2, 3, 0, 1, 2]]),
        )
        assert torch.equal(
            draft_state.physical_to_logical_map,
            torch.tensor([[1, 2, 3, 4, 5, 0, 1, 2]]),
        )


def test_allreduce_list_handles_matching_shapes(dist_init):
    state = _make_eplb_state()
    device = torch.device(current_platform.device_type)
    tensors = [
        torch.ones(2, 4, dtype=torch.int32, device=device),
        torch.ones(3, 4, dtype=torch.int32, device=device),
    ]
    result = state._allreduce_list(tensors)
    assert len(result) == 2
    assert result[0].shape == (2, 4)
    assert result[1].shape == (3, 4)


def test_allreduce_list_reduces_single_tensor_in_place(monkeypatch):
    state = _make_eplb_state()
    ep_group = SimpleNamespace(device_group=object())
    all_reduce = MagicMock()
    monkeypatch.setattr(eplb_state_module, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(eplb_state_module, "all_reduce", all_reduce)
    tensor = torch.ones(2, 4, dtype=torch.int32)

    result = state._allreduce_list([tensor])

    assert result[0] is tensor
    all_reduce.assert_called_once_with(tensor, group=ep_group.device_group)


def test_allreduce_list_handles_different_num_physical_experts(dist_init):
    state = _make_eplb_state()
    device = torch.device(current_platform.device_type)
    tensors = [
        torch.ones(2, 4, dtype=torch.int32, device=device),
        torch.ones(3, 6, dtype=torch.int32, device=device),
    ]
    result = state._allreduce_list([t.clone() for t in tensors])
    assert len(result) == 2
    assert result[0].shape == (2, 4)
    assert result[1].shape == (3, 6)
    if dist.is_initialized():
        world_size = dist.get_world_size()
        for tensor in result:
            assert torch.all(tensor == world_size)


def _allreduce_heterogeneous_loads_worker(rank: int, init_method: str) -> None:
    world_size = 2
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        state = EplbState.__new__(EplbState)
        ep_group = SimpleNamespace(device_group=dist.group.WORLD)
        draft_local_load = torch.tensor(
            [[1, 1, 1, 1] if rank == 0 else [0, 0, 0, 99]], dtype=torch.int32
        )
        target_local_load = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1] if rank == 0 else [0, 0, 0, 0, 0, 99],
                [99, 1, 1, 1, 1, 1] if rank == 0 else [1, 0, 0, 0, 0, 0],
            ],
            dtype=torch.int32,
        )
        with patch.object(eplb_state_module, "get_ep_group", return_value=ep_group):
            draft_load, target_load = state._allreduce_list(
                [draft_local_load, target_local_load]
            )

        expected_draft_load = torch.tensor([[1, 1, 1, 100]], dtype=torch.int32)
        expected_target_load = torch.tensor(
            [[1, 1, 1, 1, 1, 100], [100, 1, 1, 1, 1, 1]],
            dtype=torch.int32,
        )
        assert torch.equal(draft_load, expected_draft_load)
        assert torch.equal(target_load, expected_target_load)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_allreduce_heterogeneous_loads_across_two_ranks(tmp_path):
    """Reduce draft/target load tensors with different expert topologies."""
    init_method = f"file://{tmp_path / 'eplb_gloo_init'}"
    mp.spawn(
        _allreduce_heterogeneous_loads_worker,
        args=(init_method,),
        nprocs=2,
        join=True,
    )
