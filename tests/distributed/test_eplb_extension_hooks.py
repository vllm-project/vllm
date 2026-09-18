# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config.parallel import EPLBConfig, ParallelConfig
from vllm.distributed.eplb.async_worker import run_rebalance_experts
from vllm.distributed.eplb.eplb_state import (
    EplbLayerState,
    EplbState,
    _move_to_workspace,
    _node_count_with_rank_mapping,
)
from vllm.distributed.eplb.policy import (
    AbstractEplbPolicy,
    DefaultEplbPolicy,
    EplbPlan,
    EplbRebalanceContext,
    EplbTopology,
)
from vllm.model_executor.layers.fused_moe.router.base_router import (
    eplb_map_to_physical_and_record,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import Platform


@pytest.mark.parametrize("communicator", [None, "torch_gloo"])
def test_parallel_config_delegates_eplb_config_to_platform(communicator):
    def configure(config):
        if config.eplb_config.communicator is None:
            config.eplb_config.communicator = "torch_gloo"

    with patch("vllm.config.parallel.current_platform") as platform:
        platform.supports_eplb.return_value = True
        platform.is_cuda_alike.return_value = False
        platform.check_and_update_eplb_config.side_effect = configure
        config = ParallelConfig(
            enable_eplb=True,
            enable_expert_parallel=True,
            tensor_parallel_size=2,
            distributed_executor_backend="uni",
            eplb_config=EPLBConfig(communicator=communicator),
        )
    assert config.eplb_config.communicator == "torch_gloo"
    platform.check_and_update_eplb_config.assert_called_once_with(config)


def test_platform_eplb_capability_is_authoritative():
    with (
        patch("vllm.config.parallel.current_platform") as platform,
        pytest.raises(ValueError, match="not supported on this platform"),
    ):
        platform.supports_eplb.return_value = False
        platform.is_cuda_alike.return_value = True
        ParallelConfig(
            enable_eplb=True,
            enable_expert_parallel=True,
            tensor_parallel_size=2,
            distributed_executor_backend="uni",
        )


@pytest.mark.parametrize(
    ("nixl_available", "elastic", "expected"),
    [(True, False, "nixl"), (False, False, "torch_gloo"), (False, True, "pynccl")],
)
def test_platform_preserves_default_communicator_selection(
    nixl_available, elastic, expected
):
    config = SimpleNamespace(
        eplb_config=EPLBConfig(use_async=False),
        enable_elastic_ep=elastic,
    )
    with patch(
        "vllm.distributed.nixl_utils.is_nixl_available",
        return_value=nixl_available,
    ):
        Platform.check_and_update_eplb_config(config)
    assert config.eplb_config.communicator == expected


def test_platform_rejects_async_pynccl_fallback():
    config = SimpleNamespace(
        eplb_config=EPLBConfig(use_async=True),
        enable_elastic_ep=True,
    )
    with (
        patch("vllm.distributed.nixl_utils.is_nixl_available", return_value=False),
        pytest.raises(ValueError, match="incompatible with async EPLB"),
    ):
        Platform.check_and_update_eplb_config(config)


def test_platform_rejects_unknown_communicator():
    config = SimpleNamespace(
        eplb_config=SimpleNamespace(communicator="unknown", use_async=False),
    )
    with pytest.raises(ValueError, match="Unknown EPLB communicator"):
        Platform.check_and_update_eplb_config(config)


def test_async_planner_uses_state_hook():
    load_window = torch.tensor([[[3, 1]]])
    old_map = torch.tensor([[0, 1]])
    new_map = torch.tensor([[1, 0]])
    plan = EplbPlan(new_map)
    state = SimpleNamespace(plan_rebalance=Mock(return_value=plan))
    model_state = SimpleNamespace(
        eplb_stats=SimpleNamespace(
            global_expert_load_window=load_window,
            num_replicas=2,
            num_groups=1,
            num_nodes=1,
            num_gpus=2,
        )
    )
    stream = object()
    cpu_group = object()

    with (
        patch("torch.cuda.stream", return_value=nullcontext()),
        patch(
            "vllm.distributed.eplb.async_worker.get_eplb_group",
            return_value=SimpleNamespace(cpu_group=cpu_group),
        ),
    ):
        assert run_rebalance_experts(model_state, state, old_map, stream) is plan

    assert state.plan_rebalance.call_args.args[0] is model_state
    context = state.plan_rebalance.call_args.args[1]
    assert isinstance(context, EplbRebalanceContext)
    assert context.load_window_cpu is load_window
    assert context.physical_to_logical_map_cpu is old_map
    assert context.num_replicas == 2
    assert context.topology.num_groups == 1
    assert context.topology.num_nodes == 1
    assert context.topology.num_ranks == 2
    assert context.cpu_group is cpu_group


def test_policy_planner_adapts_temporal_window_to_legacy_contract():
    class LegacyPolicy(AbstractEplbPolicy):
        received_args = None

        def plan_rebalance(self, context):
            return self._plan_from_legacy(context)

        @classmethod
        def rebalance_experts(cls, *args):
            cls.received_args = args
            return torch.tensor([[1, 0]])

    context = EplbRebalanceContext(
        load_window_cpu=torch.tensor([[[3, 1]], [[4, 2]]]),
        physical_to_logical_map_cpu=torch.tensor([[0, 1]]),
        topology=EplbTopology(num_groups=1, num_nodes=1, num_ranks=2),
        num_replicas=2,
        cpu_group=Mock(),
    )

    plan = LegacyPolicy().plan_rebalance(context)

    assert LegacyPolicy.received_args is not None
    weight, num_replicas, num_groups, num_nodes, num_ranks, old_map = (
        LegacyPolicy.received_args
    )
    torch.testing.assert_close(weight, torch.tensor([[7, 3]]))
    assert (num_replicas, num_groups, num_nodes, num_ranks) == (2, 1, 1, 2)
    torch.testing.assert_close(old_map, context.physical_to_logical_map_cpu)
    torch.testing.assert_close(plan.physical_to_logical_map, torch.tensor([[1, 0]]))


def test_state_creates_independent_policy_instances():
    state = EplbState.__new__(EplbState)
    state.parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(policy="default")
    )

    first = state.create_policy()
    second = state.create_policy()

    assert isinstance(first, DefaultEplbPolicy)
    assert isinstance(second, DefaultEplbPolicy)
    assert first is not second


@pytest.mark.parametrize(
    "target",
    [torch.zeros(1, 1), torch.zeros(1, 1, dtype=torch.int64, device="meta")],
)
def test_state_rejects_invalid_policy_target(target):
    state = EplbState.__new__(EplbState)
    plan = EplbPlan(target)
    model_state = SimpleNamespace(
        policy=SimpleNamespace(plan_rebalance=Mock(return_value=plan))
    )

    with pytest.raises(ValueError, match="CPU int32 or int64"):
        state.plan_rebalance(model_state, Mock())


def test_state_accepts_cpu_integer_policy_target():
    state = EplbState.__new__(EplbState)
    plan = EplbPlan(torch.zeros(1, 1, dtype=torch.int32))
    model_state = SimpleNamespace(
        policy=SimpleNamespace(plan_rebalance=Mock(return_value=plan))
    )

    assert state.plan_rebalance(model_state, Mock()) is plan


def test_state_allreduces_temporal_windows_for_multiple_models():
    state = EplbState.__new__(EplbState)
    first = torch.arange(6).reshape(2, 1, 3)
    second = torch.arange(12).reshape(2, 2, 3)
    device_group = object()

    with (
        patch(
            "vllm.distributed.eplb.eplb_state.get_ep_group",
            return_value=SimpleNamespace(device_group=device_group),
        ),
        patch(
            "vllm.distributed.eplb.eplb_state.all_reduce",
            side_effect=lambda tensor, **_: tensor.add_(10),
        ) as all_reduce,
    ):
        reduced = state._allreduce_list([first, second])

    assert all_reduce.call_args.kwargs["group"] is device_group
    torch.testing.assert_close(reduced[0], first + 10)
    torch.testing.assert_close(reduced[1], second + 10)


@pytest.mark.parametrize(
    ("load_window", "next_step", "expected"),
    [
        ([1, 2, 0, 0], 2, [0, 0, 1, 2]),
        ([3, 4, 1, 2], 2, [1, 2, 3, 4]),
    ],
)
def test_state_orders_load_slots(load_window, next_step, expected):
    state = EplbState.__new__(EplbState)
    state.expert_load_window_size = 4
    state.expert_load_window_step = next_step
    ordered = state._ordered_load_window(torch.tensor(load_window))

    torch.testing.assert_close(ordered, torch.tensor(expected))


def test_dummy_step_advances_load_window_with_zero_sample():
    state = EplbState.__new__(EplbState)
    model_state = SimpleNamespace(
        expert_load_pass=torch.tensor([[7]]),
        expert_load_window=torch.full((4, 1, 1), -1),
    )
    state.model_states = {"model": model_state}
    state.expert_load_window_size = 4
    state.expert_load_window_step = 0
    state.expert_rearrangement_step = 0
    state.expert_rearrangement_step_interval = 10
    state.is_async = False
    state._should_record_current_step = Mock(return_value=True)
    state._update_layer_should_record = Mock()

    with patch(
        "vllm.distributed.eplb.eplb_state.get_ep_group",
        return_value=SimpleNamespace(device_group=Mock()),
    ):
        state.step(is_dummy=True)

    torch.testing.assert_close(
        model_state.expert_load_window[0], torch.zeros(1, 1, dtype=torch.int64)
    )
    assert state.expert_load_window_step == 1


def test_reconfigure_resets_load_timeline_for_scale_up():
    state = EplbState.__new__(EplbState)
    map_buffer = torch.zeros(1, 4, dtype=torch.int64)
    load_buffer = torch.ones(1, 4, dtype=torch.int64)
    model_state = SimpleNamespace(
        model=SimpleNamespace(num_physical_experts=2),
        physical_to_logical_map_buffer=map_buffer,
        physical_to_logical_map=map_buffer[:, :2],
        expert_load_pass_buffer=load_buffer,
        expert_load_pass=load_buffer[:, :2],
        expert_load_window=torch.ones(4, 1, 2, dtype=torch.int64),
        policy=object(),
    )
    draft_state = SimpleNamespace(
        expert_load_window=torch.ones(4, 1, 2, dtype=torch.int64),
        policy=object(),
    )
    state.model_states = {"model": model_state, "draft": draft_state}
    state.expert_load_window_step = 2
    new_policy = object()
    new_draft_policy = object()
    state.create_policy = Mock(side_effect=[new_policy, new_draft_policy])

    state.reconfigure_physical_expert_slots(
        SimpleNamespace(compute_hash=Mock(return_value="model")), 3
    )

    assert model_state.expert_load_window.shape == (4, 1, 3)
    assert not model_state.expert_load_window.any()
    assert model_state.policy is new_policy
    assert not draft_state.expert_load_window.any()
    assert draft_state.policy is new_draft_policy
    assert state.expert_load_window_step == 0


def test_empty_load_window_keeps_one_zero_sample():
    empty = torch.zeros(4, 2, 3, dtype=torch.int64)

    filtered = EplbState._nonempty_load_window(empty)

    assert filtered.shape == (1, 2, 3)
    assert not filtered.any()


@pytest.mark.parametrize(
    ("node_ids", "rank_mapping", "expected"),
    [
        ([0, 0, 0, 0, 1, 1], None, 1),
        ([0, 0, 0, 0, 1, 1, 1, 1], None, 2),
        ([0, 0, 0, 0, 1, 1, 1, 1], {i: i if i < 6 else -1 for i in range(8)}, 1),
        (
            [0, 0, 0, 0, 1, 1, 1, 1],
            {0: 0, 1: 1, 2: -1, 3: -1, 4: 2, 5: 3, 6: -1, 7: -1},
            2,
        ),
        ([0, 0, 1, 1], {0: 0, 2: 1, 1: 2, 3: 3}, 1),
    ],
)
def test_node_count_requires_uniform_contiguous_active_ranks(
    node_ids, rank_mapping, expected
):
    world_size = len(node_ids)
    mapping = rank_mapping or {rank: rank for rank in range(world_size)}

    def same_node(_group, source_rank):
        return [node == node_ids[source_rank] for node in node_ids]

    with patch(
        "vllm.distributed.eplb.eplb_state.in_the_same_node_as",
        side_effect=same_node,
    ):
        count = _node_count_with_rank_mapping(
            SimpleNamespace(world_size=world_size), mapping
        )

    assert count == expected


def test_state_creates_device_specific_layer_state():
    class CustomLayerState(EplbLayerState):
        pass

    class CustomEplbState(EplbState):
        def create_layer_state(self):
            return CustomLayerState()

    layer = SimpleNamespace(eplb_state=EplbLayerState())
    state = CustomEplbState.__new__(CustomEplbState)
    state._create_model_layer_states(SimpleNamespace(moe_layers=[layer]))
    assert isinstance(layer.eplb_state, CustomLayerState)


def test_model_communicator_uses_state_hook():
    state = EplbState.__new__(EplbState)
    state.parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(communicator="torch_gloo")
    )
    model = SimpleNamespace(expert_weights=[torch.empty(1)])
    buffer = [torch.empty(1)]
    group = object()
    communicator = object()
    with patch(
        "vllm.distributed.eplb.eplb_state.create_eplb_communicator",
        return_value=communicator,
    ) as factory:
        assert state.create_model_communicator(model, buffer, group) is communicator
    factory.assert_called_once_with(group, "torch_gloo", model.expert_weights, buffer)


def test_layer_commit_hook_runs_before_worker_ack():
    calls = []
    event = SimpleNamespace(record=lambda: calls.append("ack"))
    result = SimpleNamespace(
        layer_idx=0,
        new_physical_to_logical_map=torch.tensor([1, 0]),
        transfer_metadata=object(),
        consumed_event=event,
    )
    model_state = SimpleNamespace(
        pending_result=result,
        model=SimpleNamespace(expert_weights=[[torch.empty(1)]], num_moe_layers=1),
        expert_buffer=[torch.empty(1)],
        rebalanced=True,
    )
    state = EplbState.__new__(EplbState)
    state.on_layer_committed = lambda _state, _layer: calls.append("hook")
    with (
        patch(
            "vllm.distributed.eplb.eplb_state.move_from_buffer",
            side_effect=lambda **_: calls.append("weights"),
        ),
        patch(
            "vllm.distributed.eplb.eplb_state._commit_eplb_maps_for_layer",
            side_effect=lambda *_args, **_kwargs: calls.append("maps"),
        ),
    ):
        _move_to_workspace(
            model_state,
            ep_rank=0,
            commit_rebalance=state._commit_rebalance,
        )
    assert calls == ["weights", "maps", "hook", "ack"]
    assert model_state.pending_result is None
    assert not model_state.rebalanced


def test_failed_commit_hook_does_not_ack_buffer():
    event = SimpleNamespace(record=Mock())
    result = SimpleNamespace(
        layer_idx=0,
        new_physical_to_logical_map=torch.tensor([1, 0]),
        transfer_metadata=object(),
        consumed_event=event,
    )
    model_state = SimpleNamespace(
        pending_result=result,
        model=SimpleNamespace(expert_weights=[[torch.empty(1)]], num_moe_layers=1),
        expert_buffer=[torch.empty(1)],
        rebalanced=True,
    )

    def fail_refresh(_state, _layer):
        raise RuntimeError("refresh failed")

    state = EplbState.__new__(EplbState)
    state.on_layer_committed = fail_refresh
    with (
        patch("vllm.distributed.eplb.eplb_state.move_from_buffer"),
        patch("vllm.distributed.eplb.eplb_state._commit_eplb_maps_for_layer"),
        pytest.raises(RuntimeError, match="refresh failed"),
    ):
        _move_to_workspace(
            model_state,
            ep_rank=0,
            commit_rebalance=state._commit_rebalance,
        )
    assert model_state.pending_result is result
    event.record.assert_not_called()


def test_sync_commit_uses_layer_commit_hook():
    calls = []
    model_state = SimpleNamespace(model=SimpleNamespace(num_moe_layers=2))
    state = EplbState.__new__(EplbState)
    state.on_layer_committed = lambda _state, layer: calls.append(f"hook-{layer}")

    with patch(
        "vllm.distributed.eplb.eplb_state._commit_eplb_maps",
        side_effect=lambda *_args: calls.append("maps"),
    ):
        state._commit_rebalance(model_state, torch.tensor([[0], [0]]))

    assert calls == ["maps", "hook-0", "hook-1"]


def test_router_uses_device_specific_mapping_hook():
    mapped_ids = torch.tensor([[1]])
    callback = Mock(return_value=mapped_ids)
    state = EplbLayerState(
        expert_load_view=torch.zeros(2),
        logical_to_physical_map=torch.arange(2).unsqueeze(-1),
        logical_replica_count=torch.ones(2),
        should_record_tensor=torch.ones((), dtype=torch.bool),
        num_unpadded_tokens_tensors=[torch.tensor(1)],
        map_and_record=callback,
    )
    router = FusedTopKRouter(top_k=1, global_num_experts=2, eplb_state=state)
    logical_ids = torch.tensor([[0]])
    with patch(
        "vllm.model_executor.layers.fused_moe.router.base_router.dbo_current_ubatch_id",
        return_value=0,
    ):
        assert router._apply_eplb_mapping(logical_ids) is mapped_ids
    callback.assert_called_once_with(logical_ids, state.num_unpadded_tokens_tensors[0])


def test_router_uses_default_mapping_without_platform_gate():
    mapped_ids = torch.tensor([[1]])
    state = EplbLayerState(
        expert_load_view=torch.zeros(2),
        logical_to_physical_map=torch.arange(2).unsqueeze(-1),
        logical_replica_count=torch.ones(2),
        should_record_tensor=torch.ones((), dtype=torch.bool),
        num_unpadded_tokens_tensors=[torch.tensor(1)],
    )
    router = FusedTopKRouter(top_k=1, global_num_experts=2, eplb_state=state)
    with (
        patch(
            "vllm.model_executor.layers.fused_moe.router.base_router.dbo_current_ubatch_id",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.router.base_router.current_platform.is_cuda_alike",
            return_value=False,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.router.base_router.eplb_map_to_physical_and_record",
            return_value=mapped_ids,
        ) as default_mapping,
    ):
        assert router._apply_eplb_mapping(torch.tensor([[0]])) is mapped_ids
    default_mapping.assert_called_once()


@pytest.mark.skipif(current_platform.is_cuda_alike(), reason="Non-CUDA fallback only")
def test_non_cuda_default_mapping_fails_closed():
    with pytest.raises(RuntimeError, match="device-specific map_and_record hook"):
        eplb_map_to_physical_and_record(
            topk_ids=torch.tensor([[0]]),
            expert_load_view=torch.zeros(1),
            logical_to_physical_map=torch.tensor([[0]]),
            logical_replica_count=torch.ones(1),
            record_enabled=torch.ones((), dtype=torch.bool),
        )
