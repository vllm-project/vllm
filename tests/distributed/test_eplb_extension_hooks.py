# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
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
    old_map = torch.tensor([[0, 1]])
    new_map = torch.tensor([[1, 0]])
    state = SimpleNamespace(plan_rebalance=Mock(return_value=new_map))
    model_state = SimpleNamespace()
    stream = object()

    assert run_rebalance_experts(model_state, state, old_map, stream) is new_map
    state.plan_rebalance.assert_called_once_with(model_state, old_map, stream)


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
            on_committed=lambda _state, _layer: calls.append("hook"),
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

    with (
        patch("vllm.distributed.eplb.eplb_state.move_from_buffer"),
        patch("vllm.distributed.eplb.eplb_state._commit_eplb_maps_for_layer"),
        pytest.raises(RuntimeError, match="refresh failed"),
    ):
        _move_to_workspace(
            model_state,
            ep_rank=0,
            on_committed=fail_refresh,
        )
    assert model_state.pending_result is result
    event.record.assert_not_called()


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
