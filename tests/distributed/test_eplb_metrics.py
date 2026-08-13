# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.distributed.eplb.eplb_state as eplb_state_module
from vllm.distributed.eplb.eplb_state import EplbState


class FakeDeviceGroup:
    def __init__(self, size: int, rank: int = 0) -> None:
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


def make_state() -> tuple[EplbState, Mock]:
    expert_load_pass = torch.zeros((2, 4), dtype=torch.int32)
    state = EplbState.__new__(EplbState)
    state.parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(log_balancedness_interval=1)
    )
    state.expert_load_window_step = 0
    state.expert_load_window_size = 1
    state.expert_rearrangement_step = 0
    state.expert_rearrangement_step_interval = 100
    state.should_record_tensor = None
    state.is_async = False
    state.pending_rebalance_events = 0
    state.model_states = {
        "model": SimpleNamespace(
            expert_load_pass=expert_load_pass.clone(),
            expert_load_window=torch.zeros(
                (1, *expert_load_pass.shape),
                dtype=expert_load_pass.dtype,
            ),
            model_name="test-model",
            rebalanced=False,
        )
    }
    sync_load_pass = Mock(return_value=[expert_load_pass.clone()])
    state._sync_load_pass = sync_load_pass
    state.rearrange = Mock()
    return state, sync_load_pass


def make_rearrange_state(
    *,
    is_async: bool,
    num_models: int = 1,
) -> tuple[EplbState, list[SimpleNamespace]]:
    def make_model_state() -> SimpleNamespace:
        physical_to_logical_map = torch.tensor([[0, 1, 0, 1]])
        model = SimpleNamespace(
            num_moe_layers=1,
            num_logical_experts=2,
            num_physical_experts=4,
            num_expert_groups=1,
            expert_weights=[],
        )
        return SimpleNamespace(
            expert_load_window=torch.zeros((1, 1, 4), dtype=torch.int32),
            physical_to_logical_map=physical_to_logical_map,
            model=model,
            expert_buffer=[],
            communicator=SimpleNamespace(),
            rebalanced=False,
            eplb_stats=None,
        )

    model_states = [make_model_state() for _ in range(num_models)]
    state = EplbState.__new__(EplbState)
    state.model_states = {
        f"model-{index}": model_state for index, model_state in enumerate(model_states)
    }
    state.pending_rebalance_events = 0
    state.expert_load_window_size = 1
    state.num_valid_physical_experts = 4
    state.is_async = is_async
    state.policy = SimpleNamespace(
        rebalance_experts=Mock(
            return_value=model_states[0].physical_to_logical_map.clone()
        )
    )
    state._allreduce_list = lambda values: values
    state.rearrange_event = SimpleNamespace(record=Mock())
    return state, model_states


def test_step_reports_instance_state_and_consumes_pending_events(monkeypatch):
    state, sync_load_pass = make_state()
    state.model_states["model"].rebalanced = True
    state.pending_rebalance_events = 3
    ep_group = SimpleNamespace(device_group=FakeDeviceGroup(size=4))
    monkeypatch.setattr(eplb_state_module, "get_ep_group", lambda: ep_group)

    snapshot = state.step(log_stats=False)

    assert snapshot is not None
    assert snapshot.rebalancing is True
    assert snapshot.rebalance_events == 3
    assert state.pending_rebalance_events == 0
    sync_load_pass.assert_not_called()


def test_dummy_step_preserves_events_for_next_normal_output(monkeypatch):
    state, _ = make_state()
    state.pending_rebalance_events = 2
    ep_group = SimpleNamespace(device_group=FakeDeviceGroup(size=4))
    monkeypatch.setattr(eplb_state_module, "get_ep_group", lambda: ep_group)

    dummy_snapshot = state.step(is_dummy=True, log_stats=False)
    real_snapshot = state.step(log_stats=False)

    assert dummy_snapshot is not None
    assert dummy_snapshot.rebalance_events == 2
    assert real_snapshot is not None
    assert real_snapshot.rebalance_events == 2
    assert state.pending_rebalance_events == 0


@pytest.mark.parametrize("is_async", [False, True])
def test_rearrange_counts_one_instance_event(monkeypatch, is_async: bool):
    state, model_states = make_rearrange_state(
        is_async=is_async,
        num_models=2,
    )
    ep_group = SimpleNamespace(device_group=FakeDeviceGroup(size=2, rank=1))
    monkeypatch.setattr(eplb_state_module, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(eplb_state_module, "get_node_count", lambda: 1)
    monkeypatch.setattr(
        eplb_state_module,
        "rearrange_expert_weights_inplace",
        Mock(),
    )
    monkeypatch.setattr(eplb_state_module, "_commit_eplb_maps", Mock())

    state.rearrange()

    assert state.pending_rebalance_events == 1
    assert all(model_state.rebalanced is is_async for model_state in model_states)


def test_profile_rearrange_does_not_increment_counter(monkeypatch):
    state, _ = make_rearrange_state(is_async=False)
    ep_group = SimpleNamespace(device_group=FakeDeviceGroup(size=2, rank=1))
    monkeypatch.setattr(eplb_state_module, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(eplb_state_module, "get_node_count", lambda: 1)
    monkeypatch.setattr(
        eplb_state_module,
        "rearrange_expert_weights_inplace",
        Mock(),
    )

    state.rearrange(is_profile=True)

    assert state.pending_rebalance_events == 0


def test_skipped_rearrange_does_not_increment_counter(monkeypatch):
    state, _ = make_rearrange_state(is_async=False)
    ep_group = SimpleNamespace(device_group=FakeDeviceGroup(size=2, rank=1))
    rearrange_expert_weights = Mock()
    monkeypatch.setattr(eplb_state_module, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(eplb_state_module, "get_node_count", lambda: 1)
    monkeypatch.setattr(eplb_state_module.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(
        eplb_state_module,
        "rearrange_expert_weights_inplace",
        rearrange_expert_weights,
    )

    state.rearrange()

    assert state.pending_rebalance_events == 0
    rearrange_expert_weights.assert_not_called()
