# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from vllm.distributed.device_communicators.all2all import (
    NixlEPAll2AllManager,
    _NixlEPBufferState,
)
from vllm.distributed.elastic_ep import elastic_execute
from vllm.distributed.eplb.eplb_state import EplbState, compute_logical_maps
from vllm.v1.engine import ReconfigureRankType


def test_nixl_ep_reconnects_failed_rank_slots_before_commit(monkeypatch):
    buffer = MagicMock()
    state = _NixlEPBufferState(buffer, 4, 4, 4)
    monkeypatch.setattr(NixlEPAll2AllManager, "_buffer", state)
    monkeypatch.setattr(NixlEPAll2AllManager, "_last_mask", torch.tensor([0, 1, 0, 0]))
    monkeypatch.setattr(NixlEPAll2AllManager, "_mask", torch.zeros(4))
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    manager = object.__new__(NixlEPAll2AllManager)
    manager.world_size = 4
    manager.tcp_store_group = SimpleNamespace(store=object())

    # FT may replay its cumulative dead set; each stale endpoint is dropped once.
    manager.disconnect_ranks([1])
    manager.disconnect_ranks([1])
    buffer.disconnect_ranks.assert_called_once_with([1])

    manager.stage_ep_size()
    manager.stage_ep_size()
    buffer.connect_ranks.assert_called_once_with([1], activate=False)
    buffer.update_mask_buffer.assert_not_called()

    manager.commit_ep_size()
    buffer.update_mask_buffer.assert_called_once_with(1, mask=False)
    assert not state.ft_disconnected_ranks and not state.pending_activation_ranks
    buffer.query_mask_buffer.side_effect = lambda mask: mask.zero_()
    assert not manager.query_fault().item()


def _executor(monkeypatch, dead_ranks, new_dp_size, tp_size=1):
    old_dp_size, local_experts = 4, 2
    width = old_dp_size * tp_size * local_experts
    mapping = torch.arange(width).remainder(local_experts).repeat(2, 1)
    for rank in dead_ranks:
        start = rank * tp_size * local_experts
        mapping[:, start : start + tp_size * local_experts] = -1
    inverse, counts = compute_logical_maps(mapping, local_experts)
    model = MagicMock()
    model.num_logical_experts = local_experts
    model.num_physical_experts = width
    moe = MagicMock()
    moe.moe_config.num_local_experts = local_experts
    moe._quant_method = SimpleNamespace(wraps_legacy_quant_method=False)
    model.modules.return_value = [moe]
    model_state = SimpleNamespace(
        model=model,
        physical_to_logical_map=mapping,
        logical_to_physical_map=inverse,
        logical_replica_count=counts,
        expert_load_pass=torch.arange(2 * width).reshape(2, width),
        expert_load_window=torch.arange(6 * width).reshape(3, 2, width),
        num_unpadded_tokens_tensors=[],
    )
    capacity = max(old_dp_size, new_dp_size) * tp_size * local_experts
    model_state.physical_to_logical_map_buffer = F.pad(
        mapping, (0, capacity - width), value=-1
    )
    model_state.physical_to_logical_map = model_state.physical_to_logical_map_buffer[
        :, :width
    ]
    model_state.expert_load_pass_buffer = F.pad(
        model_state.expert_load_pass, (0, capacity - width)
    )
    model_state.expert_load_pass = model_state.expert_load_pass_buffer[:, :width]
    eplb = object.__new__(EplbState)
    eplb.model_states = {"model": model_state}
    eplb.device = torch.device("cpu")
    eplb.drain_async = MagicMock()
    eplb.update_communicator = MagicMock()
    eplb._propagate_shared_tensors = MagicMock()
    config = SimpleNamespace(
        data_parallel_size=old_dp_size - len(dead_ranks),
        data_parallel_rank=0,
        all2all_backend="allgather_reducescatter",
        eplb_config=SimpleNamespace(num_redundant_experts=width - local_experts),
    )
    worker = SimpleNamespace(
        device=torch.device("cpu"),
        vllm_config=SimpleNamespace(parallel_config=config),
        model_runner=SimpleNamespace(
            model=model,
            get_model=lambda: model,
            model_config=SimpleNamespace(compute_hash=lambda: "model"),
            eplb_state=eplb,
        ),
    )
    executor = object.__new__(elastic_execute.ElasticEPScalingExecutor)
    executor.worker_ref = lambda: worker
    executor.reconfig_request = SimpleNamespace(
        new_data_parallel_size=new_dp_size,
        new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
        new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
        new_data_parallel_master_ip="127.0.0.1",
        new_data_parallel_master_port=29500,
        data_parallel_master_rank=0,
        existing_dp_ranks=[
            rank for rank in range(old_dp_size) if rank not in dead_ranks
        ],
    )
    executor._prepared_eplb_communicator = object()
    executor._release_cuda_graphs = MagicMock()
    executor._make_eep_moe_config = MagicMock()
    executor._commit_staged_moe_quant_methods = MagicMock()
    monkeypatch.setattr(
        elastic_execute, "get_dp_group", lambda: SimpleNamespace(world_size=new_dp_size)
    )
    monkeypatch.setattr(
        elastic_execute,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=new_dp_size * tp_size),
    )
    monkeypatch.setattr(elastic_execute, "_replace_active_groups", MagicMock())
    monkeypatch.setattr(elastic_execute, "pop_standby_groups", dict)
    monkeypatch.setattr(elastic_execute, "get_ep_all2all_manager", MagicMock())
    monkeypatch.setattr(elastic_execute, "is_moe_layer", lambda module: module is moe)
    monkeypatch.setattr(
        elastic_execute, "set_current_vllm_config", lambda _: nullcontext()
    )
    return executor, model_state


@pytest.mark.parametrize(
    "dead_ranks,new_dp_size,tp_size",
    [({2}, 4, 1), ({2}, 5, 1), ({2}, 4, 2)],
)
def test_scale_up_preserves_surviving_experts_and_load_history(
    monkeypatch, dead_ranks, new_dp_size, tp_size
):
    executor, state = _executor(monkeypatch, dead_ranks, new_dp_size, tp_size)
    # Surviving ranks keep their original slots, including holes before them.
    padding = new_dp_size * tp_size * 2 - state.physical_to_logical_map.shape[-1]
    expected_map = F.pad(state.physical_to_logical_map, (0, padding), value=-1)
    expected_pass = F.pad(state.expert_load_pass, (0, padding))
    expected_window = F.pad(state.expert_load_window, (0, padding))
    original = state.physical_to_logical_map_buffer.clone()
    broadcast = MagicMock()
    monkeypatch.setattr(elastic_execute, "get_standby_dp_group", MagicMock())
    monkeypatch.setattr(elastic_execute, "broadcast_expert_mapping", broadcast)
    executor.broadcast_expert_mapping()
    torch.testing.assert_close(
        broadcast.call_args.kwargs["physical_to_logical"],
        original,
    )
    torch.testing.assert_close(state.physical_to_logical_map_buffer, original)

    executor.switch_and_prepare()

    torch.testing.assert_close(state.physical_to_logical_map, expected_map)
    torch.testing.assert_close(state.expert_load_pass, expected_pass)
    torch.testing.assert_close(state.expert_load_window, expected_window)


@pytest.mark.parametrize("commit_fails", [False, True])
def test_scale_up_resumes_eplb_only_after_successful_commit(monkeypatch, commit_fails):
    executor, _ = _executor(monkeypatch, {2}, 4)
    runner = executor.worker.model_runner
    runner.eep_eplb_suppressed = True
    executor.broadcast_expert_mapping = MagicMock()
    executor.switch_and_prepare = MagicMock(return_value=())
    executor.warm_and_capture = MagicMock()
    executor._perform_eplb_reshuffle = MagicMock()
    executor._start_group_cleanup = MagicMock()

    if commit_fails:
        executor._perform_eplb_reshuffle.side_effect = RuntimeError("reshuffle failed")
        with pytest.raises(RuntimeError, match="reshuffle failed"):
            executor.commit_scale_up(is_existing_worker=True)
        assert runner.eep_eplb_suppressed
    else:
        executor.commit_scale_up(is_existing_worker=True)
        assert not runner.eep_eplb_suppressed
