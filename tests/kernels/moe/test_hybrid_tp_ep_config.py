# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.config as config_module
import vllm.distributed as distributed
import vllm.model_executor.layers.fused_moe.config as moe_config
import vllm.model_executor.layers.fused_moe.runner.moe_runner as moe_runner
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    determine_expert_map,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


def test_explicit_ep_preserves_tensor_parallelism(monkeypatch) -> None:
    monkeypatch.setattr(
        moe_config,
        "get_dp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=1),
    )
    monkeypatch.setattr(
        moe_config,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=1),
    )
    monkeypatch.setattr(
        moe_config,
        "get_pcp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    monkeypatch.setattr(moe_config, "get_tensor_model_parallel_rank", lambda: 3)
    parallel_config = SimpleNamespace(
        enable_expert_parallel=True,
        expert_parallel_size=2,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )

    result = FusedMoEParallelConfig.make(
        tp_size_=4,
        pcp_size_=1,
        dp_size_=2,
        sp_size_=1,
        vllm_parallel_config=parallel_config,
    )

    assert result.tp_size == 4
    assert result.tp_rank == 3
    assert result.ep_size == 2
    assert result.ep_rank == 1
    assert result.dp_size == 2
    assert result.dp_rank == 1


def test_explicit_ep_owns_half_of_global_experts() -> None:
    local_count, expert_map, _ = determine_expert_map(
        ep_size=2,
        ep_rank=1,
        global_num_experts=3584,
    )

    assert local_count == 1792
    assert expert_map is not None
    assert torch.count_nonzero(expert_map >= 0).item() == 1792
    assert torch.all(expert_map[:1792] == -1)
    assert torch.equal(expert_map[1792:], torch.arange(1792, dtype=torch.int32))


def test_weight_filter_uses_explicit_ep_ownership(monkeypatch) -> None:
    parallel_config = SimpleNamespace(
        enable_expert_parallel=True,
        enable_ep_weight_filter=True,
        enable_eplb=False,
        expert_parallel_size=2,
        data_parallel_size=2,
        tensor_parallel_size=4,
        prefill_context_parallel_size=1,
        expert_placement_strategy="linear",
    )
    monkeypatch.setattr(
        config_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(parallel_config=parallel_config),
    )
    monkeypatch.setattr(
        distributed,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=1),
    )
    loader = DefaultModelLoader.__new__(DefaultModelLoader)
    loader.local_expert_ids = None
    model_config = SimpleNamespace(is_moe=True, get_num_experts=lambda: 3584)

    loader._init_ep_weight_filter(model_config)

    assert loader.local_expert_ids == set(range(1792, 3584))


def test_hybrid_ep_keeps_late_tp_reduction(monkeypatch) -> None:
    calls = []

    def fake_all_reduce(states):
        calls.append(states)
        return states + 1

    monkeypatch.setattr(moe_runner, "tensor_model_parallel_all_reduce", fake_all_reduce)
    runner = SimpleNamespace(
        moe_config=SimpleNamespace(
            is_sequence_parallel=False,
            skip_final_all_reduce=False,
            tp_size=4,
            ep_size=2,
        ),
        _fused_output_is_reduced=False,
    )
    states = torch.zeros(2, 4)

    result = MoERunner._maybe_reduce_final_output(
        runner,
        states,
        trunc_size=None,
        output_is_reduced=False,
    )

    assert calls == [states]
    assert torch.equal(result, states + 1)
