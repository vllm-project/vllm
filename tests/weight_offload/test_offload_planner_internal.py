# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch.nn as nn

from vllm.model_executor.offloader.planner import build_prefetch_offload_plan


class _ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class _ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class _ToyExperts(nn.Module):
    def __init__(self, tp_size: int = 1):
        super().__init__()
        local_intermediate = 8 // tp_size
        self.gate_up_proj = nn.Linear(4, local_intermediate, bias=False)
        self.down_proj = nn.Linear(local_intermediate, 4, bias=False)


class _ToyMoeBlock(nn.Module):
    def __init__(self, tp_size: int = 1):
        super().__init__()
        self.gate = nn.Linear(4, 2, bias=False)
        self.shared_expert_gate = nn.Linear(4, 1, bias=False)
        self.experts = _ToyExperts(tp_size=tp_size)
        self.shared_expert = _ToyMLP()


class _ToyLayer(nn.Module):
    def __init__(self, tp_size: int = 1):
        super().__init__()
        self.self_attn = _ToyAttention()
        self.mlp = _ToyMLP()
        self.block_sparse_moe = _ToyMoeBlock(tp_size=tp_size)


def test_prefetch_offload_planner_respects_group_pattern():
    layers = [_ToyLayer() for _ in range(8)]

    plan = build_prefetch_offload_plan(
        layers,
        group_size=4,
        num_in_group=1,
    )

    assert plan.modules == layers
    assert [unit.module_index for unit in plan.units] == [3, 7]


def test_prefetch_offload_planner_unions_selectors_and_include_names():
    plan = build_prefetch_offload_plan(
        [_ToyLayer()],
        group_size=1,
        num_in_group=1,
        selectors={"routed_experts"},
        include_names={"o_proj"},
    )

    assert len(plan.units) == 1
    assert set(plan.units[0].param_names) == {
        "self_attn.o_proj.weight",
        "block_sparse_moe.experts.gate_up_proj.weight",
        "block_sparse_moe.experts.down_proj.weight",
    }


@pytest.mark.parametrize(
    ("selectors", "expected_param_names"),
    [
        (
            {"routed_experts"},
            {
                "block_sparse_moe.experts.gate_up_proj.weight",
                "block_sparse_moe.experts.down_proj.weight",
            },
        ),
        (
            {"routed_experts", "shared_experts"},
            {
                "block_sparse_moe.experts.gate_up_proj.weight",
                "block_sparse_moe.experts.down_proj.weight",
                "block_sparse_moe.shared_expert.gate_up_proj.weight",
                "block_sparse_moe.shared_expert.down_proj.weight",
            },
        ),
    ],
)
def test_prefetch_offload_planner_makes_shared_expert_policy_explicit(
    selectors: set[str],
    expected_param_names: set[str],
):
    plan = build_prefetch_offload_plan(
        [_ToyLayer()],
        group_size=1,
        num_in_group=1,
        selectors=selectors,
    )

    assert len(plan.units) == 1
    assert set(plan.units[0].param_names) == expected_param_names


@pytest.mark.parametrize(
    ("selectors", "expected_param_names"),
    [
        (
            {"attention"},
            {
                "self_attn.qkv_proj.weight",
                "self_attn.o_proj.weight",
            },
        ),
        (
            {"dense_mlp"},
            {
                "mlp.gate_up_proj.weight",
                "mlp.down_proj.weight",
            },
        ),
        (
            {"shared_experts"},
            {
                "block_sparse_moe.shared_expert.gate_up_proj.weight",
                "block_sparse_moe.shared_expert.down_proj.weight",
            },
        ),
    ],
)
def test_prefetch_offload_planner_selects_non_routed_units(
    selectors: set[str],
    expected_param_names: set[str],
):
    plan = build_prefetch_offload_plan(
        [_ToyLayer()],
        group_size=1,
        num_in_group=1,
        selectors=selectors,
    )

    assert len(plan.units) == 1
    assert set(plan.units[0].param_names) == expected_param_names


@pytest.mark.parametrize(
    ("tp_size", "expected_shapes"),
    [
        (1, ((8, 4), (4, 8))),
        (2, ((4, 4), (4, 4))),
        (4, ((2, 4), (4, 2))),
    ],
)
def test_prefetch_offload_planner_tracks_local_tp_shards(
    tp_size: int,
    expected_shapes: tuple[tuple[int, ...], tuple[int, ...]],
):
    layers = [_ToyLayer(tp_size=tp_size) for _ in range(4)]

    plan = build_prefetch_offload_plan(
        layers,
        group_size=2,
        num_in_group=1,
        selectors={"routed_experts"},
    )

    assert [unit.module_index for unit in plan.units] == [1, 3]
    assert all(
        unit.param_names
        == (
            "block_sparse_moe.experts.gate_up_proj.weight",
            "block_sparse_moe.experts.down_proj.weight",
        )
        for unit in plan.units
    )

    params = dict(plan.units[0].module.named_parameters())
    assert tuple(params[plan.units[0].param_names[0]].shape) == expected_shapes[0]
    assert tuple(params[plan.units[0].param_names[1]].shape) == expected_shapes[1]
