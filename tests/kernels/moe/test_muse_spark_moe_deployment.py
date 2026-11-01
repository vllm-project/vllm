# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deployment policy for the MuseSpark1x post-expert-norm MoE (CPU only)."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.models.muse_spark.nvidia.muse_moe_runner import (
    _validate_route_major_deployment,
    needs_pre_norm_all_reduce,
    validate_deployment,
)


def _moe_config(
    *,
    tp_size: int = 1,
    ep_size: int = 1,
    dp_size: int = 1,
    sp_size: int = 1,
    enable_eplb: bool = False,
    routing_method: RoutingMethodType = RoutingMethodType.MiniMax2,
    skip_final_all_reduce: bool = False,
) -> FusedMoEConfig:
    parallel_config = FusedMoEParallelConfig(
        tp_size=tp_size,
        pcp_size=1,
        dp_size=dp_size,
        ep_size=ep_size,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=sp_size,
        use_ep=ep_size > 1,
        all2all_backend="allgather_reducescatter",
        enable_eplb=enable_eplb,
    )
    return FusedMoEConfig(
        num_experts=32,
        experts_per_token=8,
        hidden_dim=2048,
        intermediate_size=1024,
        num_local_experts=32 // ep_size,
        num_logical_experts=32,
        moe_parallel_config=parallel_config,
        in_dtype=torch.bfloat16,
        max_num_tokens=64,
        activation=MoEActivation.SILU,
        device=torch.device("cpu"),
        routing_method=routing_method,
        skip_final_all_reduce=skip_final_all_reduce,
    )


@pytest.mark.parametrize(
    "kwargs,expect_pre",
    [
        # TP shards the intermediate dim, so GEMM2 rows are partial sums and
        # must be summed *before* the row-wise norm.
        ({"tp_size": 8}, True),
        # EP leaves every expert whole on its owning rank, so the ranks hold
        # disjoint addends, summed after the norm by the runner's all-reduce...
        ({"ep_size": 8}, False),
        # ...or, with DP+EP, by the prepare/finalize combine.
        ({"ep_size": 8, "dp_size": 2, "sp_size": 4}, False),
        ({"ep_size": 8, "dp_size": 2}, False),
        ({}, False),
    ],
)
def test_supported_deployments_reduce_only_before_the_norm(kwargs, expect_pre):
    """The finalizer's only collective is the pre-norm TP all-reduce.

    Under EP, DP ranks call it with different token counts, so any other
    collective would hang.
    """
    config = _moe_config(**kwargs)
    validate_deployment(config, kernel_routes=True)
    assert needs_pre_norm_all_reduce(config) is expect_pre


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"routing_method": RoutingMethodType.Unspecified}, "MiniMax2"),
        ({"routing_method": RoutingMethodType.DeepSeekV3}, "MiniMax2"),
        ({"enable_eplb": True}, "EPLB"),
        # Sequence parallel without an all2all combine has nothing to fold
        # the reduce-scatter into.
        ({"tp_size": 8, "sp_size": 8}, "sequence-parallel"),
        # The runner reduces inside, so an un-reduced output cannot be honored.
        ({"tp_size": 8, "skip_final_all_reduce": True}, "un-reduced"),
    ],
)
def test_unsupported_deployments_are_rejected(kwargs, reason):
    """Unsupported deployments must raise, not skip the norm or miscount sums."""
    with pytest.raises((NotImplementedError, ValueError), match=reason):
        validate_deployment(_moe_config(**kwargs), kernel_routes=True)


def test_routing_is_only_constrained_when_the_kernel_routes():
    """Modular experts take topk_ids from MuseSpark1xRouter, so any routing works."""
    config = _moe_config(routing_method=RoutingMethodType.Unspecified)
    validate_deployment(config, kernel_routes=False)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"tp_size": 8}, None),
        ({"ep_size": 8}, None),
        ({"dp_size": 2}, "DP1"),
        ({"sp_size": 2}, "sequence-parallel"),
        ({"enable_eplb": True}, "EPLB"),
    ],
)
def test_hopper_route_major_deployments(kwargs, match):
    if match is None:
        _validate_route_major_deployment(_moe_config(**kwargs))
        return
    with pytest.raises(NotImplementedError, match=match):
        _validate_route_major_deployment(_moe_config(**kwargs))
