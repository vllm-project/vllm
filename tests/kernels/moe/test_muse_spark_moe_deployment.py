# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deployment policy for the Muse Spark post-expert-norm MoE (CPU only), and
the BF16 expert-TP reshard of the converted weights (CUDA)."""

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
from vllm.platforms import current_platform


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
    """Modular experts take topk_ids from MuseSparkRouter, so any routing works."""
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


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the stock conversion runs on CUDA"
)
@torch.inference_mode()
def test_bf16_expert_tp_slice_is_the_converted_slice():
    """Rank `r`'s rows `bf16_w13_rows(...)[r]` of the converted w13 and its
    K-block range of the converted w2 (`reshard_bf16`) are, bit for bit, the
    stock block-major-K conversion of its TP8 intermediate slice."""
    pytest.importorskip("flashinfer")
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        convert_moe_weights_to_flashinfer_trtllm_block_layout as convert,
    )
    from vllm.models.muse_spark.nvidia.specialized_decode.runtime import expert_tp

    tp, hidden, inter, experts, bk = 8, 256, 2048, 3, expert_tp.BLOCK_K
    s = inter // tp
    torch.manual_seed(0)
    w13 = torch.randn(experts, 2 * inter, hidden, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(experts, hidden, inter, dtype=torch.bfloat16, device="cuda")
    raw13, raw2 = w13.clone(), w2.clone()
    convert({}, w13, w2, is_gated_act_gemm=True)
    full13 = w13.view(experts, hidden // bk, 2 * inter, bk)
    full2 = w2.view(experts, inter // bk, hidden, bk)
    rows = expert_tp.bf16_w13_rows(inter, tp, "cuda").cuda()
    for r in range(tp):
        gate_up = raw13.view(experts, 2, inter, hidden)[:, :, r * s : (r + 1) * s]
        slice13 = gate_up.reshape(experts, 2 * s, hidden).contiguous()
        slice2 = raw2[:, :, r * s : (r + 1) * s].contiguous()
        convert({}, slice13, slice2, is_gated_act_gemm=True)
        assert torch.equal(
            full13.index_select(2, rows[r]),
            slice13.view(experts, hidden // bk, 2 * s, bk),
        )
        blocks = s // bk
        assert torch.equal(
            full2[:, r * blocks : (r + 1) * blocks],
            slice2.view(experts, blocks, hidden, bk),
        )
