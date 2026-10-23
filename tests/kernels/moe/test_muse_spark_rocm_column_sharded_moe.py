# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x ROCm column-sharded MoE down projection.

The finalize kernels are checked on one GPU against the full-width fused
finalize; the whole path (re-sharded ``w2``, gathered stage2 input, column
finalize) is checked on 2 and 8 GPUs against the AITER experts followed by the
reduce-scatter finalize.
"""

import os
import sys

import pytest
import torch
import torch.multiprocessing as mp

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.models.muse_spark.amd import column_sharded_moe
from vllm.models.muse_spark.common.ops.fused_finalize import (
    fused_post_expert_finalize,
)
from vllm.platforms import current_platform

requires_rocm = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="MuseSpark ROCm kernels"
)


def _moe_config(
    *,
    tp_size: int = 8,
    ep_size: int = 1,
    hidden_dim: int = 4096,
    intermediate_size: int = 4096,
    activation: MoEActivation = MoEActivation.SILU,
) -> FusedMoEConfig:
    parallel_config = FusedMoEParallelConfig(
        tp_size=tp_size,
        pcp_size=1,
        dp_size=1,
        ep_size=ep_size,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=ep_size > 1,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    return FusedMoEConfig(
        num_experts=256,
        experts_per_token=8,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        num_local_experts=256 // ep_size,
        num_logical_experts=256,
        moe_parallel_config=parallel_config,
        in_dtype=torch.bfloat16,
        max_num_tokens=64,
        activation=activation,
        device=torch.device("cpu"),
        routing_method=RoutingMethodType.MiniMax2,
    )


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, True),
        ({"tp_size": 2}, True),
        ({"tp_size": 1}, False),
        # Under EP the MoE runs at tp_size 1 and keeps the full w2 per expert.
        ({"tp_size": 1, "ep_size": 8}, False),
        ({"tp_size": 8, "ep_size": 8}, False),
        ({"hidden_dim": 3072}, False),
        ({"intermediate_size": 3072}, False),
        ({"activation": MoEActivation.GELU}, False),
    ],
)
def test_supported_deployments(monkeypatch, kwargs, expected) -> None:
    monkeypatch.setattr(
        column_sharded_moe, "_aiter_supports_stage2_override", lambda: True
    )
    monkeypatch.setenv("VLLM_ROCM_MUSE_MOE_COLUMN_SHARDED_DOWN", "1")
    config = _moe_config(**kwargs)
    assert column_sharded_moe.column_sharded_down_supported(config) is expected
    monkeypatch.setenv("VLLM_ROCM_MUSE_MOE_COLUMN_SHARDED_DOWN", "0")
    assert not column_sharded_moe.column_sharded_down_supported(config)


@requires_rocm
@pytest.mark.parametrize("num_tokens", [1, 1024])
@pytest.mark.parametrize("tp_size", [1, 8])
@pytest.mark.parametrize("weight_first", [True, False])
@pytest.mark.parametrize("has_weight", [True, False])
@torch.inference_mode()
def test_column_sharded_finalize(
    num_tokens: int, tp_size: int, weight_first: bool, has_weight: bool
) -> None:
    """Each simulated rank holds every route on its column slice; summing the
    per-rank sums of squares stands in for the all-reduce. With one rank the
    result is bit-identical to the fused finalize; with more, only the order
    of the fp32 sum of squares differs."""
    from vllm.models.muse_spark.amd.ops.column_sharded_finalize import (
        column_sharded_finalize,
        route_sum_squares,
    )

    torch.manual_seed(0)
    top_k, hidden, eps = 8, 4096, 1e-6
    gain_center = 1.0 if has_weight else 0.0
    routed = torch.randn(num_tokens, top_k, hidden, device="cuda", dtype=torch.bfloat16)
    weight = (
        torch.randn(hidden, device="cuda", dtype=torch.float32) * 0.1
        if has_weight
        else None
    )
    topk_weights = torch.rand(num_tokens, top_k, device="cuda", dtype=torch.float32)

    ref = torch.empty(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    fused_post_expert_finalize(
        routed, weight, topk_weights, eps, gain_center, weight_first, ref
    )

    cols = hidden // tp_size
    shards = [
        routed[..., r * cols : (r + 1) * cols].contiguous() for r in range(tp_size)
    ]
    weights = [
        None if weight is None else weight[r * cols : (r + 1) * cols]
        for r in range(tp_size)
    ]
    sum_squares = sum(
        route_sum_squares(x, w, gain_center, weight_first)
        for x, w in zip(shards, weights)
    )
    outs = []
    for x, w in zip(shards, weights):
        out = torch.empty(num_tokens, cols, device="cuda", dtype=torch.bfloat16)
        column_sharded_finalize(
            x,
            w,
            topk_weights,
            sum_squares,
            hidden,
            eps,
            gain_center,
            weight_first,
            out,
        )
        outs.append(out)
    got = torch.cat(outs, dim=-1)
    if tp_size == 1:
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
    else:
        torch.testing.assert_close(got, ref, rtol=1.6e-2, atol=1e-2)


# Muse-Spark-Open's MoE at TP8: hidden 4096, 512 intermediate columns per rank.
_E, _H, _I_PER_RANK, _TOP_K = 256, 4096, 512, 8


def _quantize_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    e, rows, cols = w.shape
    packed, scales = dynamic_mxfp4_quant(w.reshape(-1, cols))
    return (
        packed.view(torch.uint8).view(e, rows, -1),
        scales.view(torch.uint8).view(e, rows, -1)[..., : cols // 32].contiguous(),
    )


def _rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got.float() - ref.float()).norm() / ref.float().norm()).item()


def _run_rank(device: torch.device, num_tokens_list: list[int]) -> None:
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )
    from vllm.models.muse_spark.amd.muse_moe_runner import (
        MuseSparkTopKWeightAndReduce,
    )
    from vllm.models.muse_spark.common.layers import (
        MuseSpark1xRMSNormWithInputScale,
    )

    rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    generator = torch.Generator(device=device)
    generator.manual_seed(1234 + rank)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(
            *shape, device=device, dtype=torch.bfloat16, generator=generator
        )

    w13_raw, w13_scale = _quantize_mxfp4(randn(_E, 2 * _I_PER_RANK, _H) * 0.05)
    w2_raw, w2_scale_raw = _quantize_mxfp4(randn(_E, _H, _I_PER_RANK) * 0.05)

    # The re-sharded rows must equal this rank's rows of the gathered slices.
    cols = _H // tp_size
    for raw in (w2_raw, w2_scale_raw):
        full = tensor_model_parallel_all_gather(raw.contiguous(), dim=-1)
        expected = full[:, rank * cols : (rank + 1) * cols]
        resharded = column_sharded_moe._all_to_all_rows(raw)
        torch.testing.assert_close(resharded, expected, rtol=0, atol=0)
    weights = column_sharded_moe.ColumnShardedDownWeights.from_checkpoint(
        w2_raw, w2_scale_raw
    )
    assert weights.w2.shape[1] == cols

    fp4 = torch.float4_e2m1fn_x2
    w13_scale = e8m0_shuffle(w13_scale.reshape(_E * 2 * _I_PER_RANK, -1)).view(
        _E, 2 * _I_PER_RANK, -1
    )
    w2_scale = e8m0_shuffle(w2_scale_raw.reshape(_E * _H, -1)).view(_E, _H, -1)
    w13, w2 = rocm_aiter_ops.shuffle_weights(w13_raw.view(fp4), w2_raw.view(fp4))
    w13.is_shuffled = w2.is_shuffled = True

    norm = MuseSpark1xRMSNormWithInputScale(
        hidden_size=_H,
        eps=1e-5,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
        residual_id=None,
        output_norm_gain_center_type="zero",
    ).to(device)
    torch.manual_seed(7)
    norm.weight.data.normal_(0, 0.1)

    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    for num_tokens in num_tokens_list:
        # Every rank sees the same batch, as under TP.
        torch.manual_seed(num_tokens)
        hidden_states = torch.randn(
            num_tokens * _TOP_K, _H, device=device, dtype=torch.bfloat16
        )
        topk_ids = torch.randint(
            0, _E, (num_tokens, _TOP_K), device=device, dtype=torch.int32
        )
        topk_weights = torch.rand(num_tokens, _TOP_K, device=device)
        flat_ids = topk_ids.view(-1, 1)
        unit = torch.ones(flat_ids.shape, device=device, dtype=torch.float32)

        partial = fused_moe(
            hidden_states,
            w13,
            w2,
            unit,
            flat_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
        )
        ref_routes = tensor_model_parallel_all_reduce(partial.float())
        out = torch.empty(
            num_tokens * _TOP_K, cols, device=device, dtype=torch.bfloat16
        )
        got = column_sharded_moe.column_sharded_fused_experts(
            hidden_states,
            flat_ids,
            unit,
            w13,
            w2,
            w13_scale,
            w2_scale,
            weights,
            out=out,
        )
        assert got.shape[-1] == cols, f"no covered stage2 at T={num_tokens}"
        # Same quantized inputs and weights, so the paths differ only in
        # rounding: the reference rounds each rank's partial sum to bf16 (at
        # most 2^-8 of its magnitude), this path rounds the full sum once. The
        # bound has a 2x margin for the fp32 accumulation order.
        own = slice(rank * cols, (rank + 1) * cols)
        abs_sum = tensor_model_parallel_all_reduce(partial.float().abs())[:, own]
        diff = (got.float() - ref_routes[:, own]).abs()
        bound = 2.0**-7 * (abs_sum + ref_routes[:, own].abs()) + 1e-6
        worst = (diff / bound).max().item()
        assert worst <= 1.0, f"T={num_tokens}: route diff {worst:.3g}x bound"

        finalize = MuseSparkTopKWeightAndReduce(norm.forward, pre_norm_all_reduce=True)
        expected = finalize.apply(
            None,
            partial.view(num_tokens, _TOP_K, _H),
            topk_weights,
            topk_ids,
            apply_router_weight_on_input=False,
        )
        routes = column_sharded_moe.ColumnShardedRoutes(
            got.view(num_tokens, _TOP_K, cols)
        )
        finalize.take_route_output = lambda routes=routes: routes
        actual = finalize.apply(
            None,
            out.view(num_tokens, _TOP_K, cols),
            topk_weights,
            topk_ids,
            apply_router_weight_on_input=False,
        )
        assert actual.shape == (num_tokens, _H)
        err = _rel_err(actual, expected)
        assert err < 1e-2, f"T={num_tokens}: output error {err:.3g}"
        # Every rank must hold the same output.
        first = tensor_model_parallel_all_gather(actual[None], dim=0)[0]
        torch.testing.assert_close(actual, first, rtol=0, atol=0)


def _worker(rank: int, world_size: int, port: int, num_tokens: list[int]) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.utils.system_utils import update_environment_variables

    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(device)
    update_environment_variables(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
            "VLLM_ROCM_USE_AITER": "1",
        }
    )
    try:
        with set_current_vllm_config(VllmConfig()), torch.inference_mode():
            init_distributed_environment()
            initialize_model_parallel(tensor_model_parallel_size=world_size)
            _run_rank(device, num_tokens)
    finally:
        cleanup_dist_env_and_memory()
    # RCCL / c10d threads can still be winding down after the cleanup; a normal
    # interpreter exit then runs C++ static destructors under them and aborts
    # ("terminate called without an active exception"). Skip them, as
    # tests/kernels/moe/test_moe_layer.py does. Failures raise above instead.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


@requires_rocm
@pytest.mark.skipif(
    not column_sharded_moe._aiter_supports_stage2_override(),
    reason="AITER without fused_moe's stage2 override",
)
@pytest.mark.parametrize("world_size", [2, 8])
@pytest.mark.distributed(num_gpus=8)
def test_column_sharded_down_matches_reduce_scatter(world_size: int) -> None:
    """Experts plus finalize against the AITER experts and the reduce-scatter
    finalize, at the batch sizes that take the path in serving."""
    if not current_platform.supports_mx():
        pytest.skip("MXFP4 MoE requires CDNA4")
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    from vllm.utils.network_utils import get_open_port

    mp.spawn(
        _worker,
        args=(world_size, get_open_port(), [1024, 1031, 2048, 4096, 8192]),
        nprocs=world_size,
        join=True,
    )
