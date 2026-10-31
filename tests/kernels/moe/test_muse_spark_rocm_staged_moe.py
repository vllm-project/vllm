# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x ROCm MXFP4 experts through AITER's two-stage MoE API.

On one GPU, the native-route experts are checked bit for bit against the
repeated-input experts, and the column-sharded finalize kernels against the
full-width fused finalize. On 2 and 8 GPUs, the column-sharded down projection
(re-sharded ``w2``, gathered stage2 input, column finalize) is checked against
the AITER experts followed by the reduce-scatter finalize.
"""

import os
import sys
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

import vllm.models.muse_spark.amd.muse_moe_runner as amd_muse_moe_runner
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    determine_expert_map,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_quant_config,
)
from vllm.models.muse_spark.amd import staged_moe
from vllm.models.muse_spark.amd.muse_moe_runner import (
    MuseSparkAiterExperts,
    MuseSparkQuarkOCPMXMoEQuantConfig,
)
from vllm.models.muse_spark.common.ops.fused_finalize import (
    fused_post_expert_finalize,
)
from vllm.platforms import current_platform

ROCM_GFX950 = False
ROCM_AITER_SUPPORTED = False
if current_platform.is_rocm():
    from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
    from vllm.platforms.rocm import on_gfx950

    ROCM_GFX950 = on_gfx950()
    ROCM_AITER_SUPPORTED = is_aiter_found_and_supported()

requires_staged_aiter = pytest.mark.skipif(
    not (ROCM_GFX950 and ROCM_AITER_SUPPORTED and staged_moe.aiter_has_staged_moe()),
    reason="requires a GFX950 GPU and AITER with the two-stage MoE API",
)

# One rank of Muse Spark Open: 256 experts, top 8, hidden 4096, intermediate
# 4096, under TP8 (intermediate / 8 per rank) or EP8 (32 experts per rank).
NUM_EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, WORLD = 256, 8, 4096, 4096, 8


def _quant_config_fields(config: FusedMoEQuantConfig) -> dict:
    return {
        f.name: getattr(config, f.name) for f in fields(FusedMoEQuantConfig) if f.init
    }


def _rms_norm(x: torch.Tensor, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    variance = x.float().square().mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + 1e-6)).to(out_dtype or x.dtype)


def _post_expert_norm() -> torch.nn.Module:
    """A post-expert norm the fused (EP-masking) finalize supports."""
    from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNormWithInputScale

    norm = MuseSpark1xRMSNormWithInputScale(
        hidden_size=HIDDEN,
        eps=1e-8,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
        residual_id=None,
        output_norm_gain_center_type="zero",
    )
    with torch.no_grad():
        norm.weight.normal_(0.0, 0.1)
    return norm.cuda()


def _moe_config(ep: bool, device: str = "cuda") -> FusedMoEConfig:
    size = WORLD
    parallel_config = FusedMoEParallelConfig(
        tp_size=1 if ep else size,
        pcp_size=1,
        dp_size=1,
        ep_size=size if ep else 1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=ep,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    return FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=TOP_K,
        hidden_dim=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_local_experts=NUM_EXPERTS // parallel_config.ep_size,
        num_logical_experts=NUM_EXPERTS,
        activation=MoEActivation.SILU,
        device=torch.device(device),
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel_config,
        in_dtype=torch.bfloat16,
        moe_backend="auto",
    )


def _mxfp4_experts(num_experts: int, intermediate: int, post_expert_norm_fn):
    """Random MXFP4 experts in the AITER layout and their quant config."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    def quantize(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.randn(num_experts * rows, cols, device="cuda", dtype=torch.bfloat16)
        packed, scale = dynamic_mxfp4_quant(w * cols**-0.5)
        return packed.view(num_experts, rows, -1), scale.view(num_experts, rows, -1)

    w13, w13_scale = quantize(2 * intermediate, HIDDEN)
    w2, w2_scale = quantize(HIDDEN, intermediate)
    layer = SimpleNamespace(w13_input_scale=None, w2_input_scale=None)
    w13, w2, w13_scale, w2_scale, _, _ = (
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
            mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
            layer=layer,
            w13_weight=w13,
            w2_weight=w2,
            w13_weight_scale=w13_scale,
            w2_weight_scale=w2_scale,
        )
    )
    quant_config = make_mxfp4_moe_quant_config(
        mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
    )
    assert quant_config is not None
    return (
        w13,
        w2,
        MuseSparkQuarkOCPMXMoEQuantConfig(
            **_quant_config_fields(quant_config),
            post_expert_norm_fn=post_expert_norm_fn,
        ),
    )


@pytest.fixture(scope="module", params=["tp", "ep"])
def rank(request):
    """One rank's experts; a staged and a repeated-input-only instance."""
    if not (ROCM_GFX950 and ROCM_AITER_SUPPORTED):
        pytest.skip("requires a GFX950 GPU with supported AITER")
    if not staged_moe.aiter_has_staged_moe():
        pytest.skip("requires AITER with the two-stage MoE API")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ROCM_USE_AITER", "1")
        mp.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
        rocm_aiter_ops.refresh_env_variables()
        torch.manual_seed(0)
        ep = request.param == "ep"
        config = _moe_config(ep)
        w13, w2, quant_config = _mxfp4_experts(
            config.num_local_experts,
            config.intermediate_size_per_partition,
            _post_expert_norm().forward,
        )
        expert_map = expert_mask = None
        if ep:
            _, expert_map, expert_mask = determine_expert_map(
                WORLD, 0, NUM_EXPERTS, return_expert_mask=True
            )
            assert expert_map is not None and expert_mask is not None
            expert_map, expert_mask = expert_map.cuda(), expert_mask.cuda()
        native = MuseSparkAiterExperts(config, quant_config)
        native.process_weights_after_loading(
            SimpleNamespace(w13_weight=w13, w2_weight=w2, _expert_map=expert_map)
        )
        expanded = MuseSparkAiterExperts(config, quant_config)
        yield SimpleNamespace(
            ep=ep,
            w13=w13,
            w2=w2,
            native=native,
            expanded=expanded,
            expert_map=expert_map,
            expert_mask=expert_mask,
        )
    rocm_aiter_ops.refresh_env_variables()


def _batch(num_tokens: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states = torch.randn(
        num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=gen
    )
    logits = torch.rand(num_tokens, NUM_EXPERTS, device="cuda", generator=gen)
    topk_logits, topk_ids = logits.topk(TOP_K, dim=-1)
    return hidden_states, torch.softmax(topk_logits, dim=-1), topk_ids.to(torch.int32)


def _apply(r, experts, hidden_states, topk_weights, topk_ids, routes) -> None:
    experts.apply(
        routes,
        hidden_states,
        r.w13,
        r.w2,
        topk_weights,
        topk_ids,
        MoEActivation.SILU,
        NUM_EXPERTS,
        r.expert_mask,
        None,
        None,
        torch.empty(0, device="cuda"),
        torch.empty(0, device="cuda"),
        None,
        False,
    )


def _finalize(experts, topk_weights, topk_ids, routes) -> torch.Tensor:
    reducer = experts.finalize_weight_and_reduce_impl()
    assert reducer.fused_norm is not None
    # TP's pre-norm all-reduce is a no-op on one rank.
    reducer.pre_norm_all_reduce = False
    out = torch.empty(routes.shape[0], HIDDEN, device="cuda", dtype=torch.bfloat16)
    return reducer.apply(out, routes, topk_weights, topk_ids, False)


# TP: 1024/2048/32768-row keys use tuned FlyDSL v2 stage2s, the 512/8192-row
# keys the heuristic v1 stage2. EP: every batch size runs staged.
@pytest.mark.parametrize("num_tokens", [1, 16, 64, 65, 128, 256, 1000, 4096])
@torch.inference_mode()
def test_staged_matches_expanded_bitwise(rank, num_tokens: int) -> None:
    r = rank
    if not r.native.uses_staged_moe(num_tokens):
        assert not r.ep and num_tokens < 65
        pytest.skip("batch size stays on the expanded input under TP")
    hidden_states, topk_weights, topk_ids = _batch(num_tokens, num_tokens)
    shape = (num_tokens, TOP_K, HIDDEN)
    expanded = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    _apply(r, r.expanded, hidden_states, topk_weights, topk_ids, expanded)
    expanded = r.expanded._take_route_output().clone()
    expected = _finalize(r.expanded, topk_weights, topk_ids, expanded).clone()

    # Unwritten (non-local EP) rows keep the NaN and must never be read.
    native = torch.full(shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    _apply(r, r.native, hidden_states, topk_weights, topk_ids, native)
    assert r.native._route_output is native
    actual = _finalize(r.native, topk_weights, topk_ids, native)

    local = torch.ones_like(topk_ids, dtype=torch.bool)
    if r.ep:
        local = r.expert_map[topk_ids.long()] >= 0
        assert native[~local].isnan().all()
        assert (expanded[~local] == 0).all()
    torch.testing.assert_close(native[local], expanded[local], atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_staged_graph_replay(rank) -> None:
    r = rank
    num_tokens = 128
    hidden_states, topk_weights, topk_ids = _batch(num_tokens, 7)
    routes = torch.empty(num_tokens, TOP_K, HIDDEN, device="cuda", dtype=torch.bfloat16)

    def run() -> torch.Tensor:
        _apply(r, r.native, hidden_states, topk_weights, topk_ids, routes)
        return _finalize(r.native, topk_weights, topk_ids, routes)

    for _ in range(2):
        run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()

    hidden_states.copy_(torch.flip(hidden_states, dims=[0]))
    graph.replay()
    torch.accelerator.synchronize()
    replayed = captured.clone()
    torch.testing.assert_close(replayed, run(), atol=0, rtol=0)


def test_staged_without_aiter_api(monkeypatch) -> None:
    monkeypatch.setattr(staged_moe, "aiter_has_staged_moe", lambda: False)
    w = torch.empty(0)
    assert not staged_moe.staged_moe_supported(4096, TOP_K, w, w, is_ep=True)


@pytest.mark.parametrize("ep", [False, True])
def test_staged_batch_sizes(monkeypatch, ep: bool) -> None:
    """TP decode sizes and unsupported AITER configs keep the expanded input."""
    supported = {1: False, 64: True, 65: True, 4096: False}
    monkeypatch.setattr(
        amd_muse_moe_runner,
        "staged_moe_supported",
        lambda num_tokens, *args, **kwargs: supported[num_tokens],
    )
    config = _moe_config(ep, device="cpu")
    quant_config = MuseSparkQuarkOCPMXMoEQuantConfig(
        **_quant_config_fields(FUSED_MOE_UNQUANTIZED_CONFIG),
        post_expert_norm_fn=_rms_norm,
    )
    layer = SimpleNamespace(
        w13_weight=torch.empty(0),
        w2_weight=torch.empty(0),
        _expert_map=torch.full((NUM_EXPERTS,), -1, dtype=torch.int32),
    )

    experts = MuseSparkAiterExperts(config, quant_config)
    assert not experts.uses_staged_moe(65)
    experts.process_weights_after_loading(layer)
    assert [experts.uses_staged_moe(t) for t in supported] == [
        False,
        ep,
        True,
        False,
    ]


requires_rocm = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="MuseSpark ROCm kernels"
)


def _deployment_config(
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
    monkeypatch.setattr(staged_moe, "aiter_has_staged_moe", lambda: True)
    monkeypatch.setenv("VLLM_ROCM_MUSE_MOE_COLUMN_SHARDED_DOWN", "1")
    config = _deployment_config(**kwargs)
    assert staged_moe.column_sharded_down_supported(config) is expected
    monkeypatch.setenv("VLLM_ROCM_MUSE_MOE_COLUMN_SHARDED_DOWN", "0")
    assert not staged_moe.column_sharded_down_supported(config)


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
        resharded = staged_moe._all_to_all_rows(raw)
        torch.testing.assert_close(resharded, expected, rtol=0, atol=0)
    weights = staged_moe.ColumnShardedDownWeights.from_checkpoint(w2_raw, w2_scale_raw)
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
        hidden_states = torch.randn(num_tokens, _H, device=device, dtype=torch.bfloat16)
        # Distinct experts per token, as the router selects them.
        topk_ids = torch.rand(num_tokens, _E, device=device).topk(_TOP_K).indices
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = torch.rand(num_tokens, _TOP_K, device=device)
        flat_ids = topk_ids.view(-1, 1)
        unit = torch.ones(flat_ids.shape, device=device, dtype=torch.float32)

        partial = fused_moe(
            hidden_states.repeat_interleave(_TOP_K, dim=0),
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
        out = torch.empty(num_tokens, _TOP_K, _H, device=device, dtype=torch.bfloat16)
        routes = staged_moe.staged_fused_moe(
            hidden_states,
            w13,
            w2,
            topk_ids,
            unit.view(num_tokens, _TOP_K),
            w13_scale,
            w2_scale,
            out,
            column_weights=weights,
        )
        assert isinstance(routes, staged_moe.ColumnShardedRoutes), (
            f"no column-sharded stage2 at T={num_tokens}"
        )
        got = routes.routes.reshape(num_tokens * _TOP_K, cols)
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
        finalize.take_route_output = lambda routes=routes: routes
        actual = finalize.apply(
            None,
            out,
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


@requires_staged_aiter
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
