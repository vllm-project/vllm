# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x ROCm MXFP4 experts on the native top-k routes."""

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

import vllm.models.muse_spark.amd.muse_moe_runner as amd_muse_moe_runner
import vllm.models.muse_spark.amd.ops.route_native_moe as route_native_moe
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
from vllm.models.muse_spark.amd.muse_moe_runner import (
    MuseSparkAiterExperts,
    MuseSparkQuarkOCPMXMoEQuantConfig,
)
from vllm.platforms import current_platform

ROCM_GFX950 = False
ROCM_AITER_SUPPORTED = False
if current_platform.is_rocm():
    from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
    from vllm.platforms.rocm import on_gfx950

    ROCM_GFX950 = on_gfx950()
    ROCM_AITER_SUPPORTED = is_aiter_found_and_supported()

requires_aiter_gfx950 = pytest.mark.skipif(
    not (ROCM_GFX950 and ROCM_AITER_SUPPORTED),
    reason="requires a GFX950 GPU with supported AITER",
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
    """One rank's experts; a route-native and an expanded-only instance."""
    if not (ROCM_GFX950 and ROCM_AITER_SUPPORTED):
        pytest.skip("requires a GFX950 GPU with supported AITER")
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
# keys the heuristic v1 stage2. EP: every batch size runs route-native.
@pytest.mark.parametrize("num_tokens", [1, 16, 64, 65, 128, 256, 1000, 4096])
@torch.inference_mode()
def test_route_native_matches_expanded_bitwise(rank, num_tokens: int) -> None:
    r = rank
    if not r.native.uses_route_native(num_tokens):
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
def test_route_native_graph_replay(rank) -> None:
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


@requires_aiter_gfx950
def test_aiter_has_per_route_hooks() -> None:
    assert route_native_moe._aiter_has_per_route_hooks()


def test_route_native_without_aiter_hooks(monkeypatch) -> None:
    monkeypatch.setattr(route_native_moe, "_aiter_has_per_route_hooks", lambda: False)
    w = torch.empty(0)
    assert not route_native_moe.supports_route_native(4096, TOP_K, w, w, is_ep=True)


@pytest.mark.parametrize("ep", [False, True])
def test_route_native_batch_sizes(monkeypatch, ep: bool) -> None:
    """TP decode sizes and unsupported AITER configs keep the expanded input."""
    supported = {1: False, 64: True, 65: True, 4096: False}
    monkeypatch.setattr(
        amd_muse_moe_runner,
        "supports_route_native",
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
    assert not experts.uses_route_native(65)
    experts.process_weights_after_loading(layer)
    assert [experts.uses_route_native(t) for t in supported] == [
        False,
        ep,
        True,
        False,
    ]

    monkeypatch.setenv("VLLM_ROCM_MUSE_ROUTE_NATIVE_MOE", "0")
    disabled = MuseSparkAiterExperts(config, quant_config)
    disabled.process_weights_after_loading(layer)
    assert not disabled.uses_route_native(65)
