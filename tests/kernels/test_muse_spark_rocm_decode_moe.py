# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark ROCm decode MoE kernels."""

import functools

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    determine_expert_map,
)
from vllm.models.muse_spark.amd.muse_moe_runner import MuseSparkAiterExperts
from vllm.models.muse_spark.amd.ops import decode_moe as dmoe
from vllm.platforms import current_platform

from .moe.test_muse_spark_rocm_bf16_moe import (
    _moe_config,
    _mxfp4_weights,
    _parallel_config,
    _route_launcher,
)

NUM_EXPERTS = 256
TOP_K = 8
HIDDEN = 4096
ROUTE_SCALE = 2.5
DIVISION_EPS = 1e-15
NORM_EPS = 1e-5


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires a ROCm GPU")
@pytest.mark.parametrize("num_tokens", [1, 7, 16, 48, 128, 513, 2047])
@pytest.mark.parametrize("routing", ["random", "ties"])
@pytest.mark.parametrize("split_k_partials", [True, False])
@torch.inference_mode()
def test_decode_moe_prep_is_bitwise_the_unfused_kernels(
    num_tokens: int, routing: str, split_k_partials: bool
) -> None:
    """One prep launch, from the router's split-K partials or its logits, picks
    the experts of the router split reduce + routing kernel of the per-route
    path, with its weights, and the rows of its pre-expert norm kernel."""
    from vllm.models.muse_spark.amd.ops.fused_norm_expand import fused_rmsnorm_expand
    from vllm.models.muse_spark.amd.ops.router_bf16x3 import (
        router_bf16x3_gemm,
        split_fp32_to_bf16x3,
    )
    from vllm.models.muse_spark.common.ops.fused_routing import (
        fused_muse_spark_routing,
    )

    torch.manual_seed(num_tokens)
    device = torch.device("cuda")
    router_weight = torch.randn(NUM_EXPERTS, 8192, device=device) * 0.02
    planes = split_fp32_to_bf16x3(router_weight)
    x = torch.randn(num_tokens, 8192, device=device, dtype=torch.bfloat16)
    pooled = torch.randn(num_tokens, HIDDEN, device=device, dtype=torch.bfloat16)
    norm_weight = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16) * 0.1
    bias = torch.randn(NUM_EXPERTS, device=device) * 0.01
    if routing == "ties":
        x.zero_()
        bias = torch.randint(0, 4, (NUM_EXPERTS,), device=device).float() * 0.01

    weights, ids, normed = dmoe.decode_moe_prep(
        router_bf16x3_gemm(x, planes, reduce_splits=not split_k_partials),
        bias,
        TOP_K,
        ROUTE_SCALE,
        None,
        DIVISION_EPS,
        pooled,
        norm_weight,
        NORM_EPS,
        1.0,
        False,
    )
    ref_weights, ref_ids = fused_muse_spark_routing(
        router_bf16x3_gemm(x, planes),
        bias,
        TOP_K,
        ROUTE_SCALE,
        router_output_multiplier=None,
        router_score_normalization="sigmoid",
        router_score_division_eps=DIVISION_EPS,
    )
    ref_normed = fused_rmsnorm_expand(pooled, norm_weight, NORM_EPS, 1.0, False, TOP_K)
    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(weights, ref_weights)
    torch.testing.assert_close(normed, ref_normed)


ROCM_GFX950_AITER = False
if current_platform.is_rocm():
    from vllm._aiter_ops import is_aiter_found_and_supported
    from vllm.platforms.rocm import on_gfx950

    ROCM_GFX950_AITER = on_gfx950() and is_aiter_found_and_supported()

STAGE1 = "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2"
STAGE2 = "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic_bnt2"


def _metadata(kind: str):
    """AITER MoE metadata as `get_2stage_cfgs` builds it for each kind of
    config a tuned row can select."""
    import aiter.fused_moe as fm

    flydsl1 = functools.partial(fm._flydsl_stage1_wrapper, kernelName=STAGE1)
    flydsl2 = functools.partial(fm._flydsl_stage2_wrapper, kernelName=STAGE2)
    if kind == "flydsl":
        return fm.MOEMetadata(flydsl1, flydsl2, 32, 0)
    if kind in ("fp4", "fp8"):
        fused = functools.partial(
            fm._flydsl_stage1_wrapper, kernelName=f"{STAGE1}_{kind}"
        )
        return fm.MOEMetadata(fused, flydsl2, 32, 0, fuse_quant=kind)
    if kind == "cktile_stage2":
        return fm.MOEMetadata(flydsl1, functools.partial(fm.cktile_moe_stage2), 32, 0)
    if kind == "one_stage":
        stage = functools.partial(fm.fused_moe_1stage, kernelName="fmoe_g1u1_fp4")
        return fm.MOEMetadata(stage, None, 32, 0, True)
    assert kind == "whole_graph"
    return fm.MOEMetadata(None, None, 32, 0)


@pytest.mark.skipif(not ROCM_GFX950_AITER, reason="requires gfx950 and AITER")
@pytest.mark.parametrize(
    "kind", ["flydsl", "fp4", "fp8", "cktile_stage2", "one_stage", "whole_graph"]
)
def test_decode_moe_launches_only_flydsl_stages_with_bf16_output(kind: str) -> None:
    """Only a FlyDSL MXFP4 stage pair with bf16 stage-1 rows takes the decode
    MoE; fused-quant stage 1 (quantized rows and scales), other stage-2
    kernels, one-stage and whole-graph configs leave the call to the per-route
    path instead of failing."""
    from vllm.models.muse_spark.amd import decode_moe

    kernels = decode_moe._flydsl_stage_kernels(_metadata(kind))
    if kind == "flydsl":
        assert kernels is not None
        assert (kernels.stage1, kernels.block_m) == (STAGE1, 32)
        assert kernels.stage2["return_per_slot"]
    else:
        assert kernels is None


@pytest.mark.skipif(not ROCM_GFX950_AITER, reason="requires gfx950 and AITER")
@pytest.mark.parametrize("local_experts", [32, 64])
def test_decode_moe_takes_aiters_flydsl_stages_or_falls_back(
    local_experts: int,
) -> None:
    """For Muse Spark's EP shapes the installed AITER's configs give either
    FlyDSL stages this path launches or the per-route path."""
    from vllm.models.muse_spark.amd import decode_moe

    for num_tokens in [1, 8, 64, 256, 512, 1024, decode_moe.MAX_TOKENS]:
        kernels = decode_moe._stage_kernels(
            num_tokens, TOP_K, local_experts, HIDDEN, HIDDEN
        )
        assert kernels is None or kernels.stage1.startswith("flydsl_moe1_")


def _per_route_and_native(
    pooled: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One EP4 rank (rank 1: experts [64, 128)) at Muse Spark shapes: the per-route
    AITER experts on the repeated pre-expert-normed rows with the per-route
    finalize, and the decode MoE's experts on the T normed rows with its finalize.
    Returns `(per_route, native)` `[T, H]` outputs."""
    from vllm.models.muse_spark.amd import decode_moe
    from vllm.models.muse_spark.amd.ops.fused_norm_expand import fused_rmsnorm_expand
    from vllm.models.muse_spark.common.layers import MuseSparkRMSNormWithInputScale

    num_tokens, hidden_size = pooled.shape
    top_k = topk_ids.shape[1]
    device, dtype = pooled.device, pooled.dtype
    local, expert_map, expert_mask = determine_expert_map(
        ep_size=4, ep_rank=1, global_num_experts=NUM_EXPERTS, return_expert_mask=True
    )
    assert expert_map is not None and expert_mask is not None
    expert_map, expert_mask = expert_map.to(device), expert_mask.to(device)
    w13, w2, quant_config = _mxfp4_weights(
        torch.randn(local, 2 * hidden_size, hidden_size, dtype=dtype, device=device)
        / hidden_size**0.5,
        torch.randn(local, hidden_size, hidden_size, dtype=dtype, device=device)
        / hidden_size**0.5,
    )
    post_norm = MuseSparkRMSNormWithInputScale(
        hidden_size=hidden_size,
        eps=1e-8,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
        residual_id=None,
        output_norm_gain_center_type="zero",
    ).to(device)
    post_norm.weight.copy_(torch.randn(hidden_size, device=device) * 0.1)
    quant_config.post_expert_norm_fn = post_norm.forward
    config = _moe_config(
        parallel_config=_parallel_config(ep_size=4),
        num_experts=NUM_EXPERTS,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=hidden_size,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
    )
    experts = MuseSparkAiterExperts(config, quant_config)
    norm_weight = torch.randn(hidden_size, dtype=dtype, device=device) * 0.1

    route_buffer, launch = _route_launcher(
        experts,
        fused_rmsnorm_expand(pooled, norm_weight, NORM_EPS, 1.0, False, top_k),
        w13,
        w2,
        topk_weights,
        topk_ids,
        intermediate_size=hidden_size,
        num_experts=NUM_EXPERTS,
        num_local_experts=local,
        expert_map=expert_mask,
    )
    launch()
    per_route = experts.finalize_weight_and_reduce_impl().apply(
        torch.empty_like(pooled), route_buffer, topk_weights, topk_ids, False
    )

    kernels = decode_moe._stage_kernels(
        num_tokens, top_k, local, hidden_size, hidden_size
    )
    assert kernels is not None
    native = decode_moe.native_routes_moe(
        fused_rmsnorm_expand(pooled, norm_weight, NORM_EPS, 1.0, False, 1),
        topk_weights,
        topk_ids,
        w13,
        w2,
        quant_config,
        expert_mask,
        expert_map,
        kernels,
        post_norm,
    )
    return per_route, native


@pytest.fixture
def rocm_aiter_moe(monkeypatch: pytest.MonkeyPatch):
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    rocm_aiter_ops.refresh_env_variables()
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


@pytest.mark.skipif(not ROCM_GFX950_AITER, reason="requires gfx950 and AITER")
@pytest.mark.parametrize("num_tokens", [1, 8, 48, 128, 512, 513, 2047])
@torch.inference_mode()
def test_native_routes_match_the_per_route_path(
    rocm_aiter_moe, num_tokens: int
) -> None:
    """The decode MoE's experts and finalize give the per-route path's output."""
    torch.manual_seed(num_tokens)
    device = torch.device("cuda")
    pooled = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=device)
    topk_ids = torch.topk(
        torch.randn(num_tokens, NUM_EXPERTS, device=device), TOP_K, dim=-1
    ).indices.to(torch.int32)
    topk_weights = torch.softmax(torch.randn(num_tokens, TOP_K, device=device), -1)
    per_route, native = _per_route_and_native(pooled, topk_ids, topk_weights)
    assert torch.isfinite(native).all()
    torch.testing.assert_close(native, per_route)


@pytest.mark.skipif(not ROCM_GFX950_AITER, reason="requires gfx950 and AITER")
@pytest.mark.parametrize("num_tokens", [2, 37, 64, 256, 512])
@torch.inference_mode()
def test_native_routes_with_repeated_experts_and_nan_rows(
    rocm_aiter_moe, num_tokens: int
) -> None:
    """Tokens selecting one expert in several slots and a NaN row, as CUDA-graph
    padding rows with non-finite router logits produce: every other token's
    output is the per-route path's."""
    torch.manual_seed(num_tokens)
    device = torch.device("cuda")
    pooled = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=device)
    pooled[-1] = float("nan")
    topk_ids = torch.topk(
        torch.randn(num_tokens, NUM_EXPERTS, device=device), TOP_K, dim=-1
    ).indices.to(torch.int32)
    topk_ids[0] = torch.arange(64, 64 + TOP_K)
    topk_ids[3::3] = topk_ids[3::3, :1]
    topk_ids[1::3, 5] = topk_ids[1::3, 2]
    topk_weights = torch.softmax(torch.randn(num_tokens, TOP_K, device=device), -1)
    per_route, native = _per_route_and_native(pooled, topk_ids, topk_weights)
    slots = torch.arange(TOP_K, device=device)
    repeats = (
        (topk_ids[:, None, :] == topk_ids[:, :, None])
        & (slots[None, :] != slots[:, None])
    ).any(dim=(1, 2))
    compared = ~repeats & pooled.isfinite().all(dim=1)
    assert compared[0]
    torch.testing.assert_close(native[compared], per_route[compared])


LOCAL_EXPERTS = 64
SORT_BLOCK_M = 32
SORT_QUANT_TOKENS = [*range(1, 65), 96, 127, 128, 129, 200, 256, 257, 384, 511, 512]
# AITER's HIP quant floors a 32-value block's amax at 1e-10 (scale 2^-35),
# `sort_quant` (AITER's Triton quant) at the smallest normal scale, so blocks
# below 1e-10 get other scales; both dequantize within 6 * 2^-35 of each other.
SCALE_FLOOR = 1e-10
SCALE_FLOOR_ATOL = 6 * 2**-35
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _ep_maps(
    placement: str, rank: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """One EP4 rank's `[E]` global-to-local expert map and AITER's `[E + 1]`
    expert mask."""
    _, expert_map, expert_mask = determine_expert_map(
        ep_size=4,
        ep_rank=rank,
        global_num_experts=NUM_EXPERTS,
        expert_placement_strategy=placement,
        return_expert_mask=True,
    )
    assert expert_map is not None and expert_mask is not None
    return expert_map.to(device), expert_mask.to(device)


def _route_ids(
    num_tokens: int, kind: str, mask: torch.Tensor, g: torch.Generator
) -> torch.Tensor:
    """`[T, TOP_K]` routes; "repeat": tokens selecting an expert in several slots,
    as rows with non-finite router logits (CUDA-graph padding) do."""
    device = mask.device
    pool = NUM_EXPERTS
    if kind == "local":
        pool_ids = torch.nonzero(mask[:-1]).flatten()
        pool = pool_ids.numel()
    elif kind == "skewed":
        pool_ids = torch.randperm(NUM_EXPERTS, device=device, generator=g)[: TOP_K + 4]
        pool = pool_ids.numel()
    ids = torch.stack(
        [
            torch.randperm(pool, device=device, generator=g)[:TOP_K]
            for _ in range(num_tokens)
        ]
    )
    if kind in ("local", "skewed"):
        ids = pool_ids[ids]
    u = torch.rand(num_tokens, device=device, generator=g)
    if kind == "padding":
        ids[u < 0.3] = -1
    elif kind == "repeat":
        ids[u < 0.2] = ids[u < 0.2][:, :1]
        pair = (u >= 0.2) & (u < 0.5)
        ids[pair, 3] = ids[pair, 1]
        ids[pair, 7] = ids[pair, 1]
        ids[u >= 0.8] = 0
    return ids.to(torch.int32).contiguous()


def _rows(num_tokens: int, g: torch.Generator, device: torch.device) -> torch.Tensor:
    """bf16 rows over the exponent range, with zero, huge, tiny and NaN rows."""
    x = torch.randn(num_tokens, HIDDEN, device=device, generator=g)
    kind = num_tokens % 4
    if kind == 1:
        scale = torch.randint(-130, 125, (num_tokens, 1), device=device, generator=g)
        x = x * torch.exp2(scale.float())
    elif kind == 2:
        x[: num_tokens // 2] = 0
        if num_tokens > 2:
            x[num_tokens // 2] = 3.0e38 * torch.sign(x[num_tokens // 2])
            x[-1] = x[-1] * 1e-38
    elif kind == 3:
        x[torch.rand(num_tokens, HIDDEN, device=device, generator=g) < 0.05] = float(
            "nan"
        )
        x[: max(1, num_tokens // 4)] = float("nan")
    return x.to(torch.bfloat16).contiguous()


def _aiter_sort_quant(
    x: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    from aiter.fused_moe import moe_sorting
    from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort_hip

    num_tokens = x.shape[0]
    unused_weights = torch.empty(ids.shape, dtype=torch.float32, device=x.device)
    sorted_ids, _, sorted_eids, num_valid, _ = moe_sorting(
        ids,
        unused_weights,
        mask.numel(),
        HIDDEN,
        torch.bfloat16,
        SORT_BLOCK_M,
        mask,
        None,
        0,
        accumulate=False,
    )[:5]
    a1 = torch.empty(
        (num_tokens, HIDDEN // 2), dtype=torch.float4_e2m1fn_x2, device=x.device
    )
    scale = torch.empty(
        ((sorted_ids.shape[0] + 31) // 32 * 32, HIDDEN // 32),
        dtype=torch.float8_e8m0fnu,
        device=x.device,
    )
    fused_dynamic_mx_quant_moe_sort_hip(
        a1,
        scale,
        x,
        sorted_ids,
        num_valid,
        num_tokens,
        SORT_BLOCK_M,
        num_tokens * TOP_K,
    )
    return a1, scale, sorted_ids, sorted_eids, num_valid


def _sorted_row_scales(scale: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Scale bytes `[rows, HIDDEN // 32]` from AITER's swizzled e8m0 layout."""
    r = rows[:, None]
    k = torch.arange(HIDDEN // 32, device=rows.device)[None, :]
    n_pad = HIDDEN // 32
    offset = (
        (r // 32 * n_pad) * 32
        + (k // 8) * 256
        + (k % 4) * 64
        + (r % 16) * 4
        + (k % 8) // 4 * 2
        + (r % 32) // 16
    )
    return scale.view(torch.uint8).view(-1)[offset]


def _dequantize(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """`[N, HIDDEN // 2]` packed E2M1 bytes and `[N, HIDDEN // 32]` e8m0 bytes ->
    `[N, HIDDEN]` fp32."""
    nibbles = torch.stack([codes & 0xF, codes >> 4], dim=-1).flatten(1).long()
    lut = torch.tensor(E2M1, device=codes.device)
    values = lut[nibbles & 7] * torch.where(nibbles >= 8, -1.0, 1.0)
    scale = torch.exp2(scales.float() - 127).repeat_interleave(32, dim=1)
    return values * scale


def _assert_sort_quant_matches_aiter(
    x: torch.Tensor, ids: torch.Tensor, expert_map: torch.Tensor, mask: torch.Tensor
) -> None:
    """Every entry the stage kernels read: num_valid, sorted ids / expert ids up to
    num_valid, the a1 rows of tokens with a local route and the scales of every
    valid sorted row. For a token repeating an expert only one slot is kept and
    which one is unspecified in AITER, so those entries compare the token only.
    Blocks under the scale floors compare their dequantized values."""
    num_tokens = x.shape[0]
    ref = _aiter_sort_quant(x, ids, mask)
    new = dmoe.sort_quant(x, ids, expert_map, LOCAL_EXPERTS, mask.numel(), 32)
    for r, n in zip(ref, new):
        assert r.shape == n.shape and r.dtype == n.dtype
    assert torch.equal(ref[4], new[4])
    num_valid = int(ref[4][0])
    ref_ids, new_ids = ref[2][:num_valid], new[2][:num_valid]
    token = ref_ids & 0xFFFFFF
    slots = torch.arange(TOP_K, device=ids.device)
    repeats = (
        (ids[:, None, :] == ids[:, :, None]) & (slots[None, :] != slots[:, None])
    ).any(dim=(1, 2))
    valid = token < num_tokens
    token_only = valid & repeats[token.clamp(max=num_tokens - 1).long()]
    assert torch.equal(
        torch.where(token_only, token, ref_ids),
        torch.where(token_only, new_ids & 0xFFFFFF, new_ids),
    )
    assert torch.equal(
        ref[3][: num_valid // SORT_BLOCK_M], new[3][: num_valid // SORT_BLOCK_M]
    )
    rows = torch.nonzero(valid).flatten()
    if rows.numel() == 0:
        return
    # NaN does not raise a block's amax in either quant.
    amax = x.float().nan_to_num(nan=0.0).abs().view(num_tokens, HIDDEN // 32, 32)
    amax = amax.amax(dim=-1)
    floor = amax < SCALE_FLOOR
    row_floor = floor[token[rows].long()]
    ref_scales = _sorted_row_scales(ref[1], rows)
    new_scales = _sorted_row_scales(new[1], rows)
    assert torch.equal(ref_scales[~row_floor], new_scales[~row_floor])
    tokens = torch.unique(token[rows].long())
    first = torch.zeros(num_tokens, dtype=torch.long, device=x.device)
    first[token[rows].long()] = torch.arange(rows.numel(), device=x.device)
    ref_a1, new_a1 = (a.view(torch.uint8)[tokens] for a in (ref[0], new[0]))
    byte_floor = floor[tokens].repeat_interleave(16, dim=1)
    assert torch.equal(ref_a1[~byte_floor], new_a1[~byte_floor])
    torch.testing.assert_close(
        _dequantize(new_a1, new_scales[first[tokens]]),
        _dequantize(ref_a1, ref_scales[first[tokens]]),
        atol=SCALE_FLOOR_ATOL,
        rtol=0,
        equal_nan=True,
    )


@pytest.mark.skipif(not ROCM_GFX950_AITER, reason="requires gfx950 and AITER")
@pytest.mark.parametrize(
    "placement, rank", [("linear", 0), ("linear", 3), ("round_robin", 1)]
)
@pytest.mark.parametrize(
    "route_kind", ["random", "skewed", "local", "padding", "repeat"]
)
@torch.inference_mode()
def test_sort_quant_matches_the_aiter_sort_and_quant(
    placement: str, rank: int, route_kind: str
) -> None:
    """One `sort_quant` launch writes the buffers of AITER's moe_sorting +
    `fused_dynamic_mx_quant_moe_sort_hip` for T = 1..512: bit for bit, except
    blocks under the quant's scale floor."""
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(0)
    expert_map, mask = _ep_maps(placement, rank, device)
    for num_tokens in SORT_QUANT_TOKENS:
        ids = _route_ids(num_tokens, route_kind, mask, g)
        _assert_sort_quant_matches_aiter(
            _rows(num_tokens, g, device), ids, expert_map, mask
        )


@pytest.mark.skipif(not ROCM_GFX950_AITER, reason="requires gfx950 and AITER")
@torch.inference_mode()
def test_sort_quant_quantizes_every_finite_bf16_like_aiter() -> None:
    """Every finite bf16 value, in 32-value groups of similar magnitude (every
    reachable scale) and shuffled (flush to zero, saturation)."""
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(1)
    bits = torch.arange(-(2**15), 2**15, dtype=torch.int32, device=device)
    values = bits.to(torch.int16).view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    pad = -values.numel() % HIDDEN
    values = torch.cat([values, values[:pad]])
    by_magnitude = values[torch.argsort(values.float().abs(), stable=True)]
    shuffled = values[torch.randperm(values.numel(), device=device, generator=g)]
    expert_map, mask = _ep_maps("linear", 0, device)
    for x in (by_magnitude, shuffled):
        x = x.view(-1, HIDDEN).contiguous()
        ids = _route_ids(x.shape[0], "local", mask, g)
        _assert_sort_quant_matches_aiter(x, ids, expert_map, mask)
