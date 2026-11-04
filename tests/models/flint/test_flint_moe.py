# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint's fused NVFP4 W4A16 MoE (vllm/models/flint/ops/moe.py) must follow the
reference rounding rule on both the decode (GEMV) and the grouped-GEMM paths
and route exactly like sigmoid top-k."""

import pytest
import torch

from vllm.models.flint.gguf_nvfp4 import dequantize_nvfp4
from vllm.models.flint.ops import moe
from vllm.platforms import current_platform

E, H, INTER, TOPK = 256, 2048, 512, 6

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="SM100 only",
)


def _nvfp4(rows: int, cols: int, gen: torch.Generator):
    w = torch.randint(
        0, 256, (E, rows, cols // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    # E4M3 scales over ~2^-6..2^2 with all mantissas.
    exp = torch.randint(1, 10, (E, rows, cols // 16), device="cuda", generator=gen)
    man = torch.randint(0, 8, (E, rows, cols // 16), device="cuda", generator=gen)
    s = ((exp << 3) | man).to(torch.uint8).view(torch.float8_e4m3fn)
    gs = torch.rand(E, device="cuda", generator=gen) * 2e-3 + 1e-3
    return w, s, gs


@pytest.fixture(scope="module")
def weights():
    gen = torch.Generator(device="cuda").manual_seed(0)
    w13, s13, gs13 = _nvfp4(2 * INTER, H, gen)
    w2, s2, gs2 = _nvfp4(H, INTER, gen)
    router = (torch.randn(E, H, device="cuda", generator=gen) * 0.02).bfloat16()
    return dict(w13=w13, s13=s13, gs13=gs13, w2=w2, s2=s2, gs2=gs2, router=router)


@pytest.fixture(scope="module")
def flint_weights(weights):
    return moe.FlintMoEWeights.from_modelopt(
        weights["w13"],
        weights["s13"],
        weights["gs13"],
        weights["w2"],
        weights["s2"],
        weights["gs2"],
    )


def _inputs(num_tokens: int, seed: int | None = None):
    gen = torch.Generator(device="cuda").manual_seed(
        num_tokens if seed is None else seed
    )
    x = torch.randn(num_tokens, H, device="cuda", generator=gen)
    x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True))).bfloat16()
    shared = (torch.randn(num_tokens, H, device="cuda", generator=gen) * 0.1).bfloat16()
    return x, shared


def _reference(x, w, ids, tw, shared, bf16_points: bool):
    """MoE on exactly dequantized weights with fp32 GEMMs.

    bf16_points=False: everything in fp32 (returns fp32).
    bf16_points=True: the reference (TRTLLM MoE) rounding rule (returns bf16):
    fp32 gate/up, act = bf16(silu(g) * u), y_e = bf16(acc_e * global_scale_e),
    routed = bf16(sum_e tw_e * y_e) in fp32 (tw already bf16-rounded by the
    router), out = bf16(routed + shared).
    """
    bf = torch.bfloat16
    routed = torch.zeros(x.shape[0], H, device=x.device)
    for e in ids.unique().tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        w13 = dequantize_nvfp4(w["w13"][e], w["s13"][e], torch.tensor(1.0))
        w2 = dequantize_nvfp4(w["w2"][e], w["s2"][e], torch.tensor(1.0))
        gu = (x[tok].float() @ w13.T) * w["gs13"][e]
        g, u = gu[:, :INTER], gu[:, INTER:]
        if bf16_points:
            h = (torch.nn.functional.silu(g) * u).to(bf).float()
            y = ((h @ w2.T) * w["gs2"][e]).to(bf).float()
        else:
            h = torch.nn.functional.silu(g) * u
            y = (h @ w2.T) * w["gs2"][e]
        routed.index_add_(0, tok, tw[tok, slot, None] * y)
    if not bf16_points:
        return routed + shared.float()
    return (routed.to(bf).float() + shared.float()).to(bf)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


def _run(x, router, fw, shared, decode_max: int | None = None):
    """flint_moe with an optionally forced decode/grouped dispatch."""
    old = moe.DECODE_MAX_TOKENS
    if decode_max is not None:
        moe.DECODE_MAX_TOKENS = decode_max
    try:
        counter = moe.make_route_counter("cuda")
        return moe.flint_moe(x, router, fw, TOPK, counter, shared)
    finally:
        moe.DECODE_MAX_TOKENS = old


@torch.inference_mode()
def test_dequant_exact():
    """The E2M1 * E4M3 decode is exact for every (code, scale) pair."""
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(w_ptr, s_ptr, out_ptr, N: tl.constexpr):
        offs = tl.arange(0, N)
        o = moe._dequant8(tl.load(w_ptr + offs), tl.load(s_ptr + offs))
        for j in tl.static_range(8):
            tl.store(out_ptr + offs * 8 + j, o[j])

    # Every non-NaN E4M3 scale applied to one 16-code block holding codes
    # 0..15 (two int32 words).
    scales = torch.arange(256, dtype=torch.uint8, device="cuda")
    scales = scales[(scales & 0x7F) != 0x7F]
    codes = torch.arange(16, dtype=torch.uint8, device="cuda")
    block = codes[0::2] | (codes[1::2] << 4)
    words = block.repeat(scales.numel()).view(torch.int32)
    n = words.numel()
    padded_w = torch.zeros(512, dtype=torch.int32, device="cuda")
    padded_s = torch.zeros(512, dtype=torch.uint8, device="cuda")
    padded_w[:n] = words
    padded_s[:n] = scales.repeat_interleave(2)
    out_dtype = torch.bfloat16 if moe._DEQUANT_BF16 else torch.float16
    padded_out = torch.empty(512 * 8, dtype=out_dtype, device="cuda")
    kernel[(1,)](padded_w, padded_s, padded_out, N=512)
    ref = dequantize_nvfp4(
        block.repeat(scales.numel()).view(-1, 8),
        scales.view(torch.float8_e4m3fn).view(-1, 1),
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        padded_out[: n * 8].float(), ref.flatten(), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "num_tokens",
    [1, 5, moe.ROUTE_GEMV_MAX_TOKENS, moe.ROUTE_GEMV_MAX_TOKENS + 1, 300],
)
@torch.inference_mode()
def test_routing(weights, num_tokens: int):
    x, _ = _inputs(num_tokens)
    counter = moe.make_route_counter("cuda")
    ids, tw = moe.route(x, weights["router"], TOPK, counter)
    if num_tokens > moe.ROUTE_GEMV_MAX_TOKENS:
        # Same cuBLAS bf16 GEMM as the kernel path; only top-k is under test.
        logits = (x @ weights["router"].T).float()
    else:
        logits = (x.float() @ weights["router"].float().T).bfloat16().float()
    ref_s, ref_ids = torch.sigmoid(logits).topk(TOPK, dim=-1)
    torch.testing.assert_close(
        ids.sort(-1).values, ref_ids.int().sort(-1).values, rtol=0, atol=0
    )
    # Renormalized as the reference does, then rounded to bf16.
    ref_tw = ref_s / (ref_s.sum(-1, keepdim=True) + 1e-20)
    ref_tw = ref_tw.bfloat16().float()
    order, ref_order = ids.argsort(-1), ref_ids.argsort(-1)
    got, ref_tw = tw.gather(1, order), ref_tw.gather(1, ref_order)
    # The fp32 sigmoid / sum may differ in the last bit: rare 1-ulp bf16 flips.
    assert (got == ref_tw).float().mean() > 0.99
    torch.testing.assert_close(got, ref_tw, rtol=2**-7, atol=0)


@torch.inference_mode()
def test_topk_ties_pick_lowest_expert():
    logits = torch.zeros(2, E, device="cuda", dtype=torch.bfloat16)
    logits[0, [7, 3, 200, 9, 100, 50, 2]] = 1.0
    logits[1, :] = -1.0
    ids = torch.empty(2, TOPK, dtype=torch.int32, device="cuda")
    tw = torch.empty(2, TOPK, dtype=torch.float32, device="cuda")
    moe._topk_kernel[(2,)](logits, ids, tw, E=E, TOPK=TOPK)
    assert ids[0].tolist() == [2, 3, 7, 9, 50, 100]
    assert ids[1].tolist() == [0, 1, 2, 3, 4, 5]
    torch.testing.assert_close(
        tw, torch.full_like(tw, 1 / TOPK).bfloat16().float(), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "num_tokens",
    [
        1,
        moe.DECODE_MAX_TOKENS - 1,
        moe.DECODE_MAX_TOKENS,
        moe.DECODE_MAX_TOKENS + 1,
        64,
        600,
    ],
)
@torch.inference_mode()
def test_moe_matches_reference(weights, flint_weights, num_tokens: int):
    """Output follows the bf16 rounding rule (bitwise on >= 99% of elements)
    and stays at the bf16 rounding level of the fp32 reference."""
    x, shared = _inputs(num_tokens)
    out = _run(x, weights["router"], flint_weights, shared)
    ids, tw = moe.route(x, weights["router"], TOPK, moe.make_route_counter("cuda"))
    ref16 = _reference(x, weights, ids.long(), tw, shared, bf16_points=True)
    ref32 = _reference(x, weights, ids.long(), tw, shared, bf16_points=False)
    match = (out == ref16).float().mean().item()
    assert match >= 0.99, match
    assert _rel(out, ref32) < 1.25 * _rel(ref16, ref32)
    tok_err = (out.float() - ref32).norm(dim=-1) / ref32.norm(dim=-1)
    assert tok_err.max() < 0.01


@torch.inference_mode()
def test_experts_module_loads_per_expert_tensors(weights, flint_weights):
    """FlintNvfp4Experts assembles per-expert checkpoint tensors correctly and
    repacks only once every distinct tensor is loaded."""
    layer = moe.FlintNvfp4Experts(E, H, INTER, TOPK).cuda()

    def ckpt(experts):
        for e in experts:
            for proj, rows in (
                ("gate_proj", slice(INTER)),
                ("up_proj", slice(INTER, 2 * INTER)),
            ):
                yield f"{e}.{proj}.weight", weights["w13"][e, rows]
                yield f"{e}.{proj}.weight_scale", weights["s13"][e, rows]
                yield f"{e}.{proj}.weight_scale_2", weights["gs13"][e]
            yield f"{e}.down_proj.weight", weights["w2"][e]
            yield f"{e}.down_proj.weight_scale", weights["s2"][e]
            yield f"{e}.down_proj.weight_scale_2", weights["gs2"][e]

    # Duplicates of expert 0 must not count towards completion.
    layer.load_weights(ckpt([0] * E))
    assert layer.weights is None
    layer.load_weights(ckpt(range(E)))
    assert layer.weights is not None and not list(layer.parameters())
    x, shared = _inputs(4)
    torch.testing.assert_close(
        layer(x, weights["router"], shared),
        _run(x, weights["router"], flint_weights, shared),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("num_tokens", [1, 64])
@torch.inference_mode()
def test_shared_expert_on_aux_stream(weights, flint_weights, num_tokens: int):
    """Passing the shared-expert module (run on the auxiliary stream, joined
    by an event) gives the same output as passing its precomputed output."""
    layer = moe.FlintNvfp4Experts(E, H, INTER, TOPK).cuda()
    layer.weights = flint_weights
    for name in list(layer._parameters):
        delattr(layer, name)
    shared_mod = torch.nn.Linear(H, H, bias=False, device="cuda", dtype=torch.bfloat16)
    x, _ = _inputs(num_tokens)
    expected = layer(x, weights["router"], shared_mod(x))
    torch.testing.assert_close(
        layer(x, weights["router"], shared_mod), expected, rtol=0, atol=0
    )


@pytest.fixture(scope="module")
def bf16_weights():
    """BF16 experts as stored by vLLM's FlashInfer TRTLLM BF16 MoE backend,
    plus the checkpoint-order originals."""
    import vllm.model_executor.layers.fused_moe  # noqa: F401  (import order)
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        convert_moe_weights_to_flashinfer_trtllm_block_layout,
    )

    gen = torch.Generator(device="cuda").manual_seed(1)
    w13 = (torch.randn(E, 2 * INTER, H, device="cuda", generator=gen) * 0.03).bfloat16()
    w2 = (torch.randn(E, H, INTER, device="cuda", generator=gen) * 0.03).bfloat16()
    ref = (w13.clone(), w2.clone())
    convert_moe_weights_to_flashinfer_trtllm_block_layout(
        {}, w13, w2, is_gated_act_gemm=True
    )
    return w13, w2, ref


def _trtllm_bf16(x, ids, tw, w13, w2):
    import flashinfer
    from flashinfer.fused_moe import WeightLayout

    from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
        view_as_block_major_k,
    )

    out = flashinfer.fused_moe.trtllm_bf16_routed_moe(
        topk_ids=(ids, tw),
        hidden_states=x,
        gemm1_weights=view_as_block_major_k(w13),
        gemm2_weights=view_as_block_major_k(w2),
        num_experts=E,
        top_k=TOPK,
        n_group=None,
        topk_group=None,
        intermediate_size=INTER,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=None,
        routing_method_type=1,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.BlockMajorK,
        do_finalize=True,
        activation_type=3,  # SwiGLU
    )
    return out[0] if isinstance(out, list) else out


@pytest.mark.parametrize("num_tokens", [1, moe.BF16_DECODE_MAX_TOKENS])
@torch.inference_mode()
def test_bf16_decode_matches_trtllm(weights, bf16_weights, num_tokens: int):
    """The BF16 decode path reads the TRTLLM layout correctly and follows the
    TRTLLM BF16 MoE rounding points: on identical routing it agrees with the
    TRTLLM kernels used for larger batches up to GEMM accumulation order, and
    with an fp32-GEMM reference of the same rule."""
    w13, w2, (w13_ref, w2_ref) = bf16_weights
    maps = moe.trtllm_bf16_row_maps(H, INTER, "cuda")
    x, shared = _inputs(num_tokens)
    counter = moe.make_route_counter("cuda")
    out = moe.flint_moe_bf16_decode(
        x, weights["router"], w13, w2, maps, TOPK, counter, shared
    )
    ids, tw = moe.route(x, weights["router"], TOPK, counter)
    bf = torch.bfloat16
    trt = (_trtllm_bf16(x, ids, tw, w13, w2).float() + shared.float()).to(bf)
    routed = torch.zeros(num_tokens, H, device="cuda")
    for e in ids.unique().tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        gu = x[tok].float() @ w13_ref[e].float().T
        g, u = gu[:, :INTER], gu[:, INTER:]
        h = (torch.nn.functional.silu(g) * u).to(bf).float()
        y = (h @ w2_ref[e].float().T).to(bf).float()
        routed.index_add_(0, tok.long(), tw[tok, slot, None] * y)
    ref = (routed.to(bf).float() + shared.float()).to(bf)
    for other in (trt, ref):
        match = (out == other).float().mean().item()
        assert match >= 0.97, match
        assert _rel(out, other) < 1e-3, _rel(out, other)


def _shared_nvfp4(gen):
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )

    out = {}
    for name, (rows, cols) in (("gu", (2 * 1024, H)), ("down", (H, 1024))):
        w = torch.randint(
            0, 256, (rows, cols // 2), dtype=torch.uint8, device="cuda", generator=gen
        )
        exp = torch.randint(4, 10, (rows, cols // 16), device="cuda", generator=gen)
        s = (exp << 3).to(torch.uint8).view(torch.float8_e4m3fn)
        gs = torch.tensor([1.5e-3], device="cuda")
        flat = swizzle_blockscale(s).contiguous().view(torch.uint8).view(-1)
        out[name] = (w, s, gs, flat)
    return out


def _ll_weights(weights, flint_weights, sh):
    return moe.LLMoEWeights(
        router=weights["router"],
        experts=flint_weights,
        shared_gu=sh["gu"][0].view(torch.int32),
        shared_gu_scale=sh["gu"][3],
        shared_gu_gs=sh["gu"][2],
        shared_down=sh["down"][0].view(torch.int32),
        shared_down_scale=sh["down"][3],
        shared_down_gs=sh["down"][2],
    )


@pytest.mark.parametrize("num_tokens", [1, 2])
@torch.inference_mode()
def test_ll_moe_block(weights, flint_weights, num_tokens):
    """The fused decode MoE block equals Flint's fused_add_rms_norm followed by
    the shared expert and the regular Flint decode MoE path."""
    from vllm.models.flint.ops import norm

    gen = torch.Generator(device="cuda").manual_seed(5)
    bf = torch.bfloat16
    sh = _shared_nvfp4(gen)
    lw = _ll_weights(weights, flint_weights, sh)
    ln_w = (torch.rand(H, device="cuda", generator=gen) + 0.5).to(bf)
    attn = (torch.randn(num_tokens, H, device="cuda", generator=gen) * 0.5).to(bf)
    res = (torch.randn(num_tokens, H, device="cuda", generator=gen) * 2).to(bf)
    counter = moe.make_route_counter("cuda")
    out, res_out = moe.flint_moe_ll(attn, res, ln_w, 1e-6, lw, TOPK, counter)
    x, r = norm.fused_add_rms_norm(attn.clone(), res.clone(), ln_w, 1e-6)
    torch.testing.assert_close(res_out, r, rtol=0, atol=0)
    gu = (x.float() @ dequantize_nvfp4(*sh["gu"][:3]).T).to(bf).float()
    h = (torch.nn.functional.silu(gu[:, :1024]) * gu[:, 1024:]).to(bf)
    shared = (h.float() @ dequantize_nvfp4(*sh["down"][:3]).T).to(bf)
    ref = moe.flint_moe(
        x,
        weights["router"],
        flint_weights,
        TOPK,
        moe.make_route_counter("cuda"),
        shared,
    )
    match = (out == ref).float().mean().item()
    assert match >= 0.98, match
    assert _rel(out, ref) < 1e-3


@pytest.mark.parametrize("preset", [-(2**31), -128, 2**31 - 128])
@torch.inference_mode()
def test_router_counters_wrap(weights, flint_weights, preset):
    """The never-reset router counters elect the token's last program
    correctly across signed 32-bit wraparound (route() and the fused block)."""
    x, shared = _inputs(2, seed=3)
    counter = moe.make_route_counter("cuda")
    ref_ids, ref_tw = moe.route(x, weights["router"], TOPK, counter)
    counter.fill_(preset)
    for _ in range(3):
        ids, tw = moe.route(x, weights["router"], TOPK, counter)
        torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
        torch.testing.assert_close(tw, ref_tw, rtol=0, atol=0)

    gen = torch.Generator(device="cuda").manual_seed(5)
    lw = _ll_weights(weights, flint_weights, _shared_nvfp4(gen))
    ln_w = torch.ones(H, device="cuda", dtype=torch.bfloat16)
    counter = moe.make_route_counter("cuda")
    ref, _ = moe.flint_moe_ll(x, shared, ln_w, 1e-6, lw, TOPK, counter)
    counter.fill_(preset)
    for _ in range(3):
        out, _ = moe.flint_moe_ll(x, shared, ln_w, 1e-6, lw, TOPK, counter)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
