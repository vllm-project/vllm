# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU stage parity of the FFN launch (a TP=1 build) against vLLM's and
AITER's native ops, with MXFP4 weights from vLLM's real post-load path."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx950  # noqa: E402

if not on_gfx950():
    pytest.skip("gfx950 only", allow_module_level=True)

from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import (  # noqa: E402
    gemma_fused_add_rmsnorm,
)
from vllm.models.minimax_m3.amd.specialized_decode.mono import ffn  # noqa: E402
from vllm.models.minimax_m3.amd.specialized_decode.mono.runner import (  # noqa: E402
    FfnRunner,
)

E = ffn.EXPERTS + 1
H, INTER = ffn.HIDDEN, ffn.INTER


def _metrics(got, want):
    got, want = got.float().flatten(), want.float().flatten()
    cos = torch.nn.functional.cosine_similarity(got, want, dim=0).item()
    rel = ((got - want).norm() / want.norm()).item()
    return cos, rel


@pytest.fixture(scope="module")
def weights():
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    )

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)

    def codes(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device=dev, generator=g)

    def scales(*shape):
        # e8m0 codes around 2^-7: gate/up pre-activations of a few units
        return torch.randint(
            119, 123, shape, dtype=torch.uint8, device=dev, generator=g
        )

    def param(t):
        return torch.nn.Parameter(t, requires_grad=False)

    w13, w2, s13, s2, _, _ = convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
        mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
        layer=torch.nn.Module(),
        w13_weight=param(codes(E, 2 * INTER, H // 2)),
        w2_weight=param(codes(E, H, INTER // 2)),
        w13_weight_scale=param(scales(E, 2 * INTER, H // 32)),
        w2_weight_scale=param(scales(E, H, INTER // 32)),
    )
    return SimpleNamespace(
        w13=w13,
        w2=w2,
        s13=s13,
        s2=s2,
        post_norm=torch.randn(H, device=dev, generator=g).mul(0.1).bfloat16(),
        router=torch.randn(ffn.EXPERTS, H, device=dev, generator=g).mul(0.02),
        router_bias=torch.randn(ffn.EXPERTS, device=dev, generator=g).mul(0.01),
    )


def _native_experts(x, w, ids, wts):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    return fused_moe(
        x,
        w.w13,
        w.w2,
        wts,
        ids,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        w1_scale=w.s13,
        w2_scale=w.s2,
        gate_mode="separated",
        swiglu_limit=ffn.LIMIT,
    )


def _native_route(logits, bias):
    from aiter import biased_grouped_topk

    s = logits.shape[0]
    ids = torch.empty(s, ffn.TOPK, dtype=torch.int32, device=logits.device)
    wts = torch.empty(s, ffn.TOPK, dtype=torch.float32, device=logits.device)
    biased_grouped_topk(logits, bias, wts, ids, 1, 1, True, ffn.ROUTED_SCALE)
    return ids, wts


def _clear_choices(logits, bias, margin=1e-4):
    """Rows whose 4th and 5th biased scores differ by more than ``margin``."""
    top = (logits.sigmoid() + bias).topk(ffn.TOPK + 1, dim=-1).values
    return (top[:, ffn.TOPK - 1] - top[:, ffn.TOPK]) > margin


def _pairs(region, n):
    return region[: n * 8].view(torch.int32).view(n, 2)[:, 0].contiguous()


def _run(weights, s, experts=ffn.Experts.ALL, seed=1):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    part = torch.randn(s, H, device=dev, generator=g).bfloat16()
    residual = torch.randn(s, H, device=dev, generator=g).mul(4).bfloat16()
    peer = torch.zeros(ffn.peer_bytes(1), dtype=torch.uint8, device=dev)
    peers = torch.tensor([peer.data_ptr()], dtype=torch.int64, device=dev)
    runner = FfnRunner(1, 0, peer.data_ptr(), peers, dev, (s,), experts)
    runner.warm(weights)
    out, res_out = runner(part, residual, weights)
    torch.accelerator.synchronize()
    assert peer[: 4 * ffn.ERROR_WORDS].view(torch.int32)[0].item() == 0
    x = runner.region(s, "xn")[: s * H * 2].view(torch.bfloat16).view(s, H)
    parts = _pairs(runner.region(s, "logit"), s * ffn.EXPERTS * ffn.ROUTER_PARTS)
    parts = parts.view(torch.float32).view(s * ffn.EXPERTS, ffn.ROUTER_PARTS)
    logits = parts[:, 0].clone()
    for p in range(1, ffn.ROUTER_PARTS):
        logits += parts[:, p]
    route = _pairs(runner.region(s, "route"), s * 2 * ffn.TOPK).view(s, 2 * ffn.TOPK)
    return SimpleNamespace(
        part=part,
        residual=residual,
        out=out,
        res_out=res_out,
        x=x,
        logits=logits.view(s, ffn.EXPERTS),
        ids=route[:, : ffn.TOPK].contiguous(),
        wts=route[:, ffn.TOPK :].contiguous().view(torch.float32),
    )


@pytest.mark.parametrize("s", [1, 3, 8, 16])
def test_ffn_stages_match_native(weights, s):
    r = _run(weights, s)
    want_x, want_res = gemma_fused_add_rmsnorm(
        r.part, r.residual, weights.post_norm, ffn.EPS
    )
    torch.testing.assert_close(r.res_out, want_res, rtol=0, atol=0)
    cos, rel = _metrics(r.x, want_x)
    assert cos > 0.99999 and rel < 4e-3

    want_logits = r.x.float() @ weights.router.T
    torch.testing.assert_close(r.logits, want_logits, rtol=1e-4, atol=1e-4)

    ids, wts = _native_route(r.logits, weights.router_bias)
    clear = _clear_choices(r.logits, weights.router_bias)
    assert clear.any()
    got_ids, order = r.ids.sort(-1)
    want_ids, want_order = ids.sort(-1)
    assert torch.equal(got_ids[clear], want_ids[clear])
    torch.testing.assert_close(
        r.wts.gather(1, order)[clear],
        wts.gather(1, want_order)[clear],
        rtol=1e-5,
        atol=1e-6,
    )

    shared = torch.full((s, 1), ffn.SHARED, dtype=torch.int32, device=r.x.device)
    all_ids = torch.cat([r.ids, shared], 1)
    all_wts = torch.cat([r.wts, torch.ones_like(r.wts[:, :1])], 1)
    want = _native_experts(r.x, weights, all_ids, all_wts)
    cos, rel = _metrics(r.out, want)
    assert cos > 0.9999 and rel < 1e-2


@pytest.mark.parametrize("experts", [ffn.Experts.SHARED, ffn.Experts.ROUTED])
@pytest.mark.parametrize("s", [1, 16])
def test_ffn_expert_sides_match_native(weights, s, experts):
    r = _run(weights, s, experts)
    if experts is ffn.Experts.SHARED:
        ids = torch.full((s, 1), ffn.SHARED, dtype=torch.int32, device=r.x.device)
        wts = torch.ones(s, 1, device=r.x.device)
    else:
        ids, wts = r.ids, r.wts
    want = _native_experts(r.x, weights, ids, wts)
    cos, rel = _metrics(r.out, want)
    assert cos > 0.9999 and rel < 1e-2
