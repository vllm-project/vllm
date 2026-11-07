# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint's dense-linear and elementwise ops against high-precision references:
the small-M NVFP4 / BF16 GEMVs (``nvfp4_gemv``, incl. the fused SiLU-and-mul
epilogue), the large-M dequantize + cuBLAS path and global-scale check
(``linear``), and the Triton norms and SiLU-and-mul (``norm``)."""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)
from vllm.models.flint.gguf_nvfp4 import dequantize_nvfp4
from vllm.models.flint.ops import nvfp4_gemv
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="SM100 only",
)


def _cases(table):
    """(N, K, M) for each tuned shape: M within the first token tile, and a
    partial and a full largest tile."""
    return [
        (n, k, m)
        for (n, k), cfgs in table.items()
        for m in sorted({1, 5, 8 * max(cfgs) - 3, 8 * max(cfgs)})
    ]


def _nvfp4(n: int, k: int, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randint(
        0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    # Every finite non-negative E4M3 code, subnormals and zero included.
    s = torch.randint(
        0, 127, (n, k // 16), dtype=torch.uint8, device="cuda", generator=gen
    ).view(torch.float8_e4m3fn)
    gs = torch.tensor(1.7e-3, device="cuda")
    return w, s, gs


def _reference(x, w, s, gs):
    return x.double() @ dequantize_nvfp4(w, s, gs).double().t()


def _silu_and_mul_ref(gate_up):
    g, u = gate_up.float().chunk(2, dim=-1)
    return (g / (1 + torch.exp(-g)) * u).bfloat16()


def _check(out, ref):
    # FP32 accumulation over K <= 6144 terms vs. a float64 reference: far
    # below one BF16 ulp, so the only visible error is the output rounding.
    err = (out.double() - ref).abs()
    tol = 2.0**-8 * ref.abs() + 1e-3 * ref.abs().max()
    assert (err <= tol).all(), f"max err {err.max().item()}"


@pytest.mark.parametrize("n,k,m", _cases(nvfp4_gemv._TUNED))
def test_gemv_matches_dequantized_reference(n, k, m):
    w, s, gs = _nvfp4(n, k)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    out = nvfp4_gemv.nvfp4_gemv(x, w, swizzle_blockscale(s), gs)
    _check(out, _reference(x, w, s, gs))


def test_gemv_dequantizes_every_code_exactly():
    """A one-hot activation reads back single dequantized weights bit-exactly:
    all 16 E2M1 codes times all finite E4M3 scales, incl. subnormals."""
    n, k = 2048, 4096
    w = torch.arange(256, dtype=torch.uint8, device="cuda").repeat(n, k // 512)
    w = w[torch.randperm(n, device="cuda")]
    s = (torch.arange(n * k // 16, device="cuda") % 127).to(torch.uint8)
    s = s.reshape(n, k // 16).view(torch.float8_e4m3fn)
    gs = torch.tensor(1.7e-3, device="cuda")
    m = 16
    x = torch.zeros(m, k, dtype=torch.bfloat16, device="cuda")
    x[torch.arange(m), torch.randint(0, k, (m,))] = 1.0
    out = nvfp4_gemv.nvfp4_gemv(x, w, swizzle_blockscale(s), gs)
    ref = (x.float() @ dequantize_nvfp4(w, s, gs).t()).bfloat16()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_gemv_strided_input():
    n, k = 2048, 4096
    w, s, gs = _nvfp4(n, k)
    base = torch.randn(4, k + 256, dtype=torch.bfloat16, device="cuda")
    x = base[:, 128 : 128 + k]
    out = nvfp4_gemv.nvfp4_gemv(x, w, swizzle_blockscale(s), gs)
    _check(out, _reference(x, w, s, gs))


@pytest.mark.parametrize("n,k,m", _cases(nvfp4_gemv._TUNED_SILU_MUL))
def test_gemv_silu_mul_matches_unfused(n, k, m):
    """The fused gate_up + SiLU-and-mul epilogue equals SiLU-and-mul applied to
    the BF16-rounded GEMM output."""
    w, s, _ = _nvfp4(n, k)
    gs = torch.tensor(1e-5, device="cuda")  # gate/up values of O(1-10)
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", generator=gen)
    ref = _silu_and_mul_ref(_reference(x, w, s, gs).bfloat16())
    out = nvfp4_gemv.nvfp4_gemv(x, w, swizzle_blockscale(s), gs, silu_mul=True)
    # FP32 vs float64 sums can round a gate/up value one BF16 ulp apart, which
    # SiLU amplifies for negative gates (~3% at -8).
    err = (out.float() - ref.float()).abs()
    close = err <= 2**-7 * ref.float().abs() + 1e-6
    assert close.float().mean() > 0.999
    assert (err <= 0.1 * ref.float().abs() + 1e-5).all()


def test_merged_global_scale_mismatch_raises():
    from vllm.models.flint.ops.linear import check_merged_global_scales

    p = "model.layers.0.self_attn."
    ok = [(f"{p}{n}_proj.weight_scale_2", torch.tensor(0.5)) for n in "qkv"]
    assert len(list(check_merged_global_scales(ok))) == 3
    bad = ok[:2] + [(p + "v_proj.weight_scale_2", torch.tensor(0.25))]
    with pytest.raises(ValueError):
        list(check_merged_global_scales(bad))


@pytest.mark.parametrize("n,k,m", _cases(nvfp4_gemv._TUNED_BF16))
def test_bf16_gemv_matches_reference(n, k, m):
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    out = nvfp4_gemv.bf16_gemv(x, w)
    _check(out, x.double() @ w.double().t())


def test_bf16_gemv_silu_mul():
    n, k = 2048, 2048
    gen = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=gen) * 0.05
    x = torch.randn(4, k, dtype=torch.bfloat16, device="cuda", generator=gen)
    ref = _silu_and_mul_ref((x.double() @ w.double().t()).bfloat16())
    out = nvfp4_gemv.bf16_gemv(x, w, silu_mul=True)
    err = (out.float() - ref.float()).abs()
    assert (err <= 2**-7 * ref.float().abs() + 1e-6).float().mean() > 0.999


def test_large_m_dequant_gemm_matches_reference():
    from types import SimpleNamespace

    from vllm.models.flint.ops.linear import _dequant_gemm

    n, k = 2048, 4096
    w, s, gs = _nvfp4(n, k)
    layer = SimpleNamespace(
        weight=w,
        flint_weight_scale=swizzle_blockscale(s),
        flint_global_scale=float(gs),
    )
    x = torch.randn(256, k, dtype=torch.bfloat16, device="cuda")
    _check(_dequant_gemm(layer, x), _reference(x, w, s, gs))


def _rms_norm_ref(s, w, eps):
    rs = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    return (s * rs * w.float()).bfloat16()


@pytest.mark.parametrize("m", [1, 7, 8192])
@pytest.mark.parametrize("has_residual", [False, True])
@torch.inference_mode()
def test_rms_norm_matches_reference(m, has_residual):
    """Flint's (fused residual-add) RMSNorm: fp32 math on the unrounded
    x + residual, one bf16 rounding of each output. rsqrt.approx and the
    reduction order allow rare 1-ulp differences from the fp32 reference."""
    from vllm.models.flint.ops import norm

    gen = torch.Generator(device="cuda").manual_seed(m)
    h = 2048
    x = (torch.randn(m, h, device="cuda", generator=gen) * 2).bfloat16()
    w = torch.empty(h, device="cuda").normal_(1, 0.2, generator=gen).bfloat16()
    if has_residual:
        r = (torch.randn(m, h, device="cuda", generator=gen) * 4).bfloat16()
        s = x.float() + r.float()
        out, res_out = norm.fused_add_rms_norm(x.clone(), r.clone(), w, 1e-6)
        torch.testing.assert_close(res_out, s.bfloat16(), rtol=0, atol=0)
    else:
        out = norm.rms_norm(x, w, 1e-6)
        s = x.float()
    ref = _rms_norm_ref(s, w, 1e-6)
    assert (out == ref).float().mean() > 0.999
    torch.testing.assert_close(out, ref, rtol=2**-7, atol=0)


@pytest.mark.parametrize("m", [1, 7, 8192])
@torch.inference_mode()
def test_silu_and_mul_matches_reference(m):
    """bf16(silu(g) * u) in fp32 with one rounding."""
    from vllm.models.flint.ops import norm

    gen = torch.Generator(device="cuda").manual_seed(m)
    x = (torch.randn(m, 2 * 6144, device="cuda", generator=gen) * 3).bfloat16()
    out = norm.silu_and_mul(x)
    ref = _silu_and_mul_ref(x)
    assert (out == ref).float().mean() > 0.999
    torch.testing.assert_close(out, ref, rtol=2**-7, atol=1e-6)


@pytest.mark.parametrize("d", [3072, 1536, 768])
@torch.inference_mode()
def test_silu_and_mul_tp_sharded_widths(d):
    """TP shards the dense 6144 intermediate to widths that are not
    1024-divisible (TP4: 1536) or not powers of two (TP8: 768)."""
    from vllm.models.flint.ops import norm

    gen = torch.Generator(device="cuda").manual_seed(d)
    x = (torch.randn(5, 2 * d, device="cuda", generator=gen) * 3).bfloat16()
    out = norm.silu_and_mul(x)
    ref = _silu_and_mul_ref(x)
    assert (out == ref).float().mean() > 0.999
    torch.testing.assert_close(out, ref, rtol=2**-7, atol=1e-6)
