# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_TP_FP16_PARTIAL_SUMS: numerics of FP16 tensor-parallel partial sums.

A row-parallel layer at TP=2 computes two partial sums a and b and all-reduces
them. With BF16 partials each rank rounds its half to BF16 before the sum, a
rounding one GPU does not have; FP16 partials (same bytes) keep the result at
the single-GPU error level.
"""

import pytest
import torch

from vllm.model_executor.layers.linear import _fp16_partial_linear

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="out_dtype matmul needs CUDA"
)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("m,k,n", [(1, 512, 256), (37, 1024, 384)])
def test_fp16_partial_linear_matches_reference(bias, m, k, n):
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda").bfloat16()
    w = torch.randn(n, k, device="cuda").bfloat16()
    b = torch.randn(n, device="cuda").bfloat16() if bias else None
    out = _fp16_partial_linear(x.view(1, m, k), w, b, 1.0)
    assert out.dtype == torch.float16 and out.shape == (1, m, n)
    ref = x.double() @ w.double().T
    if b is not None:
        ref = ref + b.double()
    # one FP16 rounding of an FP32-accumulated result
    torch.testing.assert_close(out.view(m, n).double(), ref, rtol=2e-3, atol=2e-3)


def test_fp16_partials_keep_single_gpu_error():
    """Emulate TP=2 on one GPU: the output error against the exact sum, with
    FP16 partials, stays at the single-GPU (one BF16 rounding) level, while
    BF16 partials raise it."""
    torch.manual_seed(0)
    m, k, n = 256, 2048, 1024
    x = torch.randn(m, k, device="cuda").bfloat16()
    w = (torch.randn(n, k, device="cuda") / k**0.5).bfloat16()
    h = k // 2
    a = x[:, :h].float() @ w[:, :h].float().T
    b = x[:, h:].float() @ w[:, h:].float().T
    exact = a.double() + b.double()
    rms = exact.pow(2).mean(-1, keepdim=True).sqrt()

    def err(out: torch.Tensor) -> float:
        return float(((out.double() - exact).abs() / rms).mean())

    tp1 = err((a + b).bfloat16())
    bf16_partials = err((a.bfloat16().float() + b.bfloat16().float()).bfloat16())
    fp16_partials = err((a.half().float() + b.half().float()).bfloat16())
    assert fp16_partials < 1.05 * tp1
    assert bf16_partials > 1.3 * tp1


def test_power_of_two_scale_is_exact_and_avoids_overflow():
    torch.manual_seed(0)
    x = torch.randn(64, 512, device="cuda").bfloat16()
    w = torch.randn(256, 512, device="cuda").bfloat16()
    plain = _fp16_partial_linear(x, w, None, 1.0)
    scaled = _fp16_partial_linear(x, w, None, 0.25)
    # a power-of-two scale only moves the exponent: exact while the scaled value
    # stays a normal FP16 number; below that (|v| < 2^-14 * 4) FP16 subnormals
    # bound the absolute error by 2^-24 * 4
    tiny = 2.0**-14 * 4
    normal = plain.float().abs() >= tiny
    torch.testing.assert_close(
        (scaled.float() * 4)[normal], plain.float()[normal], rtol=0, atol=0
    )
    assert float((scaled.float() * 4 - plain.float()).abs().max()) <= 2.0**-24 * 4
    # partial sums 1.5x beyond the FP16 range: saturate at scale 1 (no inf),
    # represented at scale 2^-2
    fp16_max = torch.finfo(torch.float16).max
    c = (1.5 * fp16_max / float((x.double() @ w.double().T).abs().max())) ** 0.5
    xb, wb = (x.float() * c).bfloat16(), (w.float() * c).bfloat16()
    ref = xb.double() @ wb.double().T
    assert ref.abs().max() > fp16_max
    big = _fp16_partial_linear(xb, wb, None, 1.0)
    assert torch.isfinite(big).all() and big.abs().max() == fp16_max
    big_scaled = _fp16_partial_linear(xb, wb, None, 0.25).double() * 4
    torch.testing.assert_close(
        big_scaled, ref, rtol=2e-3, atol=1e-3 * float(ref.abs().max())
    )
