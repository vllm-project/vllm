# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU INT8 scaled-mm GEMM kernels (`int8_scaled_mm_with_quant`).

Run `pytest tests/kernels/quantization/test_cpu_int8_scaled_mm.py -v`.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.cpu_test

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

if not hasattr(torch.ops._C, "int8_scaled_mm_with_quant"):
    pytest.skip("int8_scaled_mm_with_quant is not built", allow_module_level=True)


def _reference_int8_scaled_mm(
    x: torch.Tensor,
    raw_w: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    x_f32 = x.to(torch.float32)
    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-7)
    qmax = torch.tensor(127.0, dtype=torch.float32)
    a_scale = amax / qmax
    q_a = torch.round(x_f32 * (qmax / amax)).to(torch.int32)
    acc = (q_a @ raw_w.to(torch.int32).T).to(torch.float32)
    out = (acc * a_scale) * scale.unsqueeze(0)
    if bias is not None:
        out = out + bias.unsqueeze(0)
    return out.to(torch.bfloat16)


@pytest.mark.parametrize(
    ("m", "k", "n", "with_bias", "strided"),
    [
        # Small M <= 3: fallback to upstream VNNI path
        (1, 128, 256, False, False),
        (1, 4096, 4096, True, False),
        (2, 4096, 512, False, True),
        (3, 2048, 512, True, False),
        # Small-to-mid M: 3D small-grid AMX BRGEMM, M = 33..63 regime
        (4, 4096, 1024, False, False),
        (16, 4096, 1024, True, True),
        (48, 4096, 1024, True, False),
        # Ragged M covering tail <= 16 and 17 <= tail <= 31
        (65, 256, 512, True, False),
        (95, 1536, 512, False, True),
        # Short K, non-512-aligned K, and Split-K
        (128, 128, 512, False, False),
        (128, 768, 512, True, True),
        (128, 4096, 1024, True, False),
        # Large M with wide N (coarse macro-tile & ragged M tail)
        (640, 2048, 4096, False, False),
        (641, 2048, 4096, True, False),
        # Non-64-aligned K fallback to upstream path
        (16, 96, 64, True, False),
    ],
)
def test_cpu_int8_scaled_mm_with_quant_parity(m, k, n, with_bias, strided):
    """Fused INT8 GEMM dispatch must match exact quantized matmul reference."""
    torch.manual_seed(42 + m + k + n)
    if strided:
        x = torch.randn(m, k + 37, dtype=torch.bfloat16)[:, :k]
        assert x.stride(0) == k + 37 and x.stride(1) == 1
    else:
        x = torch.randn(m, k, dtype=torch.bfloat16)
    raw = torch.randint(-32, 33, (n, k), dtype=torch.int8)
    packed = torch.ops._C.convert_weight_packed(raw)
    scale = torch.rand(n, dtype=torch.float32) * 0.01 + 1e-3
    bias = torch.randn(n, dtype=torch.float32) if with_bias else None

    ref = _reference_int8_scaled_mm(x, raw, scale, bias)
    got = torch.ops._C.int8_scaled_mm_with_quant(
        x, packed, scale, bias, torch.bfloat16, True
    )
    if with_bias:
        torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)
    else:
        assert torch.equal(ref.view(torch.uint16), got.view(torch.uint16))


def test_cpu_int8_varying_m_and_concurrent_callers():
    """Capacity-reusing activation cache and worker scratch stay isolated."""
    torch.manual_seed(7)
    n, k = 1024, 2048
    raw = torch.randint(-32, 33, (n, k), dtype=torch.int8)
    packed = torch.ops._C.convert_weight_packed(raw)
    scale = torch.rand(n, dtype=torch.float32) * 0.01 + 1e-3
    bias = torch.randn(n, dtype=torch.float32)

    inputs = [
        torch.randn(m, k, dtype=torch.bfloat16) * (i + 1)
        for i, m in enumerate((1, 4, 16, 48, 256, 200, 2048, 2000, 1984, 129, 256, 193))
    ]
    refs = [_reference_int8_scaled_mm(x, raw, scale, bias) for x in inputs]

    for x, ref in zip(inputs, refs):
        got = torch.ops._C.int8_scaled_mm_with_quant(
            x, packed, scale, bias, torch.bfloat16, True
        )
        torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)

    def run_worker(idx: int) -> torch.Tensor:
        return torch.ops._C.int8_scaled_mm_with_quant(
            inputs[idx], packed, scale, bias, torch.bfloat16, True
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        outs = list(pool.map(run_worker, range(len(inputs))))
    for got, ref in zip(outs, refs):
        torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)
