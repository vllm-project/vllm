# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CuTe DSL scaled GEMM kernels (FP8 and INT8) on SM90.

Run `pytest tests/kernels/quantization/test_cutedsl_scaled_mm.py`.
"""

import pytest
import torch

from tests.kernels.utils import baseline_scaled_mm, to_fp8, to_int8
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CuTe DSL kernels require CUDA", allow_module_level=True)

MNK_FACTORS = [
    (1, 256, 128),
    (1, 16384, 1024),
    (1, 24576, 4096),
    (16, 256, 4096),
    (16, 16384, 128),
    (16, 24576, 4096),
    (32, 8192, 4096),
    (32, 16384, 4096),
    (33, 1024, 1024),
    (33, 8192, 128),
    (64, 2048, 4096),
    (64, 16384, 1024),
    (100, 8192, 496),
    (128, 32768, 4096),
    (256, 4096, 4096),
    (512, 256, 1024),
    (512, 8192, 4096),
    (512, 16384, 128),
    (512, 24576, 128),
]

CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]

SCALE_COMBOS = [
    ("tensor_tensor", lambda m, n: (1, 1), lambda m, n: (1, 1)),
    ("token_tensor", lambda m, n: (m, 1), lambda m, n: (1, 1)),
    ("tensor_channel", lambda m, n: (1, 1), lambda m, n: (1, n)),
    ("token_channel", lambda m, n: (m, 1), lambda m, n: (1, n)),
]


def _cutedsl_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None):
    """Call CuTe DSL scaled MM dispatch, skipping unsupported configs."""
    from vllm.kernels.quantization.cutedsl.scaled_mm_dispatch import (
        cutedsl_scaled_mm,
    )

    result = cutedsl_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=bias)
    if result is None:
        pytest.skip("CuTe DSL does not support this configuration")
    return result

# FP8 helpers
def cutedsl_fp8_gemm_helper(
    m: int,
    n: int,
    k: int,
    out_dtype: type[torch.dtype] = torch.bfloat16,
    device: str = "cuda",
    scale_a_shape=(1, 1),
    scale_b_shape=(1, 1),
    use_bias: bool = False,
):
    a = to_fp8(torch.randn((m, k), device=device))
    b = to_fp8(torch.randn((n, k), device=device).t())

    scale_a = torch.randn(scale_a_shape, device=device, dtype=torch.float32)
    scale_b = torch.randn(scale_b_shape, device=device, dtype=torch.float32)
    bias = torch.randn(n, device=device, dtype=out_dtype) if use_bias else None

    out = _cutedsl_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=bias)
    baseline = baseline_scaled_mm(a, b, scale_a, scale_b, out_dtype)
    if bias is not None:
        baseline = baseline + bias

    torch.testing.assert_close(out, baseline, rtol=5e-1, atol=1.5e-1)

# INT8 helpers
def cutedsl_int8_gemm_helper(
    m: int,
    n: int,
    k: int,
    out_dtype: type[torch.dtype] = torch.bfloat16,
    device: str = "cuda",
    scale_a_shape=(1, 1),
    scale_b_shape=(1, 1),
    use_bias: bool = False,
):
    a = to_int8(torch.randn((m, k), device=device) * 5)
    b = to_int8(torch.randn((n, k), device=device).t() * 5)

    scale_a = torch.randn(scale_a_shape, device=device, dtype=torch.float32)
    scale_b = torch.randn(scale_b_shape, device=device, dtype=torch.float32)
    bias = torch.randn(n, device=device, dtype=out_dtype) if use_bias else None

    out = _cutedsl_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=bias)
    baseline = baseline_scaled_mm(a, b, scale_a, scale_b, out_dtype)
    if bias is not None:
        baseline = baseline + bias

    torch.testing.assert_close(out, baseline, rtol=1e-1, atol=1e0)

# Per-tensor scale tests (regression)
@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="CuTe DSL SM90 kernels require Hopper or later.",
)
def test_cutedsl_fp8_gemm(m: int, n: int, k: int):
    cutedsl_fp8_gemm_helper(m, n, k)

@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="CuTe DSL SM90 kernels require Hopper or later.",
)
def test_cutedsl_int8_gemm(m: int, n: int, k: int):
    cutedsl_int8_gemm_helper(m, n, k)

# Scale combination tests
@pytest.mark.parametrize(
    "desc,sa_fn,sb_fn",
    [(c[0], c[1], c[2]) for c in SCALE_COMBOS],
    ids=[c[0] for c in SCALE_COMBOS],
)
@pytest.mark.parametrize("m,n,k", MNK_FACTORS[:6])
@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="CuTe DSL SM90 kernels require Hopper or later.",
)
def test_cutedsl_fp8_gemm_scales(m, n, k, desc, sa_fn, sb_fn):
    cutedsl_fp8_gemm_helper(
        m, n, k, scale_a_shape=sa_fn(m, n), scale_b_shape=sb_fn(m, n))

@pytest.mark.parametrize(
    "desc,sa_fn,sb_fn",
    [(c[0], c[1], c[2]) for c in SCALE_COMBOS],
    ids=[c[0] for c in SCALE_COMBOS],
)
@pytest.mark.parametrize("m,n,k", MNK_FACTORS[:6])
@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="CuTe DSL SM90 kernels require Hopper or later.",
)
def test_cutedsl_int8_gemm_scales(m, n, k, desc, sa_fn, sb_fn):
    cutedsl_int8_gemm_helper(
        m, n, k, scale_a_shape=sa_fn(m, n), scale_b_shape=sb_fn(m, n))

# Bias tests
@pytest.mark.parametrize("m,n,k", [(64, 256, 128), (128, 512, 256),
                                     (256, 4096, 4096)])
@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="CuTe DSL SM90 kernels require Hopper or later.",
)
def test_cutedsl_fp8_gemm_bias(m, n, k):
    cutedsl_fp8_gemm_helper(m, n, k, use_bias=True)

@pytest.mark.parametrize("m,n,k", [(64, 256, 128), (128, 512, 256),
                                     (256, 4096, 4096)])
@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="CuTe DSL SM90 kernels require Hopper or later.",
)
def test_cutedsl_int8_gemm_bias(m, n, k):
    cutedsl_int8_gemm_helper(m, n, k, use_bias=True)

# Output dtype tests
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="CuTe DSL SM90 kernels require Hopper or later.",
)
def test_cutedsl_fp8_gemm_output_dtype(out_dtype: type[torch.dtype]):
    cutedsl_fp8_gemm_helper(512, 512, 512, out_dtype=out_dtype)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="CuTe DSL SM90 kernels require Hopper or later.",
)
def test_cutedsl_int8_gemm_output_dtype(out_dtype: type[torch.dtype]):
    cutedsl_int8_gemm_helper(512, 512, 512, out_dtype=out_dtype)

# Multi-device tests
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="CuTe DSL SM90 kernels require Hopper or later.",
)
def test_cutedsl_fp8_gemm_devices(device: str):
    cutedsl_fp8_gemm_helper(512, 512, 512, device=device)

@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="CuTe DSL SM90 kernels require Hopper or later.",
)
def test_cutedsl_int8_gemm_devices(device: str):
    cutedsl_int8_gemm_helper(512, 512, 512, device=device)

# M-sweep tests
@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="CuTe DSL SM90 kernels require Hopper or later.",
)
def test_cutedsl_fp8_gemm_m_sweep():
    for nk in range(32, 128, 32):
        for m in range(1, 128):
            cutedsl_fp8_gemm_helper(m, nk, nk)

@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="CuTe DSL SM90 kernels require Hopper or later.",
)
def test_cutedsl_int8_gemm_m_sweep():
    for nk in range(32, 128, 32):
        for m in range(1, 128):
            cutedsl_int8_gemm_helper(m, nk, nk)
