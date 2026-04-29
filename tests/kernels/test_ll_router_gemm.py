# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.ll_router_gemm import (
    is_available,
    ll_router_gemm,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def _check_cutedsl():
    if not is_available():
        pytest.skip("cuteDSL (CUTLASS Python) not installed")


# --- Test shapes ---

# (N, K, description)
SHAPES = [
    (256, 7168, "DSV3 router"),
    (256, 2048, "small K"),
    (128, 5120, "DeepSeek V2"),
    (8, 4096, "Mixtral-8x7B"),
    (64, 2880, "non-aligned K"),
]

M_VALUES = [1, 2, 4, 8, 16]

DTYPES_BF16 = [(torch.bfloat16, "bf16")]
DTYPES_FP8 = [(torch.float8_e4m3fn, "fp8")]
DTYPES_ALL = DTYPES_BF16 + DTYPES_FP8


def _make_inputs(M, N, K, dtype):
    if dtype == torch.bfloat16:
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    else:
        a = torch.randn(M, K, device="cuda").to(dtype)
        b = torch.randn(N, K, device="cuda").to(dtype)
    return a, b


def _reference(a, b):
    """torch.mm fp32 reference."""
    return torch.mm(a.float(), b.float().T)


def _check(out, ref, atol, rtol, label=""):
    err = (out.float() - ref.float()).abs()
    max_abs = err.max().item()
    max_rel = (err / ref.float().abs().clamp(min=1e-6)).max().item()
    print(f"  {label}: max_abs={max_abs:.2e} max_rel={max_rel:.2e}")
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


# --- Correctness tests ---


@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("N,K,desc", SHAPES, ids=[s[2] for s in SHAPES])
def test_bf16_correctness(M, N, K, desc):
    a, b = _make_inputs(M, N, K, torch.bfloat16)
    out = ll_router_gemm(a, b)
    ref = _reference(a, b)
    _check(out, ref, atol=1e-3, rtol=1e-3, label=f"{M}x{N}x{K}")


@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize(
    "N,K,desc",
    [(n, k, d) for n, k, d in SHAPES if k % 2 == 0],
    ids=[s[2] for s in SHAPES if s[1] % 2 == 0],
)
def test_fp8_correctness(M, N, K, desc):
    a, b = _make_inputs(M, N, K, torch.float8_e4m3fn)
    out = ll_router_gemm(a, b)
    ref = _reference(a, b)
    _check(out, ref, atol=1e-2, rtol=1e-2, label=f"fp8 {M}x{N}x{K}")


@pytest.mark.parametrize("M", [1, 4, 16])
def test_output_dtype_float32(M):
    N, K = 64, 2048
    a, b = _make_inputs(M, N, K, torch.bfloat16)
    out = ll_router_gemm(a, b, output_dtype=torch.float32)
    assert out.dtype == torch.float32
    assert out.shape == (M, N)


@pytest.mark.parametrize("M", [1, 16])
def test_deterministic(M):
    """Same inputs should produce identical outputs."""
    N, K = 128, 4096
    a, b = _make_inputs(M, N, K, torch.bfloat16)
    out1 = ll_router_gemm(a, b)
    out2 = ll_router_gemm(a, b)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


@pytest.mark.parametrize("N", [1, 3, 7, 17, 64, 256])
def test_arbitrary_N(N):
    """N can be any positive integer."""
    M, K = 4, 2048
    a, b = _make_inputs(M, N, K, torch.bfloat16)
    out = ll_router_gemm(a, b)
    ref = _reference(a, b)
    _check(out, ref, atol=1e-3, rtol=1e-3, label=f"{M}x{N}x{K}")


@pytest.mark.parametrize("K", [128, 256, 512, 1024, 2048, 4096, 7168])
def test_arbitrary_K(K):
    """K can be any positive integer (multiple of 2 for fp8)."""
    M, N = 4, 32
    a, b = _make_inputs(M, N, K, torch.bfloat16)
    out = ll_router_gemm(a, b)
    ref = _reference(a, b)
    _check(out, ref, atol=1e-3, rtol=1e-3, label=f"{M}x{N}x{K}")


def test_small_K():
    """Very small K that fits entirely in the scalar tail."""
    M, N, K = 2, 8, 64
    a, b = _make_inputs(M, N, K, torch.bfloat16)
    out = ll_router_gemm(a, b)
    ref = _reference(a, b)
    _check(out, ref, atol=1e-3, rtol=1e-3, label=f"{M}x{N}x{K}")


def test_cudagraph_compatible():
    """Kernel must work inside a CUDA graph."""
    M, N, K = 4, 64, 2048
    a, b = _make_inputs(M, N, K, torch.bfloat16)

    # Warm up compilation
    ll_router_gemm(a, b)
    torch.accelerator.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = ll_router_gemm(a, b)
    g.replay()
    torch.accelerator.synchronize()

    ref = _reference(a, b)
    _check(out, ref, atol=1e-3, rtol=1e-3, label=f"{M}x{N}x{K}")
