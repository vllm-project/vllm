# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for cuteDSL low-latency GEMM with fp32 weights."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.fixture(autouse=True, scope="module")
def _require_sm90_and_cutedsl():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Requires SM90+ (Hopper/Blackwell)")
    from vllm.model_executor.kernels.linear.cute_dsl.ll_fp32w import (
        is_available,
    )
    if not is_available():
        pytest.skip("cuteDSL (CUTLASS Python) not installed")


def _ref(a, b):
    return torch.mm(a.float(), b.float().T)


def _assert_close(out, ref, *, min_cos_sim=0.99, context=""):
    assert out.device.type == "cuda", f"{context}: not on CUDA"
    assert torch.isfinite(out).all(), f"{context}: NaN/Inf"
    cos = F.cosine_similarity(
        out.reshape(-1).float(), ref.reshape(-1).float(), dim=0
    ).item()
    assert cos > min_cos_sim, (
        f"{context}: cos_sim {cos:.6f} < {min_cos_sim} "
        f"(abs_err={(out.float() - ref.float()).abs().max().item():.2e})"
    )


def _gemm(a, b):
    from vllm.model_executor.kernels.linear.cute_dsl.ll_fp32w import (
        ll_fp32w_gemm,
    )
    return ll_fp32w_gemm(a, b)


SHAPES = [
    (256, 3072, "MiniMax-M2"),
    (256, 7168, "DSV3-sized"),
    (128, 5120, "DeepSeek-V2"),
    (64, 2880, "non-aligned-K"),
    (8, 4096, "Mixtral"),
]

A_DTYPES = [
    pytest.param(torch.bfloat16, id="bf16"),
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.float32, id="fp32"),
]


# =================================================================
# Correctness: all dtype combos x shapes x M values
# =================================================================


@pytest.mark.parametrize("a_dtype", A_DTYPES)
@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("N,K,desc", SHAPES, ids=[s[2] for s in SHAPES])
def test_correctness(M, N, K, desc, a_dtype):
    torch.manual_seed(42)
    a = torch.randn(M, K, device="cuda").to(a_dtype)
    b = torch.randn(N, K, dtype=torch.float32, device="cuda")
    out = _gemm(a, b)
    assert out.dtype == torch.float32
    assert out.shape == (M, N)
    _assert_close(out, _ref(a, b), context=f"{a_dtype} M={M} {desc}")


# =================================================================
# MiniMax-M2 specific shapes
# =================================================================


@pytest.mark.parametrize("M", [1, 4, 8, 16, 32])
def test_minimax_m2(M):
    torch.manual_seed(42)
    a = torch.randn(M, 3072, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(256, 3072, dtype=torch.float32, device="cuda")
    out = _gemm(a, b)
    _assert_close(out, _ref(a, b), context=f"MiniMax-M2 M={M}")


# =================================================================
# Arbitrary N
# =================================================================


@pytest.mark.parametrize("N", [1, 3, 16, 64, 256])
def test_arbitrary_N(N):
    torch.manual_seed(42)
    a = torch.randn(4, 3072, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, 3072, dtype=torch.float32, device="cuda")
    out = _gemm(a, b)
    assert out.shape == (4, N)
    _assert_close(out, _ref(a, b), context=f"N={N}")


# =================================================================
# Numerical robustness
# =================================================================


def test_large_values():
    torch.manual_seed(42)
    a = torch.randn(4, 3072, dtype=torch.bfloat16, device="cuda") * 100
    b = torch.randn(256, 3072, dtype=torch.float32, device="cuda") * 100
    out = _gemm(a, b)
    assert torch.isfinite(out).all()
    _assert_close(out, _ref(a, b), context="large_values")


def test_zeros():
    a = torch.zeros(4, 3072, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(256, 3072, dtype=torch.float32, device="cuda")
    assert (_gemm(a, b) == 0).all()


# =================================================================
# Output dtype
# =================================================================


def test_output_fp32():
    torch.manual_seed(42)
    a = torch.randn(4, 3072, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(256, 3072, dtype=torch.float32, device="cuda")
    assert _gemm(a, b).dtype == torch.float32


# =================================================================
# Determinism
# =================================================================


@pytest.mark.parametrize("M", [1, 4, 16])
def test_deterministic(M):
    torch.manual_seed(42)
    a = torch.randn(M, 3072, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(256, 3072, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(_gemm(a, b), _gemm(a, b), atol=0, rtol=0)


# =================================================================
# CUDA graph
# =================================================================


@pytest.mark.parametrize("M", [4, 16], ids=["M4", "M16"])
def test_cudagraph(M):
    torch.manual_seed(42)
    a = torch.randn(M, 3072, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(256, 3072, dtype=torch.float32, device="cuda")
    _gemm(a, b)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = _gemm(a, b)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    _assert_close(out, _ref(a, b), context=f"cudagraph M={M}")


# =================================================================
# Negative tests
# =================================================================


def test_invalid_weight_dtype():
    """Weights must be fp32."""
    a = torch.randn(4, 3072, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(256, 3072, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        _gemm(a, b)


def test_invalid_device_cpu():
    a = torch.randn(4, 3072, dtype=torch.bfloat16)
    b = torch.randn(256, 3072, dtype=torch.float32)
    with pytest.raises(ValueError):
        _gemm(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
