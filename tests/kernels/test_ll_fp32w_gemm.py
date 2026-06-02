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
# Small K (k_main=0, only tail/scalar paths)
# =================================================================


@pytest.mark.parametrize("K", [16, 32, 64, 128, 256, 512, 1024, 1536])
def test_small_K(K):
    torch.manual_seed(42)
    a = torch.randn(4, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, K, dtype=torch.float32, device="cuda")
    _assert_close(_gemm(a, b), _ref(a, b), context=f"small_K={K}")


# =================================================================
# K sweep (covers main loop + various tail combinations)
# =================================================================


@pytest.mark.parametrize("K", [2048, 2304, 2880, 3072, 4096, 5120, 7168])
def test_K_sweep(K):
    torch.manual_seed(42)
    a = torch.randn(8, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, K, dtype=torch.float32, device="cuda")
    _assert_close(_gemm(a, b), _ref(a, b), context=f"K={K}")


# =================================================================
# MiniMax-M2 specific shapes
# =================================================================


@pytest.mark.parametrize("M", [1, 4, 8, 16, 32])
def test_minimax_m2(M):
    torch.manual_seed(42)
    a = torch.randn(M, 3072, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(256, 3072, dtype=torch.float32, device="cuda")
    _assert_close(_gemm(a, b), _ref(a, b), context=f"MiniMax-M2 M={M}")


# =================================================================
# Single token (M=1, decode path)
# =================================================================


@pytest.mark.parametrize(
    "N,K",
    [(256, 3072), (256, 7168), (8, 4096)],
    ids=["MiniMax-M2", "DSV3-sized", "Mixtral"],
)
def test_single_token(N, K):
    torch.manual_seed(42)
    a = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.float32, device="cuda")
    out = _gemm(a, b)
    assert out.shape == (1, N)
    _assert_close(out, _ref(a, b), context=f"M=1 {N}x{K}")


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
    a = torch.randn(4, 5120, dtype=torch.bfloat16, device="cuda") * 100
    b = torch.randn(128, 5120, dtype=torch.float32, device="cuda") * 100
    out = _gemm(a, b)
    assert torch.isfinite(out).all()
    _assert_close(out, _ref(a, b), context="large_values")


def test_near_zero():
    torch.manual_seed(42)
    a = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda") * 1e-4
    b = torch.randn(64, 4096, dtype=torch.float32, device="cuda") * 1e-4
    out = _gemm(a, b)
    assert torch.isfinite(out).all()
    assert out.abs().max() < 1.0


def test_zeros():
    a = torch.zeros(4, 2048, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, 2048, dtype=torch.float32, device="cuda")
    assert (_gemm(a, b) == 0).all()


def test_ones():
    a = torch.ones(1, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(32, 4096, dtype=torch.float32, device="cuda")
    _assert_close(_gemm(a, b), _ref(a, b), context="ones")


# =================================================================
# Output dtype
# =================================================================


def test_output_fp32():
    torch.manual_seed(42)
    a = torch.randn(4, 5120, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(128, 5120, dtype=torch.float32, device="cuda")
    assert _gemm(a, b).dtype == torch.float32


# =================================================================
# Determinism
# =================================================================


@pytest.mark.parametrize("M", [1, 4, 8, 16])
@pytest.mark.parametrize("K", [1024, 3072, 7168])
def test_deterministic(M, K):
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, K, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(_gemm(a, b), _gemm(a, b), atol=0, rtol=0)


# =================================================================
# Cross-dtype consistency
# =================================================================


def test_bf16_vs_fp32_activations():
    """bf16 and fp32 activations on same data should roughly agree."""
    torch.manual_seed(42)
    K, N = 4096, 128
    a_bf16 = torch.randn(4, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.float32, device="cuda")
    out_bf16 = _gemm(a_bf16, b)
    a_fp32 = a_bf16.float()
    out_fp32 = _gemm(a_fp32, b)
    _assert_close(out_bf16, out_fp32, min_cos_sim=0.999,
                  context="bf16_vs_fp32_activations")


# =================================================================
# CUDA graph
# =================================================================


@pytest.mark.parametrize("M,K,N", [(4, 3072, 256), (16, 5120, 128)], ids=["MiniMax", "DeepSeek"])
def test_cudagraph(M, K, N):
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.float32, device="cuda")
    _gemm(a, b)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = _gemm(a, b)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    _assert_close(out, _ref(a, b), context=f"cudagraph M={M} K={K}")


def test_cudagraph_20x_replay():
    torch.manual_seed(42)
    a = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, 4096, dtype=torch.float32, device="cuda")
    _gemm(a, b)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = _gemm(a, b)
    results = []
    for _ in range(20):
        g.replay()
        torch.cuda.synchronize()
        results.append(out.clone())
    for i in range(1, len(results)):
        torch.testing.assert_close(results[0], results[i], atol=0, rtol=0,
                                   msg=f"Replay {i} differs")


def test_cudagraph_input_update():
    torch.manual_seed(42)
    a = torch.randn(4, 5120, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(128, 5120, dtype=torch.float32, device="cuda")
    _gemm(a, b)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = _gemm(a, b)
    a.copy_(torch.randn_like(a))
    g.replay()
    torch.cuda.synchronize()
    _assert_close(out, _ref(a, b), context="cudagraph input update")


# =================================================================
# Negative tests
# =================================================================


def test_invalid_weight_dtype():
    a = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, 4096, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        _gemm(a, b)


def test_invalid_device_cpu():
    a = torch.randn(4, 4096, dtype=torch.bfloat16)
    b = torch.randn(64, 4096, dtype=torch.float32)
    with pytest.raises(ValueError):
        _gemm(a, b)


def test_invalid_1d_input():
    a = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, 4096, dtype=torch.float32, device="cuda")
    with pytest.raises((RuntimeError, ValueError, IndexError)):
        _gemm(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
