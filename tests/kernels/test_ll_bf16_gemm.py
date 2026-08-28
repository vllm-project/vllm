# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for cuteDSL low-latency router GEMM (dot-product + split-K)."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True, scope="module")
def _require_sm90_and_cutedsl():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Requires SM90+ (Hopper/Blackwell)")
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        is_available,
    )

    if not is_available():
        pytest.skip("cuteDSL (CUTLASS Python) not installed")


# ===== Helpers =====


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


def _can_precompile(a, b):
    return (
        a.dim() == 2
        and b.dim() == 2
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and a.device.type == "cuda"
        and b.device.type == "cuda"
        and a.device == b.device
        and a.shape[1] == b.shape[1]
        and a.is_contiguous()
        and b.is_contiguous()
    )


def _gemm(a, b):
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        ll_bf16_gemm,
        ll_bf16_gemm_kernel,
    )

    if _can_precompile(a, b):
        compile_key = ll_bf16_gemm_kernel.dispatch(
            M=a.shape[0], K=a.shape[1], N=b.shape[0]
        )
        ll_bf16_gemm_kernel.compile(compile_key)
    return ll_bf16_gemm(a, b)


# ===== Shapes =====

SHAPES = [
    (256, 7168, "DSV3"),
    (256, 14400, "DSV4-Flash"),
    (128, 5120, "DeepSeek-V2"),
    (8, 4096, "Mixtral-8x7B"),
    (64, 2880, "non-tile-aligned-K"),
    (256, 2048, "split-K-boundary"),
]

SHAPES_SPLITK = [(n, k, d) for n, k, d in SHAPES if k >= 2048]


# =================================================================
# Dot-product kernel (M<=4 or K<2048)
# =================================================================


@pytest.mark.parametrize("M", [1, 2, 3, 4])
@pytest.mark.parametrize("N,K,desc", SHAPES, ids=[s[2] for s in SHAPES])
def test_dotprod(M, N, K, desc):
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    out = _gemm(a, b)
    assert out.dtype == torch.float32
    assert out.shape == (M, N)
    _assert_close(out, _ref(a, b), context=f"dotprod {M}x{N}x{K}")


@pytest.mark.parametrize("M", [1, 4, 8, 16])
def test_dotprod_small_K(M):
    """K<2048 forces dot-product regardless of M."""
    torch.manual_seed(42)
    a = torch.randn(M, 1024, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, 1024, dtype=torch.bfloat16, device="cuda")
    _assert_close(_gemm(a, b), _ref(a, b), context=f"small_K M={M}")


@pytest.mark.parametrize("K", [16, 32, 64, 128, 256, 512, 1024, 1536])
def test_dotprod_K_sweep(K):
    torch.manual_seed(42)
    a = torch.randn(4, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(32, K, dtype=torch.bfloat16, device="cuda")
    _assert_close(_gemm(a, b), _ref(a, b), context=f"K={K}")


# =================================================================
# Split-K kernel (M>4 and K>=2048)
# =================================================================


@pytest.mark.parametrize("M", [5, 6, 8, 12, 16])
@pytest.mark.parametrize("N,K,desc", SHAPES_SPLITK, ids=[s[2] for s in SHAPES_SPLITK])
def test_splitk(M, N, K, desc):
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    out = _gemm(a, b)
    assert out.dtype == torch.float32
    assert out.shape == (M, N)
    _assert_close(out, _ref(a, b), context=f"splitk {M}x{N}x{K}")


@pytest.mark.parametrize("K", [2048, 2304, 2880, 3072, 4096, 5120, 7168, 14400])
def test_splitk_K_sweep(K):
    """Includes non-tile-aligned K (2880) and uneven split (2304)."""
    torch.manual_seed(42)
    a = torch.randn(8, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, K, dtype=torch.bfloat16, device="cuda")
    _assert_close(_gemm(a, b), _ref(a, b), context=f"splitk K={K}")


# =================================================================
# Dispatch boundary (M=4/5, K=2032/2048)
# =================================================================


@pytest.mark.parametrize("M", [4, 5])
@pytest.mark.parametrize("K", [2032, 2048])
def test_dispatch_boundary(M, K):
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, K, dtype=torch.bfloat16, device="cuda")
    path = "splitk" if M > 4 and K >= 2048 else "dotprod"
    _assert_close(_gemm(a, b), _ref(a, b), context=f"M={M} K={K} ({path})")


# =================================================================
# Arbitrary N
# =================================================================


@pytest.mark.parametrize("N", [1, 3, 7, 16, 17, 64, 128, 256, 384])
def test_arbitrary_N_dotprod(N):
    torch.manual_seed(42)
    a = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, 2048, dtype=torch.bfloat16, device="cuda")
    out = _gemm(a, b)
    assert out.shape == (4, N)
    _assert_close(out, _ref(a, b), context=f"dotprod N={N}")


@pytest.mark.parametrize("N", [1, 8, 16, 17, 64, 128, 256])
def test_arbitrary_N_splitk(N):
    torch.manual_seed(42)
    a = torch.randn(8, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, 4096, dtype=torch.bfloat16, device="cuda")
    out = _gemm(a, b)
    assert out.shape == (8, N)
    _assert_close(out, _ref(a, b), context=f"splitk N={N}")


# =================================================================
# Single token (M=1, decode path)
# =================================================================


@pytest.mark.parametrize(
    "N,K",
    [(256, 7168), (256, 14400), (8, 4096), (384, 7168), (264, 6144)],
    ids=["DSV3", "DSV4-Flash", "Mixtral", "DSV4-Pro", "Inkling"],
)
def test_single_token(N, K):
    torch.manual_seed(42)
    a = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    out = _gemm(a, b)
    assert out.shape == (1, N)
    _assert_close(out, _ref(a, b), context=f"M=1 {N}x{K}")


def test_inkling_max_tokens():
    torch.manual_seed(42)
    a = torch.randn(64, 6144, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(264, 6144, dtype=torch.bfloat16, device="cuda")
    out = _gemm(a, b)
    assert out.shape == (64, 264)
    _assert_close(out, _ref(a, b), context="Inkling M=64")


# =================================================================
# Numerical robustness
# =================================================================


@pytest.mark.parametrize("M,K", [(4, 2048), (8, 4096)], ids=["dotprod", "splitk"])
def test_large_values(M, K):
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 100
    b = torch.randn(64, K, dtype=torch.bfloat16, device="cuda") * 100
    out = _gemm(a, b)
    assert torch.isfinite(out).all()
    _assert_close(out, _ref(a, b), context=f"large M={M}")


def test_near_zero():
    torch.manual_seed(42)
    a = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda") * 1e-4
    b = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda") * 1e-4
    out = _gemm(a, b)
    assert torch.isfinite(out).all()
    assert out.abs().max() < 1.0


def test_zeros():
    a = torch.zeros(4, 2048, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda")
    assert (_gemm(a, b) == 0).all()


def test_ones():
    a = torch.ones(1, 2048, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(32, 2048, dtype=torch.bfloat16, device="cuda")
    _assert_close(_gemm(a, b), _ref(a, b), context="ones")


# =================================================================
# Output dtype
# =================================================================


@pytest.mark.parametrize("M,K", [(4, 2048), (8, 4096)], ids=["dotprod", "splitk"])
def test_output_fp32(M, K):
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, K, dtype=torch.bfloat16, device="cuda")
    assert _gemm(a, b).dtype == torch.float32


# =================================================================
# Determinism
# =================================================================


@pytest.mark.parametrize(
    "M,K", [(1, 4096), (4, 4096), (5, 4096), (8, 4096), (16, 4096)]
)
def test_deterministic(M, K):
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(128, K, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(_gemm(a, b), _gemm(a, b), atol=0, rtol=0)


# =================================================================
# Cross-kernel consistency
# =================================================================


@pytest.mark.parametrize("K", [2048, 4096, 7168])
def test_dotprod_vs_splitk(K):
    """First 4 rows from split-K (M=5) match dot-product (M=4)."""
    torch.manual_seed(42)
    a = torch.randn(5, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, K, dtype=torch.bfloat16, device="cuda")
    _assert_close(
        _gemm(a, b)[:4],
        _gemm(a[:4], b),
        min_cos_sim=0.999,
        context=f"cross-kernel K={K}",
    )


# =================================================================
# CUDA graph
# =================================================================


@pytest.mark.parametrize("M,K", [(4, 2048), (8, 4096)], ids=["dotprod", "splitk"])
def test_cudagraph(M, K):
    torch.manual_seed(42)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, K, dtype=torch.bfloat16, device="cuda")
    _gemm(a, b)
    torch.accelerator.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = _gemm(a, b)
    for _ in range(5):
        g.replay()
    torch.accelerator.synchronize()
    _assert_close(out, _ref(a, b), context=f"cudagraph M={M}")


def test_cudagraph_20x_replay():
    torch.manual_seed(42)
    a = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(256, 4096, dtype=torch.bfloat16, device="cuda")
    _gemm(a, b)
    torch.accelerator.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = _gemm(a, b)
    results = []
    for _ in range(20):
        g.replay()
        torch.accelerator.synchronize()
        results.append(out.clone())
    for i in range(1, len(results)):
        torch.testing.assert_close(
            results[0], results[i], atol=0, rtol=0, msg=f"Replay {i} differs"
        )


def test_cudagraph_input_update():
    torch.manual_seed(42)
    a = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda")
    _gemm(a, b)
    torch.accelerator.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = _gemm(a, b)
    a.copy_(torch.randn_like(a))
    g.replay()
    torch.accelerator.synchronize()
    _assert_close(out, _ref(a, b), context="cudagraph input update")


# =================================================================
# GateLinear dispatch integration
# =================================================================


def _make_gate_linear(monkeypatch, *, params_dtype, out_dtype=torch.float32):
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.cute_dsl.ll_bf16.is_available",
        lambda: True,
    )
    return GateLinear(
        input_size=2048,
        output_size=64,
        bias=False,
        out_dtype=out_dtype,
        params_dtype=params_dtype,
    ).cuda()


def test_gate_linear_uses_ll_bf16_for_bf16_fast_path(monkeypatch):
    gate = _make_gate_linear(monkeypatch, params_dtype=torch.bfloat16)
    x = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda")
    calls = []

    def fake_ll_bf16_gemm(hidden_states, router_weight):
        calls.append((hidden_states, router_weight))
        return torch.full(
            (hidden_states.shape[0], router_weight.shape[0]),
            1.0,
            dtype=torch.float32,
            device=hidden_states.device,
        )

    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.cute_dsl.ll_bf16.ll_bf16_gemm",
        fake_ll_bf16_gemm,
    )
    out, bias = gate(x)
    assert bias is None
    assert out.shape == (4, 64)
    assert out.dtype == torch.float32
    assert len(calls) == 1
    assert calls[0][0] is x
    assert calls[0][1] is gate.weight


def test_gate_linear_fp32_weight_falls_back(monkeypatch):
    gate = _make_gate_linear(monkeypatch, params_dtype=torch.float32)
    assert not gate.allow_ll_bf16_gemm
    x = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda")

    def fail_ll_bf16_gemm(hidden_states, router_weight):
        raise AssertionError("ll_bf16_gemm should not run for fp32 weights")

    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.cute_dsl.ll_bf16.ll_bf16_gemm",
        fail_ll_bf16_gemm,
    )
    out, _ = gate(x)
    assert out.shape == (4, 64)
    assert out.dtype == torch.float32


def test_gate_linear_non_bf16_activation_falls_back(monkeypatch):
    gate = _make_gate_linear(monkeypatch, params_dtype=torch.bfloat16)
    x = torch.randn(4, 2048, dtype=torch.float16, device="cuda")

    def fail_ll_bf16_gemm(hidden_states, router_weight):
        raise AssertionError("ll_bf16_gemm should not run for non-bf16 activations")

    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.cute_dsl.ll_bf16.ll_bf16_gemm",
        fail_ll_bf16_gemm,
    )
    out, _ = gate(x)
    assert out.shape == (4, 64)
    assert out.dtype == torch.float32


def test_gate_linear_set_out_dtype_enables_ll_bf16(monkeypatch):
    gate = _make_gate_linear(monkeypatch, params_dtype=torch.bfloat16, out_dtype=None)
    assert not gate.allow_ll_bf16_gemm
    gate.set_out_dtype(torch.float32)
    assert gate.allow_ll_bf16_gemm


def test_gate_linear_non_fp32_out_dtype_disables_ll_bf16(monkeypatch):
    gate = _make_gate_linear(monkeypatch, params_dtype=torch.bfloat16, out_dtype=None)
    gate.set_out_dtype(torch.bfloat16)
    assert not gate.allow_ll_bf16_gemm


def test_gate_linear_m_gt_16_falls_back(monkeypatch):
    gate = _make_gate_linear(monkeypatch, params_dtype=torch.bfloat16)
    x = torch.randn(17, 2048, dtype=torch.bfloat16, device="cuda")

    def fail_ll_bf16_gemm(hidden_states, router_weight):
        raise AssertionError("ll_bf16_gemm should not run for M > 16")

    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.cute_dsl.ll_bf16.ll_bf16_gemm",
        fail_ll_bf16_gemm,
    )
    out, _ = gate(x)
    assert out.shape == (17, 64)
    assert out.dtype == torch.float32


# =================================================================
# Negative tests — invalid inputs
# =================================================================


@pytest.mark.parametrize(
    "M,K,N,dtype",
    [
        pytest.param(4, 2048, 64, torch.float32, id="fp32_input"),
        pytest.param(4, 2048, 64, torch.float16, id="fp16_input"),
        pytest.param(8, 4096, 64, torch.float32, id="fp32_splitk_path"),
    ],
)
def test_invalid_dtype(M, K, N, dtype):
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(N, K, device="cuda", dtype=dtype)
    with pytest.raises(ValueError, match="dtype=bfloat16"):
        _gemm(a, b)


def test_invalid_device_cpu():
    a = torch.randn(4, 2048, dtype=torch.bfloat16)
    b = torch.randn(64, 2048, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="device_type=cuda"):
        _gemm(a, b)


def test_invalid_1d_input():
    a = torch.randn(2048, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="2D tensors"):
        _gemm(a, b)


def test_mismatched_K():
    a = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, 1024, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="matching K dimensions"):
        _gemm(a, b)


def test_invalid_K_divisibility():
    a = torch.randn(4, 2049, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, 2049, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(ValueError, match="K to be divisible by 8"):
        _gemm(a, b)


def test_non_contiguous_input():
    a = torch.randn(2048, 4, dtype=torch.bfloat16, device="cuda").T
    b = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda")
    assert not a.is_contiguous()
    with pytest.raises(ValueError, match="contiguous row-major"):
        _gemm(a, b)


def test_invalid_output_dtype():
    a = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="output_dtype=torch.float32"):
        from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm

        ll_bf16_gemm(a, b, output_dtype=torch.bfloat16)


def test_cache_miss_compiles_dotprod():
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import LLBf16Gemm

    torch.manual_seed(42)
    a = torch.randn(3, 64, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(17, 64, dtype=torch.bfloat16, device="cuda")
    kernel = LLBf16Gemm()
    out = kernel(a, b)
    assert out.shape == (3, 17)
    _assert_close(out, _ref(a, b), context="cache miss dotprod")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
