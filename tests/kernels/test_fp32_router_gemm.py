# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fp32_router_gemm kernel: activation×weight→fp32.

Supported (hidden_size, num_experts) pairs:
  (3072, 256) -> MiniMax-M2/M2.5,  (6144, 128) -> MiniMax-M3,
  (6144, 256) -> GLM-5.2

Correctness baseline: F.linear in float32. Every M in [1, 32] is covered so
all tuned geometries (wide-block, experts-per-block, token-group; boundaries
at M=4/5, odd/even, M=15/16) are exercised on Blackwell, and the legacy
128/1 geometry everywhere else.
"""

import pytest
import torch

from vllm._custom_ops import fp32_router_gemm

# (hidden_size, num_experts)
SHAPES = [(3072, 256), (6144, 128), (6144, 256)]
ALL_M = list(range(1, 33))
# Absolute tolerance for fp32 kernel vs float64 reference
ATOL_FP32 = 2e-4
ATOL_BF16 = 2e-2  # bf16 activation has lower precision


def _requires_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 90:
        pytest.skip(f"fp32_router_gemm requires SM90+, got SM{major}{minor}")


def _ref(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    """Reference: F.linear in float32 on GPU."""
    return torch.nn.functional.linear(mat_a.float(), mat_b.float())


@pytest.mark.parametrize("hidden_dim,num_experts", SHAPES)
@pytest.mark.parametrize("num_tokens", ALL_M)
def test_fp32_activation(num_tokens: int, hidden_dim: int, num_experts: int):
    """fp32 activation → fp32 output should match reference closely."""
    _requires_sm90()
    torch.manual_seed(42)
    device = torch.device("cuda")
    mat_a = torch.randn(num_tokens, hidden_dim, dtype=torch.float32, device=device)
    mat_b = torch.randn(num_experts, hidden_dim, dtype=torch.float32, device=device)

    out = fp32_router_gemm(mat_a, mat_b)
    ref = _ref(mat_a, mat_b)

    assert out.shape == (num_tokens, num_experts)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=ATOL_FP32, rtol=0)


@pytest.mark.parametrize("hidden_dim,num_experts", SHAPES)
@pytest.mark.parametrize("num_tokens", ALL_M)
def test_bf16_activation(num_tokens: int, hidden_dim: int, num_experts: int):
    """bf16 activation → fp32 output should match reference within bf16 error."""
    _requires_sm90()
    torch.manual_seed(42)
    device = torch.device("cuda")
    mat_a_bf16 = torch.randn(
        num_tokens, hidden_dim, dtype=torch.bfloat16, device=device
    )
    mat_b = torch.randn(num_experts, hidden_dim, dtype=torch.float32, device=device)

    out = fp32_router_gemm(mat_a_bf16, mat_b)
    ref = _ref(mat_a_bf16, mat_b).to(device)

    assert out.shape == (num_tokens, num_experts)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=ATOL_BF16, rtol=0)


@pytest.mark.parametrize("hidden_dim,num_experts", SHAPES)
def test_output_shape_and_dtype(hidden_dim: int, num_experts: int):
    """Basic shape and dtype checks."""
    _requires_sm90()
    device = torch.device("cuda")
    mat_a = torch.randn(4, hidden_dim, dtype=torch.float32, device=device)
    mat_b = torch.randn(num_experts, hidden_dim, dtype=torch.float32, device=device)
    out = fp32_router_gemm(mat_a, mat_b)
    assert out.shape == (4, num_experts)
    assert out.dtype == torch.float32
    assert out.device.type == "cuda"


@pytest.mark.parametrize("hidden_dim,num_experts", SHAPES)
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16, 24, 32])
def test_topk_routing_consistency(
    num_tokens: int, hidden_dim: int, num_experts: int
):
    """The gate feeds top-k expert selection: the kernel's top-8 must match
    an fp64 reference's top-8 per token (ties tolerated). This is the
    business-level correctness of the router — numeric error only matters
    if it flips the argsort."""
    _requires_sm90()
    top_k = 8
    device = torch.device("cuda")
    for seed in range(5):
        torch.manual_seed(1000 + seed)
        mat_a = torch.randn(
            num_tokens, hidden_dim, dtype=torch.bfloat16, device=device
        )
        mat_b = torch.randn(
            num_experts, hidden_dim, dtype=torch.float32, device=device
        )
        out = fp32_router_gemm(mat_a, mat_b)
        ref = mat_a.double() @ mat_b.double().t()
        kernel_idx = out.topk(top_k, dim=-1).indices
        ref_vals, ref_idx = ref.topk(top_k, dim=-1)
        for t in range(num_tokens):
            got = set(kernel_idx[t].tolist())
            want = set(ref_idx[t].tolist())
            if got == want:
                continue
            # Tolerate genuine near-ties around the k-th value only.
            kth = ref_vals[t, -1].item()
            for e in got.symmetric_difference(want):
                gap = abs(ref[t, e].item() - kth)
                assert gap < 1e-3, (
                    f"top-{top_k} mismatch beyond tie tolerance: token {t}, "
                    f"expert {e}, gap {gap:.3e}"
                )


def test_zero_tokens_returns_empty():
    """M=0 is a graceful no-op returning an empty [0, E] tensor."""
    _requires_sm90()
    device = torch.device("cuda")
    hidden_dim, num_experts = SHAPES[0]
    mat_a = torch.empty(0, hidden_dim, dtype=torch.float32, device=device)
    mat_b = torch.randn(num_experts, hidden_dim, dtype=torch.float32, device=device)
    out = fp32_router_gemm(mat_a, mat_b)
    assert out.shape == (0, num_experts)
    assert out.dtype == torch.float32


def test_rejects_invalid_inputs():
    """The entry must fail loudly, never compute silently wrong results."""
    _requires_sm90()
    device = torch.device("cuda")
    hidden_dim, num_experts = SHAPES[0]
    mat_b = torch.randn(num_experts, hidden_dim, dtype=torch.float32, device=device)

    # num_tokens > 32 (beyond the instantiated range)
    with pytest.raises(Exception, match="num_tokens"):
        fp32_router_gemm(
            torch.randn(33, hidden_dim, dtype=torch.float32, device=device), mat_b
        )

    # unsupported (hidden_dim, num_experts) pair
    with pytest.raises(Exception, match="supported"):
        fp32_router_gemm(
            torch.randn(4, 1024, dtype=torch.float32, device=device),
            torch.randn(64, 1024, dtype=torch.float32, device=device),
        )

    # non-contiguous activation (a column-slice view)
    wide = torch.randn(4, hidden_dim * 2, dtype=torch.float32, device=device)
    with pytest.raises(Exception, match="contiguous"):
        fp32_router_gemm(wide[:, :hidden_dim], mat_b)

    # wrong weight dtype (bf16 weight is not a supported layout)
    with pytest.raises(Exception, match="float32"):
        fp32_router_gemm(
            torch.randn(4, hidden_dim, dtype=torch.float32, device=device),
            mat_b.to(torch.bfloat16),
        )

    # fp16 activation (only fp32 / bf16 are accepted)
    with pytest.raises(Exception, match="float32 or bfloat16"):
        fp32_router_gemm(
            torch.randn(4, hidden_dim, dtype=torch.float16, device=device), mat_b
        )
