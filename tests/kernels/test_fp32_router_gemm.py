# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fp32_router_gemm kernel: activation×weight→fp32.

Supported (hidden_size, num_experts) pairs:
  (3072, 256), (6144, 128), (5120, 256)

Correctness baseline: F.linear in float32. Every supported M is covered so
all tuned geometries (wide-block, experts-per-block, token-group; boundaries
at M=4/5, odd/even, M=15/16) are exercised on Blackwell, and the legacy
128/1 geometry everywhere else.
"""

import pytest
import torch

from vllm._custom_ops import fp32_router_gemm
from vllm.platforms import current_platform

# (hidden_size, num_experts)
SHAPES = [(3072, 256), (6144, 128), (5120, 256)]
CASES = [
    (m, hidden, experts)
    for hidden, experts in SHAPES
    for m in range(1, 49 if hidden == 5120 else 33)
]
# Absolute tolerance for fp32 kernel vs float64 reference
ATOL_FP32 = 2e-4
ATOL_BF16 = 2e-2  # bf16 activation has lower precision


def _requires_sm90():
    # ROCm reports a CUDA-like device capability (gfx950 -> (9, 5)), which would
    # pass the SM90 check below for a kernel that is only built for CUDA.
    if not current_platform.is_cuda():
        pytest.skip("fp32_router_gemm is built for CUDA only")
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 90:
        pytest.skip(f"fp32_router_gemm requires SM90+, got SM{major}{minor}")


def _ref(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    """Reference: F.linear in float32 on GPU."""
    return torch.nn.functional.linear(mat_a.float(), mat_b.float())


@pytest.mark.parametrize("num_tokens,hidden_dim,num_experts", CASES)
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


@pytest.mark.parametrize("num_tokens,hidden_dim,num_experts", CASES)
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
def test_topk_routing_consistency(num_tokens: int, hidden_dim: int, num_experts: int):
    """The gate feeds top-k expert selection: the kernel's top-8 must match
    an fp64 reference's top-8 per token (ties tolerated). This is the
    business-level correctness of the router — numeric error only matters
    if it flips the argsort."""
    _requires_sm90()
    top_k = 8
    device = torch.device("cuda")
    for seed in range(5):
        torch.manual_seed(1000 + seed)
        mat_a = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device=device)
        mat_b = torch.randn(num_experts, hidden_dim, dtype=torch.float32, device=device)
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


@pytest.mark.parametrize("hidden_dim,num_experts", SHAPES)
def test_rejects_invalid_inputs(hidden_dim, num_experts):
    """The entry must fail loudly, never compute silently wrong results."""
    _requires_sm90()
    device = torch.device("cuda")
    mat_b = torch.randn(num_experts, hidden_dim, dtype=torch.float32, device=device)

    # Beyond the instantiated range for this shape.
    invalid_tokens = 49 if hidden_dim == 5120 else 33
    with pytest.raises(Exception, match="num_tokens"):
        fp32_router_gemm(
            torch.randn(invalid_tokens, hidden_dim, dtype=torch.float32, device=device),
            mat_b,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "tokens", [*range(1, 17), 17, 24, 32, 33, 40, 48, 56, 64, 65, 128]
)
@torch.inference_mode()
def test_router_retains_fp32_weights_and_logits(default_vllm_config, dist_init, tokens):
    """Low-latency routing must retain FP32 weight precision and logits."""
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability_family(100):
        pytest.skip("SM100 required")
    torch.manual_seed(42)
    router = GateLinear(
        5120, 256, params_dtype=torch.float32, out_dtype=torch.float32
    ).cuda()
    router.weight.normal_(std=0.01)
    x = torch.randn(tokens, 5120, device="cuda", dtype=torch.bfloat16)
    expected = (x.double() @ router.weight.double().T).float()
    actual, bias = router(x)
    assert bias is None and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual.topk(8).indices, expected.topk(8).indices)
    if tokens in (1, 16, 48, 56):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, _ = router(x)
        x.normal_()
        graph.replay()
        expected = (x.double() @ router.weight.double().T).float()
        torch.testing.assert_close(captured, expected, rtol=2e-5, atol=2e-5)
