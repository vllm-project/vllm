# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fp32_router_gemm kernel: activation×weight→fp32, H=3072, E=256.

Correctness baseline: torch.matmul in float64.
"""

import pytest
import torch

from vllm._custom_ops import fp32_router_gemm

NUM_EXPERTS = 256
HIDDEN_DIM = 3072
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


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16, 32])
def test_fp32_activation(num_tokens: int):
    """fp32 activation → fp32 output should match reference closely."""
    _requires_sm90()
    torch.manual_seed(42)
    device = torch.device("cuda")
    mat_a = torch.randn(num_tokens, HIDDEN_DIM, dtype=torch.float32, device=device)
    mat_b = torch.randn(NUM_EXPERTS, HIDDEN_DIM, dtype=torch.float32, device=device)

    out = fp32_router_gemm(mat_a, mat_b)
    ref = _ref(mat_a, mat_b)

    assert out.shape == (num_tokens, NUM_EXPERTS)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=ATOL_FP32, rtol=0)


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16, 32])
def test_bf16_activation(num_tokens: int):
    """bf16 activation → fp32 output should match reference within bf16 error."""
    _requires_sm90()
    torch.manual_seed(42)
    device = torch.device("cuda")
    mat_a_bf16 = torch.randn(
        num_tokens, HIDDEN_DIM, dtype=torch.bfloat16, device=device
    )
    mat_b = torch.randn(NUM_EXPERTS, HIDDEN_DIM, dtype=torch.float32, device=device)

    out = fp32_router_gemm(mat_a_bf16, mat_b)
    ref = _ref(mat_a_bf16, mat_b).to(device)

    assert out.shape == (num_tokens, NUM_EXPERTS)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=ATOL_BF16, rtol=0)


def test_output_shape_and_dtype():
    """Basic shape and dtype checks."""
    _requires_sm90()
    device = torch.device("cuda")
    mat_a = torch.randn(4, HIDDEN_DIM, dtype=torch.float32, device=device)
    mat_b = torch.randn(NUM_EXPERTS, HIDDEN_DIM, dtype=torch.float32, device=device)
    out = fp32_router_gemm(mat_a, mat_b)
    assert out.shape == (4, NUM_EXPERTS)
    assert out.dtype == torch.float32
    assert out.device.type == "cuda"
