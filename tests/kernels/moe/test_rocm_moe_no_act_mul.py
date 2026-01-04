# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test for is_act_and_mul=False MoE on ROCm with Triton.

This tests the code path used by models like Nemotron-H that use
non-fused activations (e.g., relu2_no_mul) instead of SwiGLU-style
fused activations.

These tests only run on ROCm with AITER disabled.
"""

import importlib
import sys

import pytest
import torch

from vllm.platforms import current_platform

# Skip entire module if not ROCm
pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm-specific tests for is_act_and_mul=False MoE",
)


@pytest.fixture
def rocm_no_aiter(monkeypatch):
    """Fixture to disable AITER and use Triton on ROCm."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "0")

    # Force reload to pick up new env vars
    from vllm._aiter_ops import rocm_aiter_ops

    if "rocm_aiter_ops" in sys.modules:
        importlib.reload(rocm_aiter_ops)

    yield

    # Reload again to restore default state
    if "rocm_aiter_ops" in sys.modules:
        importlib.reload(rocm_aiter_ops)


@pytest.fixture
def init_workspace():
    """Initialize workspace manager for MoE tests."""
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.manual_seed(42)
    init_workspace_manager(torch.cuda.current_device())


@pytest.mark.parametrize("m", [1, 33, 64, 222])
@pytest.mark.parametrize("n", [128, 256, 1024])
@pytest.mark.parametrize("k", [128, 512])
@pytest.mark.parametrize("e", [4, 8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("activation", ["relu2_no_mul", "silu_no_mul", "gelu_no_mul"])
@torch.inference_mode()
def test_rocm_moe_no_act_mul(
    rocm_no_aiter,
    init_workspace,
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    activation: str,
):
    """
    Test MoE with is_act_and_mul=False on ROCm using Triton.

    This tests the workspace sizing and activation handling for non-fused
    activations like relu2_no_mul used by Nemotron-H.
    """
    from vllm.model_executor.layers.fused_moe import TritonExperts, fused_topk
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEModularKernel,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNoEP,
    )

    # For is_act_and_mul=False, w1 has shape (e, n, k) where n = intermediate_size
    # (not 2*intermediate_size as in fused activations)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    w1 = torch.randn((e, n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    # Create quant config with is_act_and_mul=False
    quant_config = FusedMoEQuantConfig.make(is_act_and_mul=False)

    # Create routing
    score = torch.randn((m, e), device="cuda", dtype=dtype)
    topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=True)

    # Create modular kernel with TritonExperts
    fused_experts = FusedMoEModularKernel(
        MoEPrepareAndFinalizeNoEP(),
        TritonExperts(quant_config),
    )

    # Run forward pass - this should not crash with workspace sizing issues
    output = fused_experts(
        hidden_states=a,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=activation,
    )

    # Basic shape check
    assert output.shape == (m, k), f"Expected shape {(m, k)}, got {output.shape}"

    # Output should not be all zeros or NaN
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    assert output.abs().sum() > 0, "Output is all zeros"


@torch.inference_mode()
def test_rocm_moe_workspace_shapes_no_act_mul(rocm_no_aiter):
    """Test workspace_shapes returns correct sizes for is_act_and_mul=False."""
    from vllm.model_executor.layers.fused_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

    M, N, K, topk = 64, 256, 128, 2

    quant_config = FusedMoEQuantConfig.make(is_act_and_mul=False)
    experts = TritonExperts(quant_config)
    ws1, ws2, out = experts.workspace_shapes(M, N, K, topk, 8, 8, None)

    # For non-fused: workspace1 last dim = max(N, K) = 256
    assert ws1[2] == max(N, K)
    assert out == (M, K)
