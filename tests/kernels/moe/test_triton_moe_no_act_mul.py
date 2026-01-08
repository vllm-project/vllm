# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test for is_act_and_mul=False MoE using Triton.

This tests the code path used by models like Nemotron-H that use
non-fused activations (e.g., relu2_no_mul) instead of SwiGLU-style
fused activations.

This feature is supported on both CUDA and ROCm (with AITER disabled).
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Tests for is_act_and_mul=False MoE require CUDA or ROCm",
)


@pytest.fixture
def disable_aiter_on_rocm(monkeypatch):
    """Fixture to disable AITER on ROCm to use Triton path."""
    if current_platform.is_rocm():
        from vllm._aiter_ops import rocm_aiter_ops

        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "0")
        rocm_aiter_ops.refresh_env_variables()

        yield

        rocm_aiter_ops.refresh_env_variables()
    else:
        # On CUDA, no special setup needed
        yield


@pytest.fixture
def init_workspace():
    """Initialize workspace manager for MoE tests."""
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        reset_workspace_manager,
    )

    torch.manual_seed(42)
    init_workspace_manager(torch.cuda.current_device())

    yield

    reset_workspace_manager()


@pytest.mark.parametrize("m", [1, 33, 64, 222])
@pytest.mark.parametrize("n", [128, 256, 1024])
@pytest.mark.parametrize("k", [128, 512])
@pytest.mark.parametrize("e", [4, 8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("activation", ["relu2_no_mul", "silu_no_mul", "gelu_no_mul"])
@torch.inference_mode()
def test_moe_no_act_mul(
    disable_aiter_on_rocm,
    init_workspace,
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    activation: str,
):
    """Test MoE with is_act_and_mul=False using Triton."""
    from vllm.model_executor.layers.fused_moe import TritonExperts, fused_topk
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEModularKernel,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNoEP,
    )

    a = torch.randn((m, k), device="cuda", dtype=dtype)
    w1 = torch.randn((e, n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    quant_config = FusedMoEQuantConfig.make(is_act_and_mul=False)

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=True)

    fused_experts = FusedMoEModularKernel(
        MoEPrepareAndFinalizeNoEP(),
        TritonExperts(quant_config),
    )

    output = fused_experts(
        hidden_states=a,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=activation,
    )

    assert output.shape == (m, k), f"Expected shape {(m, k)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    assert output.abs().sum() > 0, "Output is all zeros"


@torch.inference_mode()
def test_moe_workspace_shapes_no_act_mul(disable_aiter_on_rocm):
    """Test workspace_shapes returns correct sizes for is_act_and_mul=False."""
    from vllm.model_executor.layers.fused_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

    M, N, K, topk = 64, 256, 128, 2

    quant_config = FusedMoEQuantConfig.make(is_act_and_mul=False)
    experts = TritonExperts(quant_config)
    ws1, ws2, out = experts.workspace_shapes(M, N, K, topk, 8, 8, None)

    assert ws1[2] == max(N, K)
    assert out == (M, K)
