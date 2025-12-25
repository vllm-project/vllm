# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Sonic MoE integration.

Sonic MoE is a high-performance MoE implementation optimized for Hopper GPUs.
These tests verify correct integration with vLLM's modular MoE infrastructure.

Requires:
- sonicmoe package installed
- NVIDIA Hopper GPU (H100/H200)
"""

import pytest
import torch

from tests.kernels.utils import torch_experts
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform

# Check for Sonic MoE availability
try:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEModularKernel,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNoEP,
    )
    from vllm.model_executor.layers.fused_moe.sonic_moe import (
        SonicMoEExperts,
        is_sonic_moe_supported,
        sonic_moe,
    )

    SONICMOE_AVAILABLE = True
except ImportError:
    SONICMOE_AVAILABLE = False


def _is_hopper_gpu() -> bool:
    """Check if running on Hopper GPU."""
    if not current_platform.is_cuda():
        return False
    # Sonic MoE is only supported on Hopper (SM90)
    return current_platform.is_device_capability(90)


# Skip entire module if requirements not met
if not SONICMOE_AVAILABLE or not _is_hopper_gpu():
    pytest.skip(
        "Requires sonicmoe package and Hopper GPU (H100/H200)",
        allow_module_level=True,
    )


# Test configurations
MNK_FACTORS = [
    (32, 1024, 1024),
    (64, 2048, 1024),
    (128, 1536, 1024),
    (256, 1024, 1536),
]


def make_sonic_quant_config(
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> FusedMoEQuantConfig:
    """Create a quantization config for Sonic MoE (no quantization)."""
    return FusedMoEQuantConfig(
        quant_dtype=None,
        per_act_token_quant=False,
        per_out_ch_quant=False,
        block_shape=None,
        a1_scale=None,
        a2_scale=None,
        w1_scale=None,
        w2_scale=None,
    )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [8, 16, 64])
@pytest.mark.parametrize("topk", [1, 2, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_sonic_moe_basic(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    workspace_init,
):
    """Test basic Sonic MoE functionality."""
    current_platform.seed_everything(42)

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        # Create input tensor
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        # Create expert weights (GLU activation doubles intermediate size)
        # vLLM format: [E, N, K]
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

        # Check if Sonic MoE supports this configuration
        if not is_sonic_moe_supported(a, w1, w2):
            pytest.skip("Sonic MoE not supported for this configuration")

        # Create routing scores and compute topk
        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        # Create quant config (no quantization)
        quant_config = make_sonic_quant_config(e, n, k, dtype)

        # Run Sonic MoE
        sonic_experts = FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(defer_input_quant=True),
            SonicMoEExperts(quant_config),
        )

        sonic_output = sonic_experts(
            hidden_states=a,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="silu",
        )

        # Reference implementation using torch_experts with same topk_weights/ids
        torch_output = torch_experts(
            a, w1, w2, topk_weights, topk_ids, activation="silu_and_mul"
        )

        # Compare outputs (relaxed tolerance due to different computation order)
        torch.testing.assert_close(torch_output, sonic_output, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("m,n,k", [(64, 1024, 1024)])
@pytest.mark.parametrize("e", [16])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_sonic_moe_dtypes(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    workspace_init,
):
    """Test Sonic MoE with different data types."""
    current_platform.seed_everything(123)

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

        if not is_sonic_moe_supported(a, w1, w2):
            pytest.skip("Sonic MoE not supported for this configuration")

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        quant_config = make_sonic_quant_config(e, n, k, dtype)

        # Run using the high-level API
        sonic_output = sonic_moe(
            hidden_states=a,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_config=quant_config,
            activation="silu",
        )

        # Basic sanity checks
        assert sonic_output.shape == (m, k)
        assert sonic_output.dtype == dtype
        assert not torch.isnan(sonic_output).any()
        assert not torch.isinf(sonic_output).any()


@pytest.mark.parametrize("m", [1, 32, 256, 1024])
@torch.inference_mode()
def test_sonic_moe_variable_batch_sizes(
    m: int,
    workspace_init,
):
    """Test Sonic MoE with different batch sizes."""
    n, k, e, topk = 1024, 1024, 16, 2
    dtype = torch.bfloat16

    current_platform.seed_everything(456)

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

        if not is_sonic_moe_supported(a, w1, w2):
            pytest.skip("Sonic MoE not supported for this configuration")

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        quant_config = make_sonic_quant_config(e, n, k, dtype)

        sonic_output = sonic_moe(
            hidden_states=a,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_config=quant_config,
            activation="silu",
        )

        assert sonic_output.shape == (m, k)
        assert not torch.isnan(sonic_output).any()


@torch.inference_mode()
def test_sonic_moe_support_check(workspace_init):
    """Test the is_sonic_moe_supported function."""
    dtype = torch.bfloat16
    m, n, k, e = 64, 1024, 1024, 16

    # Valid configuration
    a_valid = torch.randn((m, k), device="cuda", dtype=dtype)
    w1_valid = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype)
    w2_valid = torch.randn((e, k, n), device="cuda", dtype=dtype)

    # Should be supported on Hopper with bfloat16
    if _is_hopper_gpu():
        assert is_sonic_moe_supported(a_valid, w1_valid, w2_valid)

    # Invalid dtype (float64)
    a_invalid = torch.randn((m, k), device="cuda", dtype=torch.float64)
    assert not is_sonic_moe_supported(a_invalid, w1_valid, w2_valid)

    # Invalid weight dimensions (2D instead of 3D)
    w1_invalid = torch.randn((n, k), device="cuda", dtype=dtype)
    assert not is_sonic_moe_supported(a_valid, w1_invalid, w2_valid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
