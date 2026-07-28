# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for MXFP4 MoE oracle backend selection on mi355x (GFX950).

These tests run on real hardware — no mocks. Skipped on non-GFX950 platforms.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
)
from vllm.platforms import current_platform

ROCM_AVAILABLE = current_platform.is_rocm()
ROCM_GFX950 = False
ROCM_AITER_AVAILABLE = False

if ROCM_AVAILABLE:
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.platforms.rocm import on_gfx950

    ROCM_GFX950 = on_gfx950()
    ROCM_AITER_AVAILABLE = rocm_aiter_ops.is_fused_moe_enabled()


def _make_w4a4_moe_config(moe_backend: str = "auto") -> FusedMoEConfig:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=256,
        num_local_experts=8,
        num_logical_experts=8,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_backend=moe_backend,
    )


@pytest.fixture
def mxfp4_oracle_config():
    """Stub the config the oracle reads (``model_config.quantization_config``)
    so backend dispatch resolves without a real model / user override."""
    from unittest.mock import patch

    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.get_current_vllm_config"
    ) as mock_get_config:
        mock_get_config.return_value.model_config.quantization_config = None
        yield


@pytest.mark.skipif(not ROCM_GFX950, reason="Requires GFX950 (mi355x)")
@pytest.mark.skipif(not ROCM_AITER_AVAILABLE, reason="Requires AITER enabled")
def test_w4a4_dispatches_to_aiter(mxfp4_oracle_config):
    """With AITER enabled + GFX950, W4A4 selects AITER_MXFP4_MXFP4."""
    config = _make_w4a4_moe_config()
    backend, experts_cls = select_mxfp4_moe_backend(
        config, activation_key=kMxfp4Dynamic
    )
    assert backend == Mxfp4MoeBackend.AITER_MXFP4_MXFP4
    assert experts_cls is not None


@pytest.mark.skipif(not ROCM_GFX950, reason="Requires GFX950 (mi355x)")
@pytest.mark.skipif(
    ROCM_AITER_AVAILABLE,
    reason="Test requires AITER disabled (unset VLLM_ROCM_USE_AITER)",
)
def test_w4a4_falls_back_to_triton_unfused_without_aiter(mxfp4_oracle_config):
    """Without AITER and no --moe-backend, ROCm falls back to TRITON_UNFUSED."""
    config = _make_w4a4_moe_config()
    backend, experts_cls = select_mxfp4_moe_backend(
        config, activation_key=kMxfp4Dynamic
    )
    assert backend == Mxfp4MoeBackend.TRITON_UNFUSED
    assert experts_cls is not None


@pytest.mark.skipif(not ROCM_GFX950, reason="Requires GFX950 (mi355x)")
def test_w4a4_dispatches_to_emulation_with_moe_backend(mxfp4_oracle_config):
    """With --moe-backend emulation, W4A4 selects EMULATION."""
    config = _make_w4a4_moe_config(moe_backend="emulation")
    backend, experts_cls = select_mxfp4_moe_backend(
        config, activation_key=kMxfp4Dynamic
    )
    assert backend == Mxfp4MoeBackend.EMULATION
    assert experts_cls is not None
