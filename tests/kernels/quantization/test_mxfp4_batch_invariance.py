# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests that the mxfp4 TRITON MoE backend propagates the batch invariance
flag into PrecisionConfig.enforce_bitwise_invariance.

When VLLM_BATCH_INVARIANT=1, the matmul_ogs kernel must use deterministic
accumulation order (block_m=128, split_k=1). This test verifies that
Mxfp4MoEMethod.process_weights_after_loading sets enforce_bitwise_invariance
correctly on both w13 and w2 PrecisionConfig objects.
"""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers.quantization.mxfp4 import (
    Mxfp4Backend,
    Mxfp4MoEMethod,
)


@dataclass
class FakePrecisionConfig:
    """Stand-in for triton_kernels.matmul_ogs.PrecisionConfig."""

    weight_scale: Any = None
    flex_ctx: Any = None
    enforce_bitwise_invariance: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeFlexCtx:
    """Stand-in for triton_kernels.matmul_ogs.FlexCtx."""

    rhs_data: Any = None


def _make_mock_layer(num_experts: int = 8, intermediate: int = 128, hidden: int = 64):
    """Create a mock layer with the tensors needed by the TRITON branch."""
    layer = MagicMock()
    # Weight tensors (uint8 to simulate mxfp4 packed weights)
    layer.w13_weight = torch.nn.Parameter(
        torch.randint(
            0, 255, (num_experts, 2 * intermediate, hidden), dtype=torch.uint8
        )
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.randint(0, 255, (num_experts, hidden, intermediate), dtype=torch.uint8)
    )
    # Scale tensors
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.randint(
            0, 255, (num_experts, 2 * intermediate, hidden // 32), dtype=torch.uint8
        )
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.randint(
            0, 255, (num_experts, hidden, intermediate // 32), dtype=torch.uint8
        )
    )
    # Bias tensors
    layer.w13_bias = torch.nn.Parameter(
        torch.randn(num_experts, 2 * intermediate, dtype=torch.bfloat16)
    )
    layer.w2_bias = torch.nn.Parameter(
        torch.randn(num_experts, hidden, dtype=torch.bfloat16)
    )
    return layer


def _make_mock_moe_config() -> MagicMock:
    """Create a mock FusedMoEConfig for TRITON backend."""
    parallel_config = MagicMock()
    parallel_config.ep_size = 1

    moe_config = MagicMock()
    moe_config.ep_size = 1
    moe_config.is_lora_enabled = False
    moe_config.moe_parallel_config = parallel_config
    moe_config.use_deepep_ll_kernels = False
    return moe_config


class TestMxfp4BatchInvariance:
    """Verify that Mxfp4MoEMethod propagates batch invariance to
    PrecisionConfig.enforce_bitwise_invariance in the TRITON backend."""

    @pytest.mark.parametrize(
        "batch_invariant_enabled",
        [True, False],
        ids=["batch-invariant-on", "batch-invariant-off"],
    )
    @patch(
        "vllm.model_executor.layers.quantization.mxfp4.get_mxfp4_backend",
    )
    @patch(
        "vllm.model_executor.layers.quantization.mxfp4.get_current_vllm_config",
    )
    def test_precision_config_enforce_bitwise_invariance(
        self,
        mock_get_config,
        mock_get_backend,
        batch_invariant_enabled,
    ):
        """PrecisionConfig.enforce_bitwise_invariance should match
        vllm_is_batch_invariant() for both w13 and w2 configs."""
        # Set up mocks for Mxfp4MoEMethod.__init__
        mock_get_backend.return_value = Mxfp4Backend.TRITON

        mock_compilation_config = MagicMock()
        mock_compilation_config.max_cudagraph_capture_size = 1024
        mock_vllm_config = MagicMock()
        mock_vllm_config.compilation_config = mock_compilation_config
        mock_get_config.return_value = mock_vllm_config

        moe_config = _make_mock_moe_config()
        method = Mxfp4MoEMethod(moe_config)

        layer = _make_mock_layer()

        # Dummy return from _swizzle_mxfp4: (weight, flex_data, scale)
        dummy_weight = torch.zeros(1)
        dummy_flex = MagicMock()
        dummy_scale = MagicMock()
        swizzle_return = (dummy_weight, dummy_flex, dummy_scale)

        with (
            patch(
                "vllm.model_executor.layers.quantization.mxfp4._swizzle_mxfp4",
                return_value=swizzle_return,
            ),
            patch(
                "vllm.model_executor.layers.quantization.mxfp4.vllm_is_batch_invariant",
                return_value=batch_invariant_enabled,
            ),
            patch.dict(
                "sys.modules",
                {
                    "triton_kernels": MagicMock(),
                    "triton_kernels.matmul_ogs": MagicMock(
                        PrecisionConfig=FakePrecisionConfig,
                        FlexCtx=FakeFlexCtx,
                    ),
                },
            ),
        ):
            method.process_weights_after_loading(layer)

        assert hasattr(method, "w13_precision_config"), (
            "w13_precision_config was not set — TRITON branch may not have run"
        )
        assert hasattr(method, "w2_precision_config"), (
            "w2_precision_config was not set — TRITON branch may not have run"
        )

        assert (
            method.w13_precision_config.enforce_bitwise_invariance
            == batch_invariant_enabled
        ), (
            f"w13 enforce_bitwise_invariance should be {batch_invariant_enabled}, "
            f"got {method.w13_precision_config.enforce_bitwise_invariance}"
        )
        assert (
            method.w2_precision_config.enforce_bitwise_invariance
            == batch_invariant_enabled
        ), (
            f"w2 enforce_bitwise_invariance should be {batch_invariant_enabled}, "
            f"got {method.w2_precision_config.enforce_bitwise_invariance}"
        )
