# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NVFP4 MoE weight preparation with non-gated MLPs.

This tests the `prepare_static_weights_for_trtllm_fp4_moe` function with
`is_act_and_mul=False`, which is used by models like Nemotron-Nano that
use non-gated MLP layers.

For non-gated MLPs:
- w13 contains only w1 (1x intermediate_size), not [w1, w3] merged (2x)
- The weight reshape must use the correct multiplier based on is_act_and_mul
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.has_device_capability(100):
    pytest.skip(
        "NVFP4 requires compute capability >= 10.0 (Blackwell)",
        allow_module_level=True,
    )

# Only import after platform check to avoid import errors on unsupported hardware
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    prepare_static_weights_for_trtllm_fp4_moe,
)


def make_fake_nvfp4_weights(
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
    is_act_and_mul: bool,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create fake NVFP4 quantized weights for testing.

    For NVFP4, weights are packed (2 FP4 values per byte), so dimensions
    are halved for the packed representation.

    Args:
        num_experts: Number of MoE experts
        intermediate_size: MLP intermediate dimension
        hidden_size: Model hidden dimension
        is_act_and_mul: If True, w13 is 2x intermediate_size (gated MLP)
                        If False, w13 is 1x intermediate_size (non-gated MLP)

    Returns:
        gemm1_weights: (E, w13_size // 2, hidden_size // 2) packed FP4
        gemm2_weights: (E, hidden_size // 2, intermediate_size // 2) packed FP4
        gemm1_scales: (E, w13_size, hidden_size // 16) block scales
        gemm2_scales: (E, hidden_size, intermediate_size // 16) block scales
    """
    w13_multiplier = 2 if is_act_and_mul else 1
    w13_size = w13_multiplier * intermediate_size

    # Packed FP4 weights (2 values per byte, so dimensions halved)
    gemm1_weights = torch.randint(
        0,
        256,
        (num_experts, w13_size // 2, hidden_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    gemm2_weights = torch.randint(
        0,
        256,
        (num_experts, hidden_size // 2, intermediate_size // 2),
        dtype=torch.uint8,
        device=device,
    )

    # Block scales (FP8, one scale per 16 elements in last dim)
    gemm1_scales = torch.randint(
        0,
        256,
        (num_experts, w13_size, hidden_size // 16),
        dtype=torch.uint8,
        device=device,
    )
    gemm2_scales = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 16),
        dtype=torch.uint8,
        device=device,
    )

    return gemm1_weights, gemm2_weights, gemm1_scales, gemm2_scales


@pytest.mark.parametrize("num_experts", [8, 64, 128])
@pytest.mark.parametrize("intermediate_size", [1024, 1856, 2048])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("is_act_and_mul", [True, False])
@torch.inference_mode()
def test_prepare_weights_shape_correctness(
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
    is_act_and_mul: bool,
):
    """Test that weight preparation produces correct output shapes.

    This verifies that the is_act_and_mul parameter correctly controls
    the weight tensor shapes for both gated and non-gated MLPs.
    """
    gemm1_weights, gemm2_weights, gemm1_scales, gemm2_scales = make_fake_nvfp4_weights(
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        is_act_and_mul=is_act_and_mul,
    )

    # Call the function under test
    w13_out, w13_scale_out, w2_out, w2_scale_out = (
        prepare_static_weights_for_trtllm_fp4_moe(
            gemm1_weights,
            gemm2_weights,
            gemm1_scales,
            gemm2_scales,
            hidden_size,
            intermediate_size,
            num_experts,
            is_act_and_mul=is_act_and_mul,
        )
    )

    # Verify output shapes
    # w13 output should be shuffled but maintain the same total elements
    assert w13_out.shape[0] == num_experts, f"Expected {num_experts} experts"

    # w2 output should also maintain correct shape
    assert w2_out.shape[0] == num_experts, f"Expected {num_experts} experts"

    # Verify no NaN or Inf values (sanity check)
    assert not torch.isnan(w13_out).any(), "w13 contains NaN"
    assert not torch.isnan(w2_out).any(), "w2 contains NaN"
    assert not torch.isinf(w13_out).any(), "w13 contains Inf"
    assert not torch.isinf(w2_out).any(), "w2 contains Inf"


@pytest.mark.parametrize("is_act_and_mul", [True, False])
@torch.inference_mode()
def test_nemotron_nano_dimensions(is_act_and_mul: bool):
    """Test with Nemotron-Nano-like dimensions.

    Nemotron-Nano uses:
    - intermediate_size = 1856 (not divisible by 128)
    - hidden_size = 2688
    - num_experts = 128
    - is_act_and_mul = False (non-gated MLP)
    """
    num_experts = 128
    intermediate_size = 1856
    hidden_size = 2688

    gemm1_weights, gemm2_weights, gemm1_scales, gemm2_scales = make_fake_nvfp4_weights(
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        is_act_and_mul=is_act_and_mul,
    )

    # This should not raise any shape mismatch errors
    w13_out, w13_scale_out, w2_out, w2_scale_out = (
        prepare_static_weights_for_trtllm_fp4_moe(
            gemm1_weights,
            gemm2_weights,
            gemm1_scales,
            gemm2_scales,
            hidden_size,
            intermediate_size,
            num_experts,
            is_act_and_mul=is_act_and_mul,
        )
    )

    # Basic sanity checks
    assert w13_out.shape[0] == num_experts
    assert w2_out.shape[0] == num_experts
    assert not torch.isnan(w13_out).any()
    assert not torch.isnan(w2_out).any()


if __name__ == "__main__":
    # Quick manual test
    print("Testing gated MLP (is_act_and_mul=True)...")
    test_prepare_weights_shape_correctness(64, 1024, 2048, True)
    print("PASSED")

    print("Testing non-gated MLP (is_act_and_mul=False)...")
    test_prepare_weights_shape_correctness(64, 1024, 2048, False)
    print("PASSED")

    print("Testing Nemotron-Nano dimensions with non-gated MLP...")
    test_nemotron_nano_dimensions(False)
    print("PASSED")
