# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test for NVFP4 GEMM NaN propagation from padding positions.

This test validates that NaNs in padding positions (from attention softmax 0/0)
do not leak into real token positions during FP4 quantization and GEMM.
"""
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    NvFp4LinearBackend,
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm, has_flashinfer
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="NVFP4 requires compute capability 100 or above (Blackwell+).",
        allow_module_level=True,
    )

if not has_flashinfer():
    pytest.skip(
        reason="FlashInfer is required for NVFP4 GEMM tests.",
        allow_module_level=True,
    )

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def create_nvfp4_weight(output_size: int, input_size: int, dtype: torch.dtype,
                        device: str) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    """Create random FP4 weights and scales for testing."""
    # Create random bf16 weights
    weight_bf16 = torch.randn(output_size, input_size, dtype=dtype, device=device)

    # Compute global scale
    weight_amax = torch.abs(weight_bf16).max().to(torch.float32)
    weight_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / weight_amax).to(
        torch.float32)

    # Quantize to FP4
    weight_fp4, weight_blockscale = ops.scaled_fp4_quant(
        weight_bf16, weight_global_scale, is_sf_swizzled_layout=True)

    # Swizzle block scales for CUTLASS kernel
    weight_scale_swizzled = swizzle_blockscale(
        weight_blockscale.view(torch.float8_e4m3fn))

    # Pad weight for CUTLASS alignment
    weight_fp4_padded, weights_padding_cols = pad_nvfp4_weight_for_cutlass(
        weight_fp4)

    return weight_fp4_padded, weight_scale_swizzled, weight_global_scale, weights_padding_cols


@pytest.mark.parametrize("num_tokens", [32])
@pytest.mark.parametrize("num_padding", [8])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("output_size", [1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("nan_placement", ["end"])
@pytest.mark.parametrize("use_buggy_path", [True, False])
@torch.inference_mode()
def test_nvfp4_gemm_nan_isolation(
    num_tokens: int,
    num_padding: int,
    hidden_size: int,
    output_size: int,
    dtype: torch.dtype,
    nan_placement: str,
    use_buggy_path: bool,
) -> None:
    """
    Test that NaNs in padding positions don't leak into real token positions.

    Simulates the scenario where attention softmax produces NaN at padding
    positions (0/0), which then flows through o_proj's NVFP4 GEMM.

    Args:
        num_tokens: Number of real (non-padding) tokens
        num_padding: Number of padding tokens with NaN
        hidden_size: Input dimension (K)
        output_size: Output dimension (N)
        dtype: Input data type
        nan_placement: Where to place NaN tokens ("end", "middle", "scattered")
        use_buggy_path: If True, don't mask NaNs before quantization (buggy).
                        If False, mask NaNs before quantization (fixed).
    """
    set_random_seed(42)
    device = "cuda:0"
    torch.set_default_device(device)

    total_tokens = num_tokens + num_padding

    # Create input with NaNs at padding positions
    x = torch.randn(total_tokens, hidden_size, dtype=dtype, device=device)

    # Create a mask: 1 for real tokens, 0 for padding
    mask = torch.ones(total_tokens, dtype=torch.bool, device=device)

    # Inject NaNs at padding positions based on placement strategy
    if nan_placement == "end":
        # NaNs at the end (most common case)
        x[num_tokens:, :] = float('nan')
        mask[num_tokens:] = False
    elif nan_placement == "middle":
        # NaNs in the middle
        mid_start = num_tokens // 2
        x[mid_start:mid_start + num_padding, :] = float('nan')
        mask[mid_start:mid_start + num_padding] = False
    elif nan_placement == "scattered":
        # Scattered NaN positions
        nan_indices = torch.randperm(total_tokens)[:num_padding]
        x[nan_indices, :] = float('nan')
        mask[nan_indices] = False

    # Verify NaNs are present at padding positions
    assert torch.isnan(x[~mask]).all(), "NaN injection failed"
    assert not torch.isnan(x[mask]).any(), "Real tokens should not have NaN"

    # Create FP4 weights
    weight_fp4, weight_scale, weight_global_scale, weights_padding_cols = \
        create_nvfp4_weight(output_size, hidden_size, dtype, device)

    # Compute input global scale
    # Always use clean tokens for global scale (even in buggy path)
    # because NaN global scale would make everything NaN
    input_amax = torch.abs(x[mask]).max().to(torch.float32)
    input_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / input_amax).to(
        torch.float32)
    input_global_scale_inv = 1.0 / input_global_scale
    alpha = (input_global_scale * weight_global_scale).to(torch.float32)

    # **KEY DIFFERENCE**: Buggy path vs fixed path
    if use_buggy_path:
        # BUGGY: Pass input with NaNs directly to quantization
        # This allows NaNs to contaminate block scales
        x_to_quantize = x
    else:
        # FIXED: Mask NaNs before quantization
        # This prevents NaNs from contaminating block scales
        x_to_quantize = torch.where(torch.isnan(x), torch.zeros_like(x), x)

    # Quantize input to FP4 (this is where NaN propagation can happen)
    x_fp4, x_blockscale = ops.scaled_fp4_quant(
        x_to_quantize, input_global_scale_inv, is_sf_swizzled_layout=True,
        backend="flashinfer-cutlass")

    # Pad activations to match weight K-dimension padding
    x_fp4_padded = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

    # Run the FP4 GEMM (FlashInfer CUTLASS backend)
    output = flashinfer_scaled_fp4_mm(
        x_fp4_padded,
        weight_fp4,
        x_blockscale,
        weight_scale,
        alpha,
        dtype,
        backend="cutlass",
    )

    # Slice output to remove N-dimension padding
    output = slice_nvfp4_output(output, output_size)

    # Check for NaN propagation
    real_output = output[mask]  # Real token outputs
    padding_output = output[~mask]  # Padding token outputs

    has_nan_in_real = torch.isnan(real_output).any()
    has_nan_in_padding = torch.isnan(padding_output).any()

    # Collect statistics for debugging
    if has_nan_in_real:
        num_nan_elements = torch.isnan(real_output).sum().item()
        total_real_elements = real_output.numel()
        nan_percentage = 100.0 * num_nan_elements / total_real_elements

        if use_buggy_path:
            # Expected to fail on buggy path - this confirms the bug exists
            pytest.fail(
                f"NaN LEAK DETECTED (buggy path - expected to fail)!\n"
                f"  Configuration: {num_tokens} real + {num_padding} padding tokens\n"
                f"  NaN placement: {nan_placement}\n"
                f"  Input shape: {x.shape}, Output shape: {output.shape}\n"
                f"  NaN in real output: {num_nan_elements}/{total_real_elements} "
                f"({nan_percentage:.2f}%)\n"
                f"  NaN in padding output: {has_nan_in_padding}\n"
                f"This confirms the hypothesis that NaNs leak from padding to real tokens."
            )
        else:
            # Should NOT fail on fixed path
            pytest.fail(
                f"NaN LEAK DETECTED (fixed path - should not happen)!\n"
                f"  The fix (NaN masking) did not work as expected.\n"
                f"  Configuration: {num_tokens} real + {num_padding} padding tokens\n"
                f"  NaN placement: {nan_placement}\n"
                f"  NaN in real output: {num_nan_elements}/{total_real_elements} "
                f"({nan_percentage:.2f}%)"
            )

    # If we reach here, NaNs are properly isolated
    path_type = "buggy" if use_buggy_path else "fixed"
    print(f"✓ NaN isolation verified ({path_type} path): {nan_placement} placement, "
          f"{num_tokens} real + {num_padding} padding tokens")


@pytest.mark.parametrize("num_tokens", [32])
@pytest.mark.parametrize("num_padding", [8])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("output_size", [1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_nvfp4_gemm_nan_masking_fix(
    num_tokens: int,
    num_padding: int,
    hidden_size: int,
    output_size: int,
    dtype: torch.dtype,
) -> None:
    """
    Test a potential fix: masking NaNs before FP4 quantization.

    This demonstrates the most efficient solution: replace NaNs with 0
    before quantization, which prevents them from contaminating block scales.
    """
    set_random_seed(42)
    device = "cuda:0"
    torch.set_default_device(device)

    total_tokens = num_tokens + num_padding

    # Create input with NaNs at padding positions (end)
    x = torch.randn(total_tokens, hidden_size, dtype=dtype, device=device)
    x[num_tokens:, :] = float('nan')

    # Create mask
    mask = torch.ones(total_tokens, dtype=torch.bool, device=device)
    mask[num_tokens:] = False

    # **FIX**: Replace NaNs with 0 before quantization
    # This is zero-cost if we piggyback on existing attention masking
    x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

    # Verify masking worked
    assert not torch.isnan(x_masked).any(), "Masking should remove all NaNs"

    # Create FP4 weights
    weight_fp4, weight_scale, weight_global_scale, weights_padding_cols = \
        create_nvfp4_weight(output_size, hidden_size, dtype, device)

    # Compute scales using masked input
    input_amax = torch.abs(x_masked).max().to(torch.float32)
    input_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / input_amax).to(
        torch.float32)
    input_global_scale_inv = 1.0 / input_global_scale
    alpha = (input_global_scale * weight_global_scale).to(torch.float32)

    # Quantize masked input
    x_fp4, x_blockscale = ops.scaled_fp4_quant(
        x_masked, input_global_scale_inv, is_sf_swizzled_layout=True,
        backend="flashinfer-cutlass")

    # Pad and run GEMM
    x_fp4_padded = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)
    output = flashinfer_scaled_fp4_mm(
        x_fp4_padded,
        weight_fp4,
        x_blockscale,
        weight_scale,
        alpha,
        dtype,
        backend="cutlass",
    )
    output = slice_nvfp4_output(output, output_size)

    # With the fix, no NaNs should appear in any position
    assert not torch.isnan(output).any(), (
        "With NaN masking before quantization, output should be NaN-free"
    )

    print(f"✓ NaN masking fix verified: no NaNs in output after masking input")


@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_nvfp4_quant_nan_in_block_scale(
    block_size: int,
    dtype: torch.dtype,
) -> None:
    """
    Test how NaN affects FP4 block scale computation.

    This isolates the quantization step to understand how NaN in a block
    affects the block's scaling factor.
    """
    device = "cuda:0"
    torch.set_default_device(device)

    # Create a tensor with one block containing NaN
    num_blocks = 4
    x = torch.randn(1, num_blocks * block_size, dtype=dtype, device=device)

    # Inject NaN into the second block
    x[0, block_size:2*block_size] = float('nan')

    # Compute global scale (will be NaN if computed from the whole tensor)
    input_amax_with_nan = torch.abs(x).max().to(torch.float32)

    # Compute global scale without NaN (using nanmax equivalent)
    input_amax_no_nan = torch.abs(x[torch.isfinite(x)]).max().to(torch.float32)

    print(f"Max with NaN: {input_amax_with_nan}")
    print(f"Max without NaN: {input_amax_no_nan}")

    # If the entire tensor's max is NaN, the global scale is NaN
    if torch.isnan(input_amax_with_nan):
        input_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX /
                             input_amax_no_nan).to(torch.float32)
    else:
        input_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX /
                             input_amax_with_nan).to(torch.float32)

    input_global_scale_inv = 1.0 / input_global_scale

    # Quantize
    x_fp4, x_blockscale = ops.scaled_fp4_quant(
        x, input_global_scale_inv, is_sf_swizzled_layout=False,
        backend="flashinfer-cutlass")

    # Check block scales
    print(f"\nBlock scales (FP8): {x_blockscale}")
    print(f"Block scales (FP32): {x_blockscale.to(torch.float32)}")

    # Check if NaN in one block contaminates neighboring blocks
    # Convert to float32 for inspection
    scales_fp32 = x_blockscale.view(torch.float8_e4m3fn).to(torch.float32)

    # The block with NaN will likely have inf or nan scale
    # Check if this contaminates other blocks
    has_nan_scale = torch.isnan(scales_fp32).any()
    has_inf_scale = torch.isinf(scales_fp32).any()

    print(f"Has NaN in block scales: {has_nan_scale}")
    print(f"Has Inf in block scales: {has_inf_scale}")

    # This test is for observation - we don't assert, just report behavior
    if has_nan_scale or has_inf_scale:
        print("⚠ NaN in input produces NaN/Inf in block scales")
    else:
        print("✓ Block scales remain finite despite NaN in input")


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running NVFP4 NaN propagation tests...")
    test_nvfp4_gemm_nan_isolation(
        num_tokens=32, num_padding=8, hidden_size=1024, output_size=1024,
        dtype=torch.bfloat16, nan_placement="end")
    test_nvfp4_gemm_nan_masking_fix(
        num_tokens=32, num_padding=8, hidden_size=1024, output_size=1024,
        dtype=torch.bfloat16)
    test_nvfp4_quant_nan_in_block_scale(block_size=16, dtype=torch.bfloat16)
    print("All tests passed!")
