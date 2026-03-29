# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration test for NVFP4 NaN handling through the full apply_nvfp4_linear path.

This verifies that the production code fix in apply_nvfp4_linear() properly
masks NaNs before quantization.
"""
import pytest
import torch
import torch.nn as nn

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    apply_nvfp4_linear,
    convert_to_nvfp4_linear_kernel_format,
    select_nvfp4_linear_backend,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="NVFP4 requires compute capability 100 or above (Blackwell+).",
        allow_module_level=True,
    )

if not has_flashinfer():
    pytest.skip(
        reason="FlashInfer is required for NVFP4 tests.",
        allow_module_level=True,
    )

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def create_nvfp4_layer(input_size: int, output_size: int, dtype: torch.dtype,
                       device: str) -> tuple[nn.Module, any]:
    """Create a mock NVFP4 linear layer for testing."""
    layer = nn.Module()
    layer.input_size_per_partition = input_size
    layer.output_size_per_partition = output_size

    # Create and quantize random weights
    weight_bf16 = torch.randn(output_size, input_size, dtype=dtype, device=device)
    weight_amax = torch.abs(weight_bf16).max().to(torch.float32)
    weight_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / weight_amax).to(torch.float32)

    # Quantize weights to FP4
    weight_fp4, weight_blockscale = ops.scaled_fp4_quant(
        weight_bf16, weight_global_scale, is_sf_swizzled_layout=True,
        backend="flashinfer-cutlass")

    layer.weight = nn.Parameter(weight_fp4, requires_grad=False)
    layer.weight_scale = nn.Parameter(weight_blockscale, requires_grad=False)

    # Global scales
    layer.weight_global_scale = nn.Parameter(weight_global_scale, requires_grad=False)

    # Input scale (will be computed per-batch, this is just placeholder)
    input_global_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    layer.input_global_scale_inv = nn.Parameter(1.0 / input_global_scale, requires_grad=False)
    layer.alpha = nn.Parameter(input_global_scale * weight_global_scale, requires_grad=False)

    # Convert to kernel format
    backend = select_nvfp4_linear_backend()
    convert_to_nvfp4_linear_kernel_format(backend, layer)

    return layer, backend


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_nvfp4_linear_with_nan_input(dtype: torch.dtype) -> None:
    """
    Test that apply_nvfp4_linear handles NaN inputs correctly.

    This is an end-to-end integration test using the production code path.
    """
    device = "cuda:0"
    torch.set_default_device(device)
    torch.manual_seed(42)

    input_size = 64
    output_size = 128
    batch_size = 4

    # Create layer
    layer, backend = create_nvfp4_layer(input_size, output_size, dtype, device)

    # Create input with NaN in some positions
    x = torch.randn(batch_size, input_size, dtype=dtype, device=device)

    # Inject NaN into token 2, block 1 (dimensions 16-31)
    x[2, 16:32] = float('nan')

    print(f"\nInput shape: {x.shape}")
    print(f"Token 2, block 1 has NaN: {torch.isnan(x[2, 16:32]).all()}")
    print(f"Other tokens clean: {not torch.isnan(x[[0,1,3]]).any()}")

    # Apply NVFP4 linear (production code path with fix)
    output = apply_nvfp4_linear(backend=backend, layer=layer, x=x, bias=None)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output token 0 (clean input): {output[0, :8]}")
    print(f"Output token 2 (had NaN input): {output[2, :8]}")

    # Check results
    has_nan = torch.isnan(output).any()
    print(f"\nHas NaN in output: {has_nan}")

    if has_nan:
        nan_percentage = 100.0 * torch.isnan(output).sum().item() / output.numel()
        pytest.fail(
            f"NaN detected in output!\n"
            f"  {nan_percentage:.1f}% of output is NaN\n"
            f"  The fix in apply_nvfp4_linear should have masked NaNs before quantization."
        )

    print("✓ No NaN in output - fix is working correctly!")


if __name__ == "__main__":
    test_nvfp4_linear_with_nan_input(dtype=torch.bfloat16)
