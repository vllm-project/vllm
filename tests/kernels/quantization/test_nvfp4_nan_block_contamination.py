# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test for NVFP4 NaN propagation within a SINGLE TOKEN when NaN appears in some
feature dimensions but not others.

This is the REAL bug: if a single token has NaN in some dimensions (e.g., from
a buggy attention output), the block scale for that block becomes NaN, which
then contaminates the ENTIRE output for that token.
"""
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm, has_flashinfer

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


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_fix", [True, False])
@torch.inference_mode()
def test_nvfp4_nan_within_token_contamination(dtype: torch.dtype, use_fix: bool) -> None:
    """
    Test that NaN in a few dimensions of a token contaminates the entire token output.

    Setup:
    - Single token with mostly clean values
    - NaN injected into ONE BLOCK of the token (e.g., dimensions 16-31)
    - This makes that block's scale = NaN
    - The entire token output becomes NaN (not just the output dimensions
      corresponding to that block)
    """
    device = "cuda:0"
    torch.set_default_device(device)
    torch.manual_seed(42)

    # Single token with hidden_size=64 (4 blocks of 16)
    x = torch.randn(1, 64, dtype=dtype, device=device)

    # Inject NaN into the SECOND BLOCK (dims 16-31) of this token
    x[0, 16:32] = float('nan')

    print(f"\nInput token:")
    print(f"  Block 0 (dims 0-15): clean, sample={x[0, 0:4]}")
    print(f"  Block 1 (dims 16-31): NaN, sample={x[0, 16:20]}")
    print(f"  Block 2 (dims 32-47): clean, sample={x[0, 32:36]}")
    print(f"  Block 3 (dims 48-63): clean, sample={x[0, 48:52]}")

    # Apply fix if requested
    if use_fix:
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        print(f"\n[FIX APPLIED] NaNs masked to zero")

    # Compute global scale
    input_amax = torch.abs(x[torch.isfinite(x)]).max().to(torch.float32)
    input_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / input_amax).to(torch.float32)
    input_global_scale_inv = 1.0 / input_global_scale

    # Quantize
    x_fp4, x_blockscale = ops.scaled_fp4_quant(
        x, input_global_scale_inv, is_sf_swizzled_layout=False,
        backend="flashinfer-cutlass")

    print(f"\nBlock scales after quantization:")
    for i in range(4):
        scale_val = x_blockscale.view(torch.float8_e4m3fn)[0, i].to(torch.float32)
        print(f"  Block {i}: {scale_val}")

    # Create weights
    output_size = 128
    weight = torch.randn(output_size, 64, dtype=dtype, device=device)
    weight_amax = torch.abs(weight).max().to(torch.float32)
    weight_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / weight_amax).to(torch.float32)
    weight_fp4, weight_blockscale = ops.scaled_fp4_quant(
        weight, weight_global_scale, is_sf_swizzled_layout=False)

    alpha = (input_global_scale * weight_global_scale).to(torch.float32)

    # Run GEMM
    output = flashinfer_scaled_fp4_mm(
        x_fp4, weight_fp4, x_blockscale, weight_blockscale, alpha, dtype,
        backend="cutlass")

    print(f"\nOutput shape: {output.shape}")
    print(f"Output sample (first 8): {output[0, :8]}")

    has_nan = torch.isnan(output).any()
    print(f"Has NaN in output: {has_nan}")

    if has_nan:
        nan_percentage = 100.0 * torch.isnan(output).sum().item() / output.numel()
        print(f"NaN percentage: {nan_percentage:.1f}%")

        if use_fix:
            pytest.fail(
                f"NaN contamination detected even with fix applied!\n"
                f"  {nan_percentage:.1f}% of output is NaN\n"
                f"  The fix should have prevented this."
            )
        else:
            pytest.fail(
                f"NaN contamination detected (expected on buggy path)!\n"
                f"  A single NaN block in the input caused {nan_percentage:.1f}% of output to be NaN\n"
                f"  This demonstrates the bug: NaN in one block contaminates the entire token output."
            )
    else:
        print("✓ No NaN contamination detected")


if __name__ == "__main__":
    print("="*60)
    print("Testing BUGGY PATH (no NaN masking)")
    print("="*60)
    test_nvfp4_nan_within_token_contamination(dtype=torch.bfloat16, use_fix=False)

    print("\n" + "="*60)
    print("Testing FIXED PATH (with NaN masking)")
    print("="*60)
    test_nvfp4_nan_within_token_contamination(dtype=torch.bfloat16, use_fix=True)
